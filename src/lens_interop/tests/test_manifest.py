"""Synthetic manifest checks; these require no tensor or model dependency."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lens_interop import LensInteropError, validate_manifest


def synthetic_manifest() -> dict:
    """A toy identity; not evidence about any released model or lens."""
    return {
        "schema_version": "lens-interop/1",
        "method": "jpp",
        "method_evidence": "synthetic test fixture",
        "fit_identity_evidence": "synthetic test fixture",
        "model": {
            "id": "synthetic/toy", "revision": "1" * 40,
            "tokenizer_id": "synthetic/toy", "tokenizer_revision": "1" * 40,
            "width": 2, "num_layers": 3,
        },
        "artifact": {
            "sha256": "0" * 64, "revision": "2" * 40,
            "repository": "synthetic/lenses", "path": "toy.pt",
        },
        "capture_site": "block_output_pre_final_norm",
        "layer_indexing": "zero_based", "stream_reduction": "none",
        "readout": "model_final_norm_then_lm_head",
        "source_layers": [0, 1], "target_layer": 2,
    }


@pytest.mark.parametrize("field,code", [
    ("method", "missing_or_unsupported_method"),
    ("source_layers", "missing_source_layers"),
    ("target_layer", "missing_or_invalid_target_layer"),
    ("capture_site", "unsupported_capture_site"),
    ("method_evidence", "missing_method_evidence"),
])
def test_required_semantics_are_never_guessed(field: str, code: str) -> None:
    manifest = synthetic_manifest()
    manifest.pop(field)
    with pytest.raises(LensInteropError) as caught:
        validate_manifest(manifest)
    assert caught.value.code == code


def test_unknown_fit_revisions_are_explicitly_unsupported() -> None:
    manifest = synthetic_manifest()
    expected = copy.deepcopy(manifest["model"])
    manifest["model"]["revision"] = None
    manifest["model"]["tokenizer_revision"] = None
    manifest.pop("fit_identity_evidence")
    report = validate_manifest(manifest, expected)
    assert report["status"] == "unsupported"
    assert report["reasons"] == [{"code": "unknown_fit_identity", "fields": ["revision", "tokenizer_revision"]}]
    assert manifest["model"]["revision"] is None


@pytest.mark.parametrize("key,value", [
    ("id", "synthetic/toy-base"), ("tokenizer_id", "synthetic/other"),
    ("revision", "3" * 40), ("tokenizer_revision", "3" * 40),
    ("width", 4), ("num_layers", 4),
])
def test_same_width_alone_cannot_establish_identity(key: str, value: object) -> None:
    manifest = synthetic_manifest()
    expected = copy.deepcopy(manifest["model"])
    expected[key] = value
    with pytest.raises(LensInteropError, match="model_mismatch"):
        validate_manifest(manifest, expected)


@pytest.mark.parametrize("layers", [[1, 0], [0, 0], [False], [3], [], None])
def test_layer_metadata_must_be_explicit_valid_and_unique(layers: object) -> None:
    manifest = synthetic_manifest()
    manifest["source_layers"] = layers
    with pytest.raises(LensInteropError):
        validate_manifest(manifest)


def test_malformed_method_is_a_named_refusal() -> None:
    manifest = synthetic_manifest()
    manifest["method"] = ["jpp"]
    with pytest.raises(LensInteropError, match="missing_or_unsupported_method"):
        validate_manifest(manifest)


def test_cli_dry_run_requires_no_model_libraries(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = synthetic_manifest()
    path.write_text(json.dumps(manifest))
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    result = subprocess.run(
        [sys.executable, "-m", "lens_interop", "--manifest", str(path)],
        capture_output=True, text=True, env=env, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["artifact_loaded"] is False
    manifest["model"]["revision"] = None
    path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, "-m", "lens_interop", "--manifest", str(path)],
        capture_output=True, text=True, env=env, check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "unsupported"


@pytest.mark.parametrize("layers", [[2], [0, 2]])
def test_sources_must_strictly_precede_transport_target(layers: list[int]) -> None:
    manifest = synthetic_manifest()
    manifest["source_layers"] = layers
    with pytest.raises(LensInteropError, match="invalid_source_layers"):
        validate_manifest(manifest)


@pytest.mark.parametrize("key,value", [
    ("id", "synthetic/other"), ("revision", "3" * 40),
    ("width", 4), ("num_layers", 4),
])
def test_topology_requires_audited_exact_identity(key: str, value: object) -> None:
    manifest = synthetic_manifest()
    manifest["model"][key] = value
    with pytest.raises(LensInteropError, match="unsupported_residual_topology"):
        validate_manifest(manifest, manifest["model"])


def test_reviewed_qwen_topology_is_not_a_fit_identity_substitute() -> None:
    manifest = synthetic_manifest()
    manifest["model"] = {
        "id": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "tokenizer_id": "Qwen/Qwen3-4B", "tokenizer_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "width": 2560, "num_layers": 36,
    }
    manifest["fit_identity_evidence"] = "Synthetic metadata assertion only; no released fit claim"
    assert validate_manifest(manifest)["status"] == "validated"
    manifest["model"]["revision"] = None
    report = validate_manifest(manifest)
    assert report["status"] == "unsupported"
    assert report["reasons"] == [{"code": "unknown_fit_identity", "fields": ["revision"]}]


def test_recipe_presets_match_the_audited_upstream_literal() -> None:
    import ast

    from lens_interop.artifact import _LRP_MODES

    source = Path(__file__).resolve().parents[2] / "workspace_lens/lrp/types.py"
    tree = ast.parse(source.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "LrpMode" for target in node.targets))
    assert isinstance(assignment.value, ast.Subscript)
    assert set(ast.literal_eval(assignment.value.slice)) == _LRP_MODES
