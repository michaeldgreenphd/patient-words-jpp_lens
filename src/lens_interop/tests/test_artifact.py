"""Tensor fixtures are tiny and synthetic; no downloaded models or lenses."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from lens_interop import LensInteropError, load_verified_lens, paired_readouts
from lens_interop.tests.test_manifest import synthetic_manifest

torch = pytest.importorskip("torch", reason="isolated tensor test environment required")


def _checkpoint(file_format: str) -> dict:
    matrices = {0: torch.eye(2), 1: torch.eye(2)}
    if file_format == "released_J":
        return {
            "J": matrices, "source_layers": [0, 1], "d_model": 2, "n_prompts": 1,
            "provenance": {
                "method": "jpp", "hf_model_name": "synthetic/toy",
                "target_layer": 2, "capture_point": "block_output", "stream_reduce": "none",
                "extra_field": "retained verbatim",
            },
        }
    checkpoint = {
        "source_layers": [0, 1],
        "config": {
            "hf_model_name": "synthetic/toy", "lens_type": "jacobian", "d_model": 2,
            "relative_end_transport_layer": -1, "source_layers": [0, 1],
            "lrp_mode": "rlens",
        },
    }
    if file_format == "base_lens_parameters":
        checkpoint["parameters"] = {"jacobians": matrices}
    else:
        checkpoint["J"] = matrices
    return checkpoint


def _save(tmp_path: Path, checkpoint: dict) -> tuple[Path, dict]:
    path = tmp_path / "synthetic_lens.pt"
    torch.save(checkpoint, path)
    manifest = synthetic_manifest()
    manifest["artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, manifest


@pytest.mark.parametrize("file_format", ["released_J", "base_lens_parameters", "configured_J"])
def test_both_formats_identity_transport_and_final_norm_parity(tmp_path: Path, file_format: str) -> None:
    checkpoint = _checkpoint(file_format)
    path, manifest = _save(tmp_path, checkpoint)
    loaded = load_verified_lens(path, manifest, manifest["model"])
    assert loaded.serialization_format == file_format
    assert loaded.manifest["method"] == "jpp"
    raw_key = "provenance" if file_format == "released_J" else "config"
    assert loaded.raw_metadata[raw_key] == checkpoint[raw_key]
    residual = torch.tensor([[2.0, -1.0], [0.5, 2.0]])
    norm = torch.nn.LayerNorm(2)
    head = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [2.0, 1.0]]))
    calls = []

    def native_unembed(value: object) -> object:
        calls.append(value)
        return head(norm(value))

    torch.testing.assert_close(loaded.transport(residual, 0), residual)
    torch.testing.assert_close(loaded.readout(residual, 0, native_unembed), head(norm(residual)))
    assert len(calls) == 1
    assert not torch.allclose(loaded.readout(residual, 0, native_unembed), head(residual))
    with pytest.raises(LensInteropError, match="unfitted_layer"):
        loaded.transport(residual, 2)


def test_safe_load_never_retries_unrestricted_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, manifest = _save(tmp_path, _checkpoint("released_J"))
    calls = []

    def refusing_load(*args: object, **kwargs: object) -> None:
        calls.append(kwargs)
        raise RuntimeError("synthetic refusal")

    monkeypatch.setattr(torch, "load", refusing_load)
    with pytest.raises(LensInteropError, match="safe_checkpoint_load_failed"):
        load_verified_lens(path, manifest, manifest["model"])
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_hash_and_unknown_identity_refuse_before_tensor_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, manifest = _save(tmp_path, _checkpoint("released_J"))

    def forbidden_load(*args: object, **kwargs: object) -> None:
        pytest.fail("torch.load must not run")

    monkeypatch.setattr(torch, "load", forbidden_load)
    manifest["artifact"]["sha256"] = "f" * 64
    with pytest.raises(LensInteropError, match="artifact_hash_mismatch"):
        load_verified_lens(path, manifest, manifest["model"])
    expected = copy.deepcopy(manifest["model"])
    manifest["model"]["revision"] = None
    with pytest.raises(LensInteropError, match="unknown_fit_identity"):
        load_verified_lens(path, manifest, expected)


@pytest.mark.parametrize("mutation,code", [
    (lambda cp: cp.pop("source_layers"), "checkpoint_layer_mismatch"),
    (lambda cp: cp["provenance"].update(method="jlens"), "checkpoint_method_mismatch"),
    (lambda cp: cp["provenance"].update(hf_model_name="synthetic/toy-base"), "model_mismatch"),
    (lambda cp: cp["provenance"].update(target_layer=1), "checkpoint_provenance_mismatch"),
    (lambda cp: cp["provenance"].update(capture_point="block_input"), "checkpoint_provenance_mismatch"),
    (lambda cp: cp["provenance"].update(source_config={"hf_model_name": "synthetic/toy-base"}), "checkpoint_provenance_mismatch"),
    (lambda cp: cp["provenance"].update(source_config={"relative_end_transport_layer": -2}), "checkpoint_provenance_mismatch"),
    (lambda cp: cp["J"].update({1: torch.eye(3)}), "matrix_shape_mismatch"),
    (lambda cp: cp["J"].update({1: torch.full((2, 2), float("nan"))}), "invalid_matrix"),
])
def test_checkpoint_conflicts_refuse(tmp_path: Path, mutation: Callable[[dict], None], code: str) -> None:
    checkpoint = _checkpoint("released_J")
    mutation(checkpoint)
    path, manifest = _save(tmp_path, checkpoint)
    with pytest.raises(LensInteropError) as caught:
        load_verified_lens(path, manifest, manifest["model"])
    # A source model conflict is reported under the checkpoint provenance code.
    assert caught.value.code in {code, "checkpoint_provenance_mismatch"}


def test_configured_format_requires_its_algebra_metadata(tmp_path: Path) -> None:
    checkpoint = _checkpoint("base_lens_parameters")
    checkpoint["config"].pop("lens_type")
    path, manifest = _save(tmp_path, checkpoint)
    with pytest.raises(LensInteropError, match="missing_or_unsupported_checkpoint_method"):
        load_verified_lens(path, manifest, manifest["model"])


def test_actual_output_is_extracted_independently_of_last_fitted_layer(tmp_path: Path) -> None:
    path, manifest = _save(tmp_path, _checkpoint("released_J"))
    lens = load_verified_lens(path, manifest, manifest["model"])
    residual = torch.tensor([[[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]]])
    actual_logits = torch.tensor([[[9.0, 8.0], [7.0, 6.0], [5.0, 4.0]]])
    result = paired_readouts(
        lens, {1: residual}, SimpleNamespace(logits=actual_logits),
        positions=[0, 2], native_unembed=lambda value: value,
    )
    torch.testing.assert_close(result["actual_output_logits"], actual_logits[0, [0, 2]])
    torch.testing.assert_close(result["lens_logits"][1], residual[0, [0, 2]])
    assert not torch.equal(result["actual_output_logits"], result["lens_logits"][1])
    with pytest.raises(LensInteropError, match="missing_actual_output"):
        paired_readouts(lens, {1: residual}, SimpleNamespace(), positions=[0], native_unembed=lambda value: value)
    with pytest.raises(LensInteropError, match="invalid_positions"):
        paired_readouts(lens, {1: residual}, SimpleNamespace(logits=actual_logits), positions=[3], native_unembed=lambda value: value)
