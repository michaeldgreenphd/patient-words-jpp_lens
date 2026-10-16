"""Verified local linear-lens loading without guessing an artifact's method.

Both released ``J`` files and ``BaseLens.save`` files hold linear maps. That
algebra does not identify J-Lens, R-Lens or J++. A separately reviewed manifest
must supply that identity and exact fit revisions. Unknown revisions remain
inspectable but cannot authorize transport. No model or file is downloaded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "lens-interop/1"
METHODS = frozenset({"j_lens", "jpp", "r_lens"})


class LensInteropError(ValueError):
    """A named refusal suitable for a machine-readable boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise LensInteropError(code, detail)


def _digest(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def validate_manifest(
    manifest: Mapping[str, Any], expected_model: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate semantics and report whether fit identity is known.

    ``expected_model`` identifies the model supplying residuals, independently
    of the lens. Its identity fields must exactly match known fit fields.
    A manifest is evidence supplied by the caller, not a discovery mechanism;
    unknown fit revisions must be JSON null, never the intended run revision.
    """
    _require(isinstance(manifest, Mapping), "invalid_manifest", "expected an object")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "unsupported_schema", SCHEMA_VERSION)
    _require(isinstance(manifest.get("method"), str) and manifest["method"] in METHODS, "missing_or_unsupported_method", "explicit method required")
    _require(
        isinstance(manifest.get("method_evidence"), str) and bool(manifest["method_evidence"].strip()),
        "missing_method_evidence",
        "cite checkpoint provenance or a reviewed source; filenames alone are insufficient",
    )
    _require(manifest.get("capture_site") == "block_output_pre_final_norm", "unsupported_capture_site", "block outputs required")
    _require(manifest.get("layer_indexing") == "zero_based", "unsupported_layer_indexing", "zero-based indices required")
    _require(manifest.get("stream_reduction") == "none", "unsupported_stream_reduction", "single-stream residuals required")
    _require(manifest.get("readout") == "model_final_norm_then_lm_head", "unsupported_readout", "native final norm and head required")
    model = manifest.get("model")
    _require(isinstance(model, Mapping), "missing_model_identity", "model object required")
    assert isinstance(model, Mapping)
    _require(isinstance(model.get("id"), str) and bool(model["id"]), "missing_model_identity", "id")
    _require("tokenizer_id" in model, "missing_model_identity", "tokenizer_id: use null for unknown")
    unknown = []
    if model["tokenizer_id"] is None:
        unknown.append("tokenizer_id")
    else:
        _require(isinstance(model["tokenizer_id"], str) and bool(model["tokenizer_id"]), "missing_model_identity", "tokenizer_id")
    for key in ("width", "num_layers"):
        _require(type(model.get(key)) is int and model[key] > 0, "invalid_geometry", key)
    for key in ("revision", "tokenizer_revision"):
        _require(key in model, "missing_fit_revision", f"{key}: use null for unknown")
        if model[key] is None:
            unknown.append(key)
        else:
            _require(_digest(model[key], 40), "unpinned_fit_revision", key)
    if expected_model is not None:
        _require(isinstance(expected_model, Mapping), "missing_expected_model_identity", "object required")
        for key in ("id", "tokenizer_id", "revision", "tokenizer_revision", "width", "num_layers"):
            _require(key in expected_model, "missing_expected_model_identity", key)
            if key in ("id", "tokenizer_id"):
                _require(isinstance(expected_model[key], str) and bool(expected_model[key]), "missing_expected_model_identity", key)
            if key in ("width", "num_layers"):
                _require(type(expected_model[key]) is int and expected_model[key] > 0, "invalid_expected_geometry", key)
            if key in ("revision", "tokenizer_revision"):
                _require(_digest(expected_model[key], 40), "unpinned_expected_revision", key)
            if model[key] is not None:
                _require(model[key] == expected_model[key], "model_mismatch", key)
    layers = manifest.get("source_layers")
    _require(isinstance(layers, list) and bool(layers), "missing_source_layers", "nonempty explicit list required")
    assert isinstance(layers, list)
    _require(
        all(type(layer) is int and 0 <= layer < model["num_layers"] for layer in layers),
        "invalid_source_layers", "indices must be in the model",
    )
    _require(layers == sorted(set(layers)), "invalid_source_layers", "sorted unique indices required")
    target = manifest.get("target_layer")
    _require(type(target) is int and 0 <= target < model["num_layers"], "missing_or_invalid_target_layer", "explicit target required")
    _require(all(layer < target for layer in layers), "invalid_source_layers", "sources must precede target")
    artifact = manifest.get("artifact")
    _require(isinstance(artifact, Mapping), "missing_artifact_identity", "artifact object required")
    assert isinstance(artifact, Mapping)
    _require(_digest(artifact.get("sha256"), 64), "missing_artifact_hash", "sha256 required")
    _require(_digest(artifact.get("revision"), 40), "unpinned_artifact_revision", "commit required")
    for key in ("repository", "path"):
        _require(isinstance(artifact.get(key), str) and bool(artifact[key]), "missing_artifact_identity", key)
    if not unknown:
        _require(
            isinstance(manifest.get("fit_identity_evidence"), str) and bool(manifest["fit_identity_evidence"].strip()),
            "missing_fit_identity_evidence", "exact fit revisions require a source",
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "unsupported" if unknown else "validated",
        "reasons": [{"code": "unknown_fit_identity", "fields": unknown}] if unknown else [],
        "method": manifest["method"],
        "source_layers": list(layers),
        "target_layer": target,
        "artifact_loaded": False,
    }


@dataclass(frozen=True)
class LoadedLens:
    """Linear maps plus original metadata; no hidden model-name conversion."""

    manifest: dict[str, Any]
    matrices: dict[int, Any]
    serialization_format: str
    raw_metadata: dict[str, Any]

    def transport(self, residual: Any, layer: int) -> Any:
        """Apply ``h @ J.T`` in fp32, retaining all leading dimensions."""
        import torch

        _require(type(layer) is int and layer in self.matrices, "unfitted_layer", str(layer))
        width = self.manifest["model"]["width"]
        _require(isinstance(residual, torch.Tensor) and residual.ndim >= 1, "invalid_residual", "tensor required")
        _require(residual.shape[-1] == width, "residual_width_mismatch", str(width))
        _require(residual.is_floating_point() and bool(torch.isfinite(residual).all()), "invalid_residual", "finite floating tensor required")
        return residual.float() @ self.matrices[layer].to(device=residual.device).T

    def readout(self, residual: Any, layer: int, native_unembed: Callable[[Any], Any]) -> Any:
        """Caller supplies native final normalization and head, applied once.

        This returns lens logits. Actual model output must be captured from the
        full forward separately; the last fitted source layer is not output.
        """
        return native_unembed(self.transport(residual, layer))


def paired_readouts(
    lens: LoadedLens,
    block_outputs: Mapping[int, Any],
    forward_output: Any,
    *,
    positions: list[int],
    native_unembed: Callable[[Any], Any],
) -> dict[str, Any]:
    """Read cached residuals and independently extract actual forward logits.

    Inputs are an already-completed single-sequence forward and the caller's
    cached ``[1, sequence, width]`` block outputs. Nothing is tokenized or run
    here. The caller proves these came from the same forward/prompt/token IDs.
    """
    import torch

    logits = getattr(forward_output, "logits", None)
    _require(isinstance(logits, torch.Tensor) and logits.ndim == 3 and logits.shape[0] == 1, "missing_actual_output", "forward_output.logits [1, sequence, vocabulary] required")
    _require(logits.is_floating_point() and bool(torch.isfinite(logits).all()), "invalid_actual_output", "finite floating logits required")
    _require(
        isinstance(positions, list) and bool(positions)
        and all(type(position) is int and 0 <= position < logits.shape[1] for position in positions)
        and len(set(positions)) == len(positions),
        "invalid_positions", "explicit unique in-range zero-based positions required",
    )
    _require(isinstance(block_outputs, Mapping) and bool(block_outputs), "missing_residuals", "explicit layer map required")
    lens_logits = {}
    for layer, residual in block_outputs.items():
        _require(isinstance(residual, torch.Tensor) and residual.ndim == 3 and residual.shape[:2] == logits.shape[:2], "residual_sequence_mismatch", str(layer))
        readout = lens.readout(residual[0, positions], layer, native_unembed)
        _require(isinstance(readout, torch.Tensor) and readout.shape == (len(positions), logits.shape[-1]), "readout_shape_mismatch", str(layer))
        _require(readout.is_floating_point() and bool(torch.isfinite(readout).all()), "invalid_readout", str(layer))
        lens_logits[layer] = readout.detach().float().cpu()
    return {
        "lens_logits": lens_logits,
        "actual_output_logits": logits[0, positions].detach().float().cpu(),
        "positions": list(positions),
    }


def _check_metadata(checkpoint: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[dict[int, Any], str]:
    """Check explicit on-disk fields without inventing absent provenance."""
    _require(not ("J" in checkpoint and "parameters" in checkpoint), "ambiguous_checkpoint_format", "both J and parameters")
    model = manifest["model"]
    layers = manifest["source_layers"]
    on_disk_layers = checkpoint.get("source_layers")
    _require(isinstance(on_disk_layers, list) and all(type(layer) is int for layer in on_disk_layers) and on_disk_layers == layers, "checkpoint_layer_mismatch", "source_layers")
    if "config" in checkpoint:
        config = checkpoint["config"]
        _require(isinstance(config, Mapping), "invalid_checkpoint_config", "object required")
        _require(config.get("lens_type") == "jacobian", "missing_or_unsupported_checkpoint_method", "linear Jacobian family required")
        _require(config.get("hf_model_name") == model["id"], "model_mismatch", "checkpoint hf_model_name")
        _require(config.get("d_model") == model["width"], "checkpoint_width_mismatch", "config.d_model")
        _require("relative_end_transport_layer" in config, "missing_checkpoint_target", "relative_end_transport_layer")
        target = config["relative_end_transport_layer"]
        _require(type(target) is int and model["num_layers"] + target == manifest["target_layer"], "checkpoint_target_mismatch", "relative target")
        if config.get("source_layers") is not None:
            _require(config["source_layers"] == layers, "checkpoint_layer_mismatch", "config.source_layers")
        if "parameters" in checkpoint:
            parameters = checkpoint["parameters"]
            _require(isinstance(parameters, Mapping) and set(parameters) == {"jacobians"}, "unsupported_parameters", "one linear map family required")
            matrices = parameters["jacobians"]
            file_format = "base_lens_parameters"
        else:
            _require("J" in checkpoint, "unsupported_checkpoint_format", "parameters or J required")
            matrices = checkpoint["J"]
            file_format = "configured_J"
    else:
        _require("J" in checkpoint, "unsupported_checkpoint_format", "J required")
        _require(checkpoint.get("d_model") == model["width"], "checkpoint_width_mismatch", "d_model")
        matrices = checkpoint["J"]
        file_format = "released_J"
    _require(isinstance(matrices, dict) and all(type(key) is int for key in matrices), "invalid_matrices", "integer layer keys required")
    _require(sorted(matrices) == layers, "checkpoint_layer_mismatch", "matrix keys")
    provenance = checkpoint.get("provenance")
    if provenance is not None:
        _require(isinstance(provenance, Mapping), "invalid_checkpoint_provenance", "object required")
        for key, expected in (
            ("hf_model_name", model["id"]),
            ("target_layer", manifest["target_layer"]),
            ("capture_point", "block_output"),
            ("stream_reduce", "none"),
        ):
            if key in provenance:
                _require(provenance[key] == expected, "checkpoint_provenance_mismatch", key)
        if "method" in provenance:
            external_names = {"jpp": "jpp", "jlens": "j_lens", "j_lens": "j_lens", "rlens": "r_lens", "r_lens": "r_lens"}
            _require(isinstance(provenance["method"], str) and external_names.get(provenance["method"]) == manifest["method"], "checkpoint_method_mismatch", "method")
        elif "config_json" in provenance:
            # This legacy release field has estimator semantics; the converted
            # J++ source_config.lrp_mode field is a recipe, not a method label.
            try:
                legacy = json.loads(provenance["config_json"])
            except (TypeError, ValueError) as error:
                raise LensInteropError("invalid_legacy_estimator_metadata", "provenance.config_json") from error
            _require(isinstance(legacy, Mapping), "invalid_legacy_estimator_metadata", "object required")
            estimator = legacy.get("estimator")
            _require(isinstance(estimator, str) and estimator in {"relp", "standard"},
                     "unsupported_legacy_estimator", "explicit standard or relp estimator required")
            legacy_method = "r_lens" if estimator == "relp" else "j_lens"
            _require(manifest["method"] == legacy_method, "checkpoint_method_mismatch", "provenance.config_json.estimator")
        # The converted J++ release puts its original BaseLens config here.
        # Check it when present; keep it verbatim in raw_metadata below.
        if "source_config" in provenance:
            source = provenance["source_config"]
            _require(isinstance(source, Mapping), "invalid_checkpoint_provenance", "source_config")
            for key, expected in (
                ("hf_model_name", model["id"]), ("d_model", model["width"]),
                ("source_layers", layers),
                ("relative_end_transport_layer", manifest["target_layer"] - model["num_layers"]),
                ("model_revision", model["revision"]), ("tokenizer_id", model["tokenizer_id"]),
                ("tokenizer_revision", model["tokenizer_revision"]),
            ):
                if key in source:
                    _require(source[key] == expected, "checkpoint_provenance_mismatch", f"source_config.{key}")
        for key, model_key in (("model_revision", "revision"), ("tokenizer_id", "tokenizer_id"),
                               ("tokenizer_revision", "tokenizer_revision")):
            if key in provenance:
                _require(provenance[key] == model[model_key], "model_mismatch", key)
    return matrices, file_format


def _load_verified_checkpoint(path: str | Path, expected_sha256: str) -> Mapping[str, Any]:
    """Hash and safely load the same open file for inspection or transport."""
    _require(_digest(expected_sha256, 64), "invalid_artifact_hash", "lowercase SHA256 required")
    path = Path(path)
    try:
        import torch
    except ImportError as error:
        raise LensInteropError("dependency_unavailable", "torch is required for tensor loading") from error
    try:
        handle = path.open("rb")
    except OSError as error:
        raise LensInteropError("artifact_unavailable", str(error)) from error
    with handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        _require(digest == expected_sha256, "artifact_hash_mismatch", str(path))
        handle.seek(0)
        try:
            # Load the same open file that was hashed, not a pathname that
            # could have been replaced after the identity check.
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as error:
            raise LensInteropError("safe_checkpoint_load_failed", str(error)) from error
    _require(isinstance(checkpoint, Mapping), "unsupported_checkpoint_format", "object required")
    return checkpoint


def _inspection_metadata(value: Any, path: str) -> Any:
    """Preserve JSON-shaped metadata exactly, refusing lossy conversions."""
    import math

    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        _require(math.isfinite(value), "unsupported_inspection_metadata", path)
        return value
    if type(value) is list:
        return [_inspection_metadata(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        _require(all(type(key) is str for key in value), "unsupported_inspection_metadata", path)
        return {key: _inspection_metadata(item, f"{path}.{key}") for key, item in value.items()}
    raise LensInteropError("unsupported_inspection_metadata", f"{path}: {type(value).__name__}")


def inspect_verified_artifact(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Inspect a verified local artifact without granting transport permission.

    This deliberately needs no model-fit manifest: embedded metadata is evidence
    to inspect, not a claim to invent before opening the file. No tensor values
    are returned. Missing method, layer or fit metadata remains missing; callers
    must still use ``load_verified_lens`` and its independent identity checks.
    A verified hash does not establish model compatibility or bound memory use.
    """
    checkpoint = _load_verified_checkpoint(path, expected_sha256)

    import torch

    _require(all(type(key) is str for key in checkpoint), "unsupported_checkpoint_format", "string field names required")
    _require(not ("J" in checkpoint and "parameters" in checkpoint), "ambiguous_checkpoint_format", "both J and parameters")
    if "parameters" in checkpoint:
        parameters = checkpoint["parameters"]
        _require(isinstance(parameters, Mapping) and set(parameters) == {"jacobians"}, "unsupported_parameters", "one linear map family required")
        matrices = parameters["jacobians"]
        file_format = "base_lens_parameters"
        matrix_path = ["parameters", "jacobians"]
    else:
        _require("J" in checkpoint, "unsupported_checkpoint_format", "J or parameters.jacobians required")
        matrices = checkpoint["J"]
        file_format = "configured_J" if "config" in checkpoint else "released_J"
        matrix_path = ["J"]
    _require(isinstance(matrices, Mapping) and bool(matrices), "invalid_matrices", "nonempty layer map required")
    _require(all(type(layer) is int for layer in matrices), "invalid_matrices", "integer layer keys required")
    tensor_metadata = []
    for layer, matrix in matrices.items():
        _require(isinstance(matrix, torch.Tensor), "invalid_matrix", str(layer))
        _require(matrix.layout == torch.strided, "invalid_matrix", f"{layer}: dense tensor required")
        tensor_metadata.append({
            "layer_key": layer, "layer_key_type": "int", "shape": list(matrix.shape),
            "dtype": str(matrix.dtype), "device": str(matrix.device),
            "all_finite": bool(torch.isfinite(matrix).all()),
        })
    raw_metadata = _inspection_metadata(
        {key: value for key, value in checkpoint.items() if key not in {"J", "parameters"}}, "checkpoint",
    )
    return {
        "schema_version": "lens-artifact-inspection/1", "status": "inspected",
        "artifact_sha256": expected_sha256, "artifact_loaded": True,
        "transport_authorized": False, "model_fit_compatibility": "not_evaluated",
        "serialization_format": file_format, "checkpoint_fields": list(checkpoint),
        "matrix_path": matrix_path, "matrices": tensor_metadata, "raw_metadata": raw_metadata,
    }


def load_verified_lens(
    path: str | Path, manifest: Mapping[str, Any], expected_model: Mapping[str, Any]
) -> LoadedLens:
    """Verify a local file hash, then deserialize tensors with weights-only.

    Unknown fit identity is refused before reading the checkpoint. There is no
    fallback to unrestricted pickle or to another lens method. A trusted hash
    and manifest are still required: weights-only is not a resource-size gate.
    """
    report = validate_manifest(manifest, expected_model)
    _require(report["status"] == "validated", "unknown_fit_identity", str(report["reasons"]))
    checkpoint = _load_verified_checkpoint(path, manifest["artifact"]["sha256"])

    import torch

    matrices, file_format = _check_metadata(checkpoint, manifest)
    width = manifest["model"]["width"]
    for layer, matrix in matrices.items():
        _require(isinstance(matrix, torch.Tensor), "invalid_matrix", str(layer))
        _require(tuple(matrix.shape) == (width, width), "matrix_shape_mismatch", str(layer))
        _require(matrix.is_floating_point() and bool(torch.isfinite(matrix).all()), "invalid_matrix", str(layer))
    raw_metadata = {key: copy.deepcopy(value) for key, value in checkpoint.items() if key not in {"J", "parameters"}}
    return LoadedLens(
        copy.deepcopy(dict(manifest)),
        {layer: matrix.detach().float() for layer, matrix in matrices.items()},
        file_format,
        raw_metadata,
    )
