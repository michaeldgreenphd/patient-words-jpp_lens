"""Small, explicit artifact boundary; importing it does not load model libraries."""

from .artifact import (
    LensInteropError,
    LoadedLens,
    load_verified_lens,
    paired_readouts,
    validate_manifest,
)

__all__ = ["LensInteropError", "LoadedLens", "load_verified_lens", "paired_readouts", "validate_manifest"]
