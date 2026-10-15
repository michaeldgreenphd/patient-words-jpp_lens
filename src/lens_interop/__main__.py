"""Offline manifest validation; no model loading or network calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact import LensInteropError, load_verified_lens, validate_manifest


def main() -> int:
    """Emit one structured status and a nonzero exit code for refusals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-model", type=Path)
    parser.add_argument("--artifact", type=Path, help="Optional local artifact to hash and safely validate")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        expected = json.loads(args.expected_model.read_text()) if args.expected_model else None
        report = validate_manifest(manifest, expected)
        if args.artifact is not None:
            if expected is None:
                raise LensInteropError("missing_expected_model_identity", "--expected-model is required with --artifact")
            loaded = load_verified_lens(args.artifact, manifest, expected)
            report.update(artifact_loaded=True, serialization_format=loaded.serialization_format)
    except LensInteropError as error:
        report = {"status": "error", "code": error.code, "detail": error.detail}
    except (OSError, ValueError, TypeError) as error:
        report = {"status": "error", "code": "invalid_manifest_file", "detail": str(error)}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
