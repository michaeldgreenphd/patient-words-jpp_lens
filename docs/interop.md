# Explicit local lens interchange

This optional boundary is separate from the existing fitting and evaluation
recipes. It does not change their loaders, defaults or scores. Study orchestration,
prompt selection, annotations, scoring and publication remain in the calling
repository. The new module uses the repository's Apache-2.0 license; upstream
J++ and Anthropic J-Lens attribution and license files are unchanged.

`lens_interop` validates a versioned JSON manifest without importing Torch,
Transformers, or the fitting package. `load_verified_lens` additionally checks
the local checkpoint SHA256 before `torch.load(..., weights_only=True)` on CPU.
There is no unrestricted-pickle fallback, download, or model loader. Trusted
artifact hashes do not replace a memory preflight for a large tensor file.
The structural schema is `schemas/lens-interop-v1.schema.json`; the validator
also enforces cross-field geometry and identity requirements that JSON Schema
does not express here.

The manifest identifies the scientific method (`j_lens`, `jpp`, `r_lens`)
separately from the upstream algebraic `lens_type="jacobian"`. The latter alone
does not identify a method. `method_evidence` cites the actual provenance source.
The model's exact **fit** revision and tokenizer revision must be known and
supported by `fit_identity_evidence`; the intended run revision must not be
substituted for an unknown fit revision. JSON null is supported in a manifest
and returns `unsupported / unknown_fit_identity`; tensor loading then refuses.

`model` contains `id`, `revision`, `tokenizer_id`, `tokenizer_revision`, `width`,
and `num_layers`. The caller supplies these same fields independently as
`expected_model` when loading. Identity, revisions and geometry must match.
`artifact` contains `repository`, `path`, pinned `revision`, and `sha256`.
The semantic fields are:

| Field | Required meaning |
| --- | --- |
| `schema_version` | `lens-interop/1` |
| `source_layers` | Sorted unique explicit zero-based source indices |
| `target_layer` | Explicit absolute transport target index |
| `capture_site` | `block_output_pre_final_norm` |
| `layer_indexing` | `zero_based` |
| `stream_reduction` | `none`; multi-stream transport is outside this slice |
| `readout` | `model_final_norm_then_lm_head` |

The loader accepts released `J` files and configured `parameters.jacobians`
files, plus the configured top-level `J` variation. All must carry explicit
`source_layers`; every matrix must match the declared width, layer set, and
finite floating dtype. Every fitted source must strictly precede its transport
target, matching the upstream trainer. A configured checkpoint can preserve
negative indices, unsorted/duplicate entries, or `None` in its original
`config.source_layers` (also `provenance.source_config.source_layers`). For
comparison only, resolve negative indices against the model layer count, sort
and deduplicate; `None` means all layers strictly before the target. The resolved
set must match the explicit top-level layer list and matrix keys. The original
configuration is retained verbatim, and out-of-range or target-equal sources
still refuse. Files containing both top-level `J` and
`parameters` refuse as ambiguous in inspection and transport, rather than
choosing one of two potentially different maps. The configured format must identify its algebra,
relative target, and `lrp_mode`. For a released file without
`provenance.target_layer`, the manifest must name the final block
(`num_layers - 1`), matching `workspace_lens.utils.load_legacy_format_lens`.
An explicit provenance target may precede that block. Neither missing target
metadata nor matrix keys authorize an arbitrary transport target. The raw file
is not rewritten to insert this convention.
On-disk provenance, including converted J++ `source_config`,
is cross-checked where present. Missing fields are not filled into the raw
provenance: `LoadedLens.raw_metadata` preserves it unchanged.

Every present `lrp_mode` in `config`, direct provenance, or `source_config`
is checked against the upstream `LensConfig` presets. `none` denotes the
standard-gradient recipe and contradicts an R-Lens label; the RelP presets
contradict a J-Lens label. A recipe alone cannot distinguish J++'s learned
expert mixture from its pooled R-Lens reference, so J++ still requires separate
method evidence and is never relabeled from `lrp_mode`. This check covers both
standard and RelP expert-fit recipes. Unknown modes refuse, and multiple
embedded recipe fields must agree. The configured layout requires the field;
missing legacy fields are not invented.

A legacy `provenance.config_json.estimator` is also cross-checked: `relp` is
consistent with R-Lens and `standard` with J-Lens, following the upstream legacy
loader's estimator distinction. An explicit `provenance.method=jpp` retains
that distinct method, while any present `lrp_mode` must still agree with the
estimator. Other explicit method labels cannot override a contradictory
estimator. Malformed or unknown estimator metadata refuses instead of becoming
a default method. This check does not turn the converted J++
`source_config.lrp_mode=rlens` fitting recipe into an R-Lens scientific label.
Embedded tokenizer IDs in direct
provenance and `source_config` must agree with the independently supplied manifest,
just as embedded fit revisions must. Raw metadata remains unchanged.

`LoadedLens.transport(h, layer)` performs `h.float() @ J.float().T`.
`readout(h, layer, native_unembed)` calls the supplied native final normalization
and output head exactly once. The caller must establish parity of that callable
with the model's actual forward logits, including any model-specific softcap.
Readout logits are never presented as the original model's output distribution.
The caller captures actual forward output separately; the last fitted source
layer may precede both the transport target and the model's final block.
`paired_readouts` accepts cached block outputs and the separate completed
forward's `.logits`, selects the same explicit positions, and returns both
without substituting lens readouts for output. The caller proves that cached
activations and output came from identical prompt bytes and token IDs.

## Bounded single-stream topology support

A square matrix and `stream_reduction="none"` in a caller's manifest are not
independent evidence of residual topology. The upstream multi-stream fitter
(`workspace_lens/residual_streams.py` and `fitting/jacobian_fitting.py`) averages
target streams and sums source-stream gradients under a uniform-perturbation
convention. A saved `BaseLens` file need not record that reduction. This adapter
therefore refuses `unsupported_residual_topology` before deserialization for
known fit identities outside its deliberately narrow audit:

- `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`, width 2560,
  36 blocks. Its previously retrieved pinned `config.json` has SHA256
  `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a`,
  `model_type=qwen3`, `architectures=["Qwen3ForCausalLM"]`, and no custom
  `auto_map`. An independent local source inspection of Transformers 5.16.1
  `models/qwen3/modeling_qwen3.py` (SHA256
  `cbb7f2dc274c2f5592746c0dc6985ca50353efa07376f92cc922b77680a74f69`)
  confirmed that `Qwen3DecoderLayer.forward` preserves the single
  `[batch, sequence, hidden]` residual through both additions; it does not add
  or collapse a stream axis. This is an architecture audit, not a completed
  J++ locked-environment installation or real-model parity run.
- The exact `synthetic/toy` fixture identity at forty `1` characters, width 2,
  three blocks. This is a test identity only; it must never label real
  activations or a released artifact. Its tensors are constructed locally by
  tests and provide no evidence about any released model.

The tuple includes the exact model revision, width and block count: neither a
matching spelling nor dimensions authorize a different revision or family.
Adding a model requires a new independent architecture audit. In particular,
DeepSeek's saved stream-mean maps are unsupported even if a caller labels them
`none`; there is no silent averaging path. Unknown fit identity still returns
`unknown_fit_identity` before transport, with topology unevaluated. The released
Qwen J++ checkpoint's historical fit revisions remain unknown: this capability
list does not establish them. Inspection remains available for unsupported
models and reports no transport authorization. Schema shape stays unchanged;
previously accepted unaudited model identities or contradictory metadata now
fail closed, intentionally.

## Offline use

From this repository, with a separately prepared environment:

```bash
PYTHONPATH=src python -m lens_interop --manifest manifest.json
PYTHONPATH=src python -m lens_interop --manifest manifest.json --expected-model model.json --artifact /path/to/local/lens.pt
PYTHONPATH=src python -m pytest -q src/lens_interop/tests
```

`examples/interop/synthetic-manifest.json` is a metadata-only toy; its zero
checksum is a sentinel, replaced with the generated file hash in tensor tests.
`examples/interop/qwen3-4b-jpp-unknown-fit.json` records the pinned release but
deliberately returns exit 2 because fit identity is unknown. Its full local file
now matches the pinned release LFS SHA256; this establishes artifact integrity,
not fit compatibility.

Validation emits one JSON status. Exit 0 means the manifest or supplied artifact
passed the checks requested. Exit 2 means an explicit unsupported state or error.
Manifest validation alone does not load the artifact or demonstrate compatibility
with a real model. The tiny tensors generated by tests are synthetic and are not
research measurements.

## Inspection without a fit-compatibility claim

`inspect_verified_artifact(path, expected_sha256)` is a separate, inspection-only
entry point. It shares the same SHA256 check and `torch.load` call as the transport
loader: an immutable byte snapshot is hashed, then only those verified bytes
are loaded with `weights_only=True` and `map_location="cpu"`, with no unrestricted
fallback. Replacing or modifying the source file after hashing cannot substitute
different bytes during loading. It does not construct a `LoadedLens`, accept a
guessed fit revision, or weaken `load_verified_lens`'s unknown-identity refusal.
The memory preflight must allow the serialized snapshot in addition to loaded
tensors and loader overhead; this is not a resource-size gate. Inspection does
not load a model or download anything. Earlier inspection timings/memory records
predate this snapshot change and are not fresh benchmarks of it.

```bash
PYTHONPATH=src python -m lens_interop \
  --inspect /path/to/Qwen3-4B_jpp_lens.pt \
  --sha256 893f7da243ab60d0628548a8c2b3afebaa45afdb1877d1c11521fde53209722d
```

An inspection result has `schema_version="lens-artifact-inspection/1"` and
`status="inspected"`, plus `transport_authorized=false` and
`model_fit_compatibility="not_evaluated"`. Exit 0 means only that the expected
artifact was safely loaded and described. It is not a validation of a fit or
research execution. `checkpoint_fields` preserves all top-level field names;
`matrix_path` distinguishes released `J`, configured `J`, and
`parameters.jacobians`. Each matrix record preserves its integer layer key,
shape, dtype and CPU device and reports a finite-value check without converting
the tensor precision. `raw_metadata` preserves every nonmatrix field, including
embedded provenance, without supplying absent method, layer or revision fields.
Metadata outside the supported JSON-shaped types is a named refusal rather than
a lossy conversion. Originals are opened read-only and no tensors are returned.

The historical audit below predates the complete download. The engine handoff
records the current complete-artifact inspection and provenance readiness
separately from this fork's installation and transport readiness.

## Compatibility audit at fork commit 93f02cd6c27db765abb127f20eff656bb5d27340

No root or nested `AGENTS.md` was present in this fork at that commit.

- `pyproject.toml` requires Python >=3.12 and pins Transformers 5.14.1,
  Accelerate 1.12.0, Hugging Face Hub 1.26.0, NumPy 2.5.1, Pandas 3.0.5,
  Datasets 5.0.1, PyArrow 25.0.0. Torch is 2.13.0 for the CPU extra or
  2.11.0 for CUDA 12.8. `uv.lock` exists. Keep this environment separate from
  PatientWords; this slice changes none of those pins.
- `workspace_lens.utils.load_legacy_format_lens` turns every legacy matrix
  file into `JacobianLens`; missing target provenance defaults to final block.
  Its raw scientific method is therefore not retained as a distinct identity.
  `check_model_matches_config` checks model name and width, not revision.
- `workspace_lens.get_hf_model` forwards model kwargs but its tokenizer load
  does not receive the model revision. Without `device_map`, it calls `.cuda()`.
  `jlens.hf.HFLensModel` may force BOS insertion and its encoder truncates.
  A controlled run must instead load both model and tokenizer at pinned
  revisions, pretokenize without truncation, and verify token IDs/BOS behavior.
- `BaseLens.apply` records zero-based block outputs, adds the actual final
  model block even when it is not fitted, then separately applies native final
  normalization and the head. `HFLensModel.unembed` includes the model's final
  logit softcap when configured. Preserve these conventions and test parity.
- The upstream quickstart defaults to Qwen3.6-27B, the final prompt position,
  and fixed recipe layers. Those are not Qwen3-4B sampling rules.
- The readout recipe filters out baseline failures and splits individual items.
  Neither rule is appropriate for a family-split study that measures failures.
  Its punctuation filter spares candidate IDs but still needs a study-specific
  exclusion audit and raw-rank reporting.
- `probe_swap.py`/`runner.py` are a separate intervention baseline: they swap
  pinv coordinates on unit-normalized, **pre-final-norm** `W_U[token] @ J`
  directions at all positions except index 0. They do not replace observed
  native MLP channels. Near-parallel layers and concepts with no single-token
  surface are skipped and recorded. Default scales are 0.5, 1, 2, 4; the helper
  `best_scale_rows` selects a best score on evaluated data and the success table
  conditions on clean correctness. Study use must predeclare scales and
  separately retain failed clean cases and all exclusions.

The follow-up complete-file inspection on 2026-10-15 confirmed the release's
458,763,919-byte J++ artifact SHA256
`893f7da243ab60d0628548a8c2b3afebaa45afdb1877d1c11521fde53209722d`.
The adjacent configuration SHA256 is
`6cf51651e2dfaab696cf86ce9e183738b53c9d6f4e3e7add428b6b20c451f957`.
The authoritative loaded provenance says `method=jpp`, `lrp_mode=rlens`,
`hf_model_name=Qwen/Qwen3-4B`, 96 fitting prompts, block-output capture, no stream
reduction, and a relative transport target of -1. `lrp_mode` is a recipe setting,
not permission to relabel the scientific method. Matrix keys are the integers
0..34 and each is a finite float16 [2560,2560] CPU tensor. The adjacent config
identifies absolute target block 35. Both original files remain unchanged.

The complete embedded `provenance` and `source_config` contain no model-fit
revision, tokenizer identity or tokenizer-fit revision. Those remain unknown;
the fit timestamp, name and dimensions do not establish them. No intended study
revision is inserted. The original recipe attribution, paths and every metadata
field are retained in the external inspection report. A matching name and width
do not authorize a same-revision claim or transport execution. The engine
compatibility note owns the pinned source identities, report hash, measured
inspection runtime/memory and cross-repository execution plan.

## Local verification on 2026-10-15

The isolated minimal test environment used Python 3.13 and Torch 2.13.0,
Pytest 9.1.1. This was **not** a full `uv.lock` installation and did not install
Transformers, Accelerate, NumPy, Pandas, or Datasets. Its freeze was:

```text
filelock==4.2.0
fsspec==2026.9.0
iniconfig==2.3.1
Jinja2==3.1.6
MarkupSafe==3.0.4
mpmath==1.3.0
networkx==3.7
packaging==26.3
pluggy==1.7.0
Pygments==2.21.0
pytest==9.1.1
setuptools==84.0.0
sympy==1.14.0
torch==2.13.0
typing_extensions==4.16.0
```

The new adapter's 36 offline tests passed, including both serialization families,
identity transport/native final-norm parity, separately extracted actual output,
safe-loading refusal, metadata/model/hash conflicts, and an unknown-fit dry run.
The seven existing `src/jlens/tests/test_hf_layout.py` tests also passed, with one
warning about absent NumPy. Ruff passed for `src/lens_interop`. That initial pass loaded no released checkpoint tensor, model weight, fitting
run, or real-model forward. The later complete-artifact inspection above loaded
only the lens checkpoint; it performed no model forward or fitting.
The full upstream test suite remains unrun in its locked environment.
An attempted run of existing `test_lenses.py` and `test_utils.py` stopped with
two collection errors because `transformers` is absent; those were environment
blocks, not passing regression checks.


The inspection-only follow-up adds offline coverage for all three accepted
serialization layouts, retained raw provenance and integer layer mapping,
unchanged original bytes, unknown-fit transport refusal after inspection,
pre-deserialization hash rejection, weights-only loading without fallback,
malformed objects/metadata, and CLI operation without an invented fit identity.

The follow-up checks passed 55 tests in 2.08 seconds, including the seven existing
HF-layout tests. Ruff and `git diff --check` passed. The full upstream suite still
requires its separate locked environment; no package pins were changed.
