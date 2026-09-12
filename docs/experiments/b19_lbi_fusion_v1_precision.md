# LBI server evidence and explicit fuse precision

The server package confirms a **raw candidate failure before top-k** at commit
`bc090ac0a01dc9c406b40fd45e9b9dcdae0764fe`. Candidate-identity auditing remains necessary, but sorting alone does
not explain this failure. This document supersedes the earlier repair's lack of server evidence.

## Immutable evidence

The local file `lbi_fuse_failure_20260912_043011.tar.gz` matches the supplied original package SHA256:

```text
6d834be5da6cab854c14a99a72a20beedde6ac36930ab418e9610546a0af4101
```

All 64 regular files were extracted outside the repository after rejecting absolute/traversal paths, links, special
files and duplicate names. The archive remains intact. Archived Python and full-model pickles were not executed.
Source/state/raw/layer tensors were read using `weights_only=True, map_location="cpu"`. The six recorded source hashes
match the archived source. The supplemental `LBI_Fuse_Server_Evidence_Report.md` was read; its separate probe script,
machine summary and reduced binary package were not present at implementation time. Complete original fixtures were
available and used instead of random replacement inputs.

Independent readback confirms the following original CUDA results, with unchanged `atol=rtol=1e-4`:

| Variant     | Raw boxes maximum error | Outside / 2100 | First captured failing block | Changed final ranks |
| ----------- | ----------------------: | -------------: | ---------------------------- | ------------------: |
| native      |    0.016036033630371094 |           1402 | model.2.cv1                  |                  57 |
| lbi_zero    |    0.016036033630371094 |           1402 | model.2.cv1                  |                  57 |
| lbi_nonzero |     0.01635909080505371 |           1384 | model.2.cv1                  |                  47 |

`model.2.cv1` is native Conv-BN-SiLU before LBI. Its native/zero error is `0.051792144775390625`, with 19,841 elements
outside tolerance, while captured model.0 and model.1 have none. Raw boxes are regression outputs, not pixel coordinates.
Decoded same-candidate box errors are about 0.2199/0.2199/0.2617 pixels; positional final-table errors are about
144.8/144.8/145.8. Thus numerical changes and sorting amplification coexist. The selected sets happen to match for
these fixtures; the implementation retains general boundary-set checks.

Within each original device group, all three inputs and common source/folded states match exactly. Native and zero LBI
have exact raw/decoded/final equality, separately before and after fuse. The original CPU and CUDA groups used different
inputs. The new replay uses the **CUDA group's saved input and weights on both local devices**.
For every variant/device, 192 standard Conv-BN folded weight/bias tensors were independently checked against the
FP64 folding formula cast to FP32: no violations, maximum parameter difference `1.9073486328125e-6`.

## Actual local controls and their limit

Local replay used Python 3.11.15, torch 2.7.1+cu118, cuDNN 90100 and RTX 2060 (compute capability 7.5), without upgrades.
The recorded server used Python 3.12.3, torch 2.8.0+cu128, cuDNN 91002 and RTX 4090.

Using the same saved model.1 output for both original Conv-BN-SiLU and saved fused Conv-SiLU:

| Execution      | cuDNN allow_tf32 |            Block maximum error | Outside / 51200 |
| -------------- | ---------------- | -----------------------------: | --------------: |
| Local CPU      | True / False     | 5.340576171875e-5 in both arms |               0 |
| Local RTX 2060 | True / False     | 3.814697265625e-5 in both arms |               0 |

Full models restored from all three complete source fixtures also passed all raw/decode/identity checks in both arms:

| Execution                   |    native raw-box max |      zero raw-box max |  nonzero raw-box max |
| --------------------------- | --------------------: | --------------------: | -------------------: |
| Local CPU, either flag      |  2.956390380859375e-5 |  2.956390380859375e-5 | 2.956390380859375e-5 |
| Local RTX 2060, either flag | 1.8358230590820312e-5 | 1.8358230590820312e-5 |     1.52587890625e-5 |

All have zero raw-box violations. RTX 2060 lacks Ampere TF32 execution, so changing the permission flag here does not
establish the RTX 4090 causal effect. **Recorded-server GPU A/B: NOT RUN. Native server B32/640: NOT RUN. Formal
training: NOT STARTED.** No claim is made that the original server conditions now pass. The evidence supports an
explicit precision protocol and reproducible pending server controls; server causal closure remains outstanding.

## Explicit protocol and rejection conditions

The default single `fuse_audit` remains a strict audit. Lifecycle now calls `fuse_precision_checks` with the same
native, zero and nonzero fixtures and input:

1. Run and retain the native precision arm. Run a second arm with cuDNN TF32 disabled. Both folding and both forwards
   occur inside their arm's context, in eval FP32 with autocast disabled. Every arm starts from a fresh unfused copy.
   Inputs, full source states/attributes and folded states must match exactly across arms. Shared native weights and
   zero/nonzero output projection identities are also checked.
2. Explicit FP32 raw, decode, union-candidate and postprocess checks must all pass at the original tolerance. A failure
   always blocks preflight. Original-precision raw failures remain `passed=false` in their own `audit.json`, with
   `native_precision_raw_close=false` in the protocol receipt. Numerical statistics are collected before the numerical
   gate, so exact gather/top-k/boundary checks still run and have an independent verdict. Non-finite, malformed,
   state and postprocess failures are never eligible for precision attribution.
3. If native precision fails, require a TF32-capable CUDA device, recorded TF32 permission, deterministic execution,
   failing native and zero controls, the same first failing block before layer15, and exact shared prefix outputs
   through layer14 for both LBI fixtures. A nonzero-only failure or a different first failing block blocks preflight.
   Require exact cross-arm folded states and strict FP32 success for nonzero LBI as well as the controls. Repeat the
   original arm from fresh source copies after the FP32 arm and require identical raw/decoded outputs. This detects
   unrelated state changes and non-repeatable behavior; native failures are not relabeled as native strict PASS.

This is a controlled convolution-precision classification, not identification of a particular CUDA instruction or a
claim about training accuracy. The native-failure classification path has not been exercised on the recorded server
GPU here; the new runtime gates will reject an unsupported attribution rather than use a CPU-only result to waive it.

`fuse_precision` uses the legacy 2.8 APIs. It snapshots full backend conditions and ambient autocast, synchronizes CUDA
before switching and before restoration, and uses `finally` to restore cuDNN permission and grouped matmul permission /
`get_float32_matmul_precision()` state. Normal and exceptional exits must match the original snapshot. The verifier
also checks the original complete backend snapshot immediately before the native B32 stage. No shell/profile override,
newer precision API, dependency update, production cast or permanent TF32 change is introduced.

## Replaying the immutable original fixtures

After safe extraction outside WORK, this read-only tool runs the isolated block and all complete CUDA-source models:

```bash
python tools/experiments/replay_lbi_fuse_precision.py \
  --evidence-root /absolute/path/to/validated/extraction \
  --device cuda:0 \
  --output /absolute/path/to/new/replay_directory
```

Use `--device cpu` only for an explicitly labeled CPU replay. Requested CUDA never silently falls back to CPU. A reduced
first-block package is not accepted as a complete model. The tool records actual device, versions, backend differences,
source hashes, restoration and separate native/FP32 verdicts; a differing runtime is labeled `NOT RUN` for recorded-server
A/B. The command performs no training.

Each lifecycle keeps new evidence under `fuse_precision/{native,explicit_fp32,native_repeat}/<variant>/` and
`precision_checks.json`. The repeat arm exists only when native numerical failure needs attribution. Existing
`fuse_native/` failure directories remain untouched. Input/state, all candidates, indices, classes and full layer outputs
are detached CPU clones saved before assertions; failures retain complete tracebacks. Binary evidence stays outside Git.

Regressions cover CPU/CUDA normal/exception restoration from highest/high/medium matmul settings, ambient autocast,
actual full-model native/zero/nonzero checks, saved server first-block and nonzero-state restoration, folded-bias errors
in both arms or only native, and nonzero-only forward anomalies. Prior candidate, MuSGD, AMP, reload, EMA and archive
tests remain. `LBI_SERVER_EVIDENCE` is an optional **test-only** path to the validated extraction; missing fixtures are
explicitly skipped, never synthesized as server evidence.

The production model, b19 recipe and fixed server/run/finish entry are unchanged. A fresh formal trainer is still
constructed only after independent preflight succeeds. This task does not execute the formal `train` entry.

Deleted: lifecycle's implicit single-precision fuse loop and early numerical abort that prevented postprocess diagnostics.
Reused: native fusion/inference, existing candidate audit, backend snapshot, state/receipt and packaging utilities.
Net additions are required for explicit reversible precision, original-data replay and independently gated native
diagnostics; deletion alone cannot provide those controls. No tolerance or global common assertion changed.

PyTorch 2.8 documents the distinct matmul/convolution controls in
[set_float32_matmul_precision](https://docs.pytorch.org/docs/2.8/generated/torch.set_float32_matmul_precision.html),
[numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html) and
[CUDA semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html). The pinned local and archived source, rather than
newer online APIs, determines this implementation.
