# B32/P3 module AMP update audit

The deployed base `5148916e2094ddae6d231b11d4e0f64f68fb1c3d` still used autocast followed by plain
`loss.backward()` in `module_checks`. That function was text-identical to the corresponding function at
`bc090ac0a01dc9c406b40fd45e9b9dcdae0764fe`. The separate synthetic detection/MuSGD path already used GradScaler.
This repair concerns the isolated ordinary-SGD module fixture, not candidate sorting or the native training recipe.

## Reference evidence and local reproduction

The supplied `LBI_B32_AMP_Probe_Evidence.zip` has SHA256:

```text
c04ab9e35a00e467a7cc28d11927575b4ae86793021ace2fb2a0ba3064bc9257
```

All eight files were extracted to an ignored evidence directory after validating paths and rejecting links/special
files. README, probe, old sources, all three result JSON files and the hash manifest were read. All recorded hashes
match; the old module matches the current production module exactly. The archived code was not executed or copied
over the verifier. The package's results are **reference CPU PyTorch 2.10.0+cpu measurements**, not local or server runs.
Its FP32 and scaled FP16 arms pass, while unscaled FP16 leaves all three upstream gradients and deltas at zero.

The unchanged current `module_checks("cuda:0", True, spatial=(80,80), batch=32)` was then actually run on local
RTX 2060, Python 3.11.15, torch 2.7.1+cu118, cuDNN 90100, four intra-op threads. The GPU was checked before work
(0% utilization, about 473 MiB used); no other process was killed and no environment or precision overrides were added.
The original assertion failed after all three attempts, with `proj_l.weight`, `proj_s.weight`, and `dw.weight` having
zero gradients and zero deltas throughout. Only `out.weight` updated. A diagnostic observer saved the exact original
initial module state, S/L/target tensors, RNG and failed state before any code edit.

The saved full B32/80x80 fixture SHA256 is:

```text
e7dc60388a23718bce2516b20ca6c34d7cc9a8cdd68b3dd9a44d76927650a6c1
```

Three independent local CUDA controls reused that same fixture, SGD(lr=0.01, weight_decay=0), unchanged
`(output.float() * target).mean() * 100` loss, zero output projection and exactly three attempted steps:

| Local CUDA mode            | Four-parameter update gate | First effective update attempts: out / proj_l / proj_s / dw |
| -------------------------- | -------------------------- | ----------------------------------------------------------- |
| FP32                       | PASS                       | 0 / 1 / 1 / 2                                               |
| FP16 autocast, no scaling  | FAIL                       | 0 / missing / missing / missing                             |
| FP16 autocast + GradScaler | PASS                       | 0 / 1 / 1 / 2                                               |

Attempt numbering starts at zero. In the scaled arm, scale stayed at 65536, all three optimizer steps executed,
and the last attempt had these **unscaled** gradient/update statistics:

| Parameter     |      Gradient max_abs | Parameter max_abs_delta | Changed elements |
| ------------- | --------------------: | ----------------------: | ---------------: |
| proj_l.weight | 3.0919909477233887e-7 |    3.725290298461914e-9 |              518 |
| proj_s.weight | 2.8568319976329803e-7 |    3.725290298461914e-9 |              164 |
| dw.weight     |  5.722977221012115e-7 |  2.3283064365386963e-10 |                1 |
| out.weight    | 0.0012874603271484375 |   1.2874603271484375e-5 |             2048 |

At attempt 1, dw already had a nonzero gradient but no observable FP32 parameter change. It only satisfies the
unchanged gate at attempt 2; finite/nonzero gradients alone were not accepted. A read-only activation-gradient hook
found out's input gradient zero throughout the unscaled arm, versus unscaled maxima about 4.8203e-10 and 9.6134e-10
on attempts 1 and 2 with scaling. This supports FP16 gradient underflow in the local reproduced path.
**Original AutoDL RTX 4090/PyTorch 2.8 reproduction: NOT RUN.** The screenshot alone does not identify its missing
parameters, and this local reproduction does not establish the server's unique cause.

## Repair and retained gates

`module_checks` delegates its existing update loop to `module_updates`, which owns the loop and its failure receipts.
CUDA AMP creates one standard `torch.amp.GradScaler("cuda")` for the whole fixture, scales before backward, unscales
once before gradient inspection, then calls scaler.step/update. A local optimizer post-step hook observes actual SGD
execution; SGD's usual `None` return is not interpreted as a skip. FP32 disables the scaler. CPU default BF16 is not
reported as the same FP16 experiment.

The optimizer, loss, learning rate, zero initialization and three-attempt budget are unchanged. The existing staged
gradient audit is reused: the output projection must unlock before any upstream task update, and every parameter
must have finite nonzero task gradients and actual parameter changes on a successful step within the fixed budget.
Because module SGD has zero decay and no momentum, no decay/momentum-only motion can satisfy this fixture's gate.
Its receipt reuses the `unscaled_before_clip` field name from the staged audit schema; this module fixture does not clip.

Inf gradients are recorded and allowed to trigger the scaler's normal skipped step and backoff. Skips consume the
same three-attempt budget, count as no successful update, and must leave parameters unchanged. No loop is extended
until it passes. Disconnected gradients, omitted optimizer parameters, lost updates, or exhausted budgets still fail
with parameter names. No model, detection loss, MuSGD, training precision, b19 arguments, native 64-batch budget,
server/run/finish entry or prior fuse implementation changes.

Actual main calls write under `module_cpu/`, `module_cuda/`, `module_amp/`, and `module_B32_P3/` in the existing preflight
directory. `updates.json` and `summary.json` are written before update/staged assertions. They contain input shapes,
dtypes/strides, versions/device/GPU, seed/commit/source hashes, unchanged loss and values, parameter membership and
dtype, attempted/successful steps, scaler values/skips, unscaled gradients/nonzero counts, delta/changed elements,
first gradient/update attempts, missing names, precision conditions and traceback.

On failure, `failure.pt` additionally retains CPU initial/current state, S/L/target, parameter gradients, optimizer/scaler
state and the relevant CPU/CUDA torch RNG states. Only detached local CPU evidence is saved. New directories cannot
overwrite previous evidence. Backend/autocast are compared at the fixture boundary; main checks the original backend
again between the module test and native detection B32 preflight. Prior fuse precision recovery remains intact.

## Regression and scope

Tests cover actual CPU/CUDA module calls, a persistent scaler with one unscale per attempt, staged zero initialization,
real injected overflow with both recoverable and budget-exhausting skips, detached/zero upstream gradients, omitted
optimizer membership, nonzero gradients with deliberately lost updates, and failure artifacts with missing names.
The preceding fuse precision context is checked before a following AMP fixture. Optional `LBI_B32_FIXTURE` test input
enables full saved B32 controls and the complete B32 identity/RMS/update path; absence is explicitly skipped. The full
path verifies that its initial state and S/L/target still match the original captured fixture exactly. Unscaled FP16 is
a diagnostic control, not a requirement that every GPU must fail.

Ordinary-SGD module checks, local synthetic B2/160 detection/MuSGD checks, real server B32/640 detection preflight,
and formal training have separate receipts. Local tests cannot certify the latter two. **Server native B32/640:
NOT RUN. Formal training: NOT STARTED.** Existing candidate/fuse/precision/reload/EMA/package tests remain applicable.

Deleted: the module's unscaled backward/step loop and its anonymous final all-parameter assertion.
Reused: native AMP/SGD, tensor statistics, staged-gradient gates, snapshots and JSON provenance utilities.
Net additions are necessary because the formerly discarded per-attempt failure evidence and independently replayable
update fixture cannot be supplied by deletion alone.

The fixed-version [PyTorch 2.8 AMP examples](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html) describe
scale/backward/unscale/step/update ordering, while the [AMP reference](https://docs.pytorch.org/docs/2.8/amp.html)
explains why a final FP32 loss does not prevent earlier FP16 backward underflow. These principles support, but do not
replace, the saved measurements above.
