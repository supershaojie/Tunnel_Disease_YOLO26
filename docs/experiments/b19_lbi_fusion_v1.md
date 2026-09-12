# b19 LBI-Fusion v1

This is a single-layer YOLO26n tunnel-crack experiment. Accuracy, recall and localization gains are hypotheses,
not established results. Formal AutoDL training is **NOT STARTED** by the implementation task.

## Identity and provenance

| Item                                  | Fixed value                                                        |
| ------------------------------------- | ------------------------------------------------------------------ |
| Baseline source                       | `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`                         |
| Branch                                | `codex/exp-yolo26n-b19-lbi-fusion-v1`                              |
| RUN                                   | `yolo26n_b19_lbi_fusion_v1`                                        |
| Baseline RUN                          | `b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42`            |
| Module                                | `Concat_LBI_Fusion`                                                |
| Model                                 | `ultralytics/cfg/models/26/yolo26n-lbi-fusion-v1.yaml`             |
| Original initialization weight SHA256 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| Complete historical b19 args SHA256   | `b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9` |
| User module ZIP SHA256                | `a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf` |

The local Git object, native YAML, original checkpoint and CCA v2 archive provenance agree on the canonical b19
source. The complete `run/provenance/b19_original_args.yaml` and expanded launcher were read from the user's
CCA v2 archive. All 112 configuration fields are compared; no current defaults may replace historical settings.
The archived launcher is an expanded historical record, not the missing original shell file.

The six requested ZIP files were read and hashed. See [reference audit](lbi_reference_audit.json).
Only their list-input interface, channel organization and native registration conventions informed the implementation.
IIA axial attention, DPCF mixing/resizing and RLAB upsampling/QKV structures were not copied. Custom-file licensing
and publication/performance claims were not independently established. Local paths and the full ZIP inventory stay
in ignored `runs/lbi_local/reference_audit.json` and `reference_zip_inventory.txt`; the ZIP itself is never committed.

Neutral tools were selectively ported from FDV
`d0b9d6feeb4a125f2dc44d110edd24340a84959f` and DCS
`d4a3b83940bdb306d008ede322b1e45bcd08cafb`.
Exact source files, hashes and adaptations are in [utility sources](lbi_utility_sources.json).
No other experiment's model, YAML, loss or optimizer implementation was merged. Prior DCS/FDV server success or
accuracy is not assumed from these tool sources.

## Fixed computation

Layer15 retains `f=[-1,4]`: S is layer14's upsampled semantic input and L is layer4's P3 detail input.
For nano/640 the parser channels are `[128,128]`; the result remains `[B,256,80,80]`.
Layer16 still consumes 256 channels. Layers9/10/12, every other Concat and Detect `[16,19,22]` remain native b19.

```text
u = Conv1x1_L(L), v = Conv1x1_S(S)                     # rank16, no bias
un = u / sqrt(mean_channel(u²) + 1e-4)
vn = v / sqrt(mean_channel(v²) + 1e-4)
b = un * vn
t = SiLU(DW3x3(b))                                    # stride1, padding1, dilation1, zeros
r = Conv1x1_out(t).to(L.dtype)
output = cat([S, L + r], dim=1)
```

Projection convolutions follow the native autocast context. Squares, channel mean, rsqrt and interaction are
explicitly FP32 via the existing `autocast` helper, then the interaction returns to projection dtype. Denominators
remain attached to autograd. There is no interpolation, cropping, gate, affine normalization, bias or auxiliary loss.
S is not cast or updated; native AMP may yield FP32 S and FP16 L, and native concatenation promotion is retained.

The stateless `torch.nn.functional.silu(..., inplace=False)` implements exactly the specified non-inplace SiLU.
This replaces the parameter-free `nn.SiLU` container because b19's native `initialize_weights` and
`load_checkpoint` force registered activations' `inplace` flags to true. No shared initialization/load policy was edited.
This is an implementation adaptation, not a mathematical change.

| New tensor               | Shape                        | Parameters |
| ------------------------ | ---------------------------- | ---------: |
| `model.15.proj_l.weight` | `[16,128,1,1]`               |       2048 |
| `model.15.proj_s.weight` | `[16,128,1,1]`               |       2048 |
| `model.15.dw.weight`     | `[16,1,3,3]`                 |        144 |
| `model.15.out.weight`    | `[128,16,1,1]`               |       2048 |
| Total                    | Four tensors; no new buffers |       6288 |

The convolution-only budget at 80×80 is 0.0804864 GFLOPs (two operations per MAC), excluding RMS, elementwise
operations and memory movement. Whole-model THOP profiling is recorded separately; a small parameter count does not
establish lower latency.

The three upstream convolutions retain ordinary Conv2d initialization; only `out.weight` is zeroed at construction.
`fork_rng(devices=[])` isolates initialization of the four new CPU-created weights, restoring the native RNG stream
before later layers are created. The verifier checks both RNG equality and every shared state_dict tensor, including
BN buffers and both Detect branches. No output reset exists in forward/load/EMA/reload/fuse/predict.

## Validation and training ownership

Zero output projection starts at exact `cat(S,L)`. Its first data gradient can be nonzero while the upstream data
gradients are zero. The verifier requires an observable task-driven output update and later observable task-driven
updates in all three upstream tensors. Native MuSGD replay and an otherwise identical zero-task-gradient replay
separate task updates from decay/momentum motion. Rounding-only changes remain incomplete unless separately proven;
they are never silently accepted.

The independent server verifier uses the actual `BaseTrainer._do_train` loop. Native parameter grouping, warmup,
accumulation, scaling, unscale, clipping, MuSGD and EMA remain in the original trainer. Optimizer hooks only observe
the legal step boundary. A dedicated callback exception ends preflight after all stages pass, or fails at the fixed
64 native B32 batch limit. Accumulation without step, AMP overflow skips and actual optimizer steps are recorded
separately. Failure evidence is saved. No LR, batch, assertion, CUDA/TF32/CUBLAS override is introduced to pass tests.
The original `init_seeds` policy is reused unchanged.

`train` always invokes preflight in a separate Python process. After a matching PASS receipt it constructs a fresh
trainer, source model, optimizer, scaler, EMA, RNG and DataLoader from seed42. Preflight state is not inherited.
Original `yolo26n.pt` is the sole initialization source; b19 trained best.pt is used only for evaluation comparison.

The original 200 epochs, patience60, B32, 640, workers8, device0, MuSGD, AMP, nbs64, scaled decay and all augmentations
are retained. Config differences are limited to model and audited equivalent source/output identity paths. The existing
dataset is fingerprinted without splitting or augmenting it again. Inherited split limitations are not disproved by
matching the recipe.

Local tests cover CPU FP32, available CUDA FP32/AMP, rectangular noncontiguous inputs, zero/tiny/mixed-sign RMS,
parser and topology, exact whole-model train/eval state/output, both Detect branches, native detection loss with
local synthetic batches, staged native MuSGD updates, FP32 state_dict/checkpoint reload in a fresh process, native
FP16 save/load controls, EMA and fused inference. A separate local runtime is not AutoDL evidence.
Exact zero-init tolerance is `(0,0)`; independent division-vs-rsqrt formula tests use `(1e-5,1e-5)` FP32 and
`(1e-3,1e-3)` AMP. Fuse uses fixed `(atol=1e-4, rtol=1e-4)` for every retained one2one raw and decoded candidate in an
explicit FP32 diagnostic context, while preserving separate native-precision diagnostics and restoring original backend
conditions before native B32. See the [precision protocol and server evidence](b19_lbi_fusion_v1_precision.md).
Final detections are checked by actual candidate identity, exact gather and top-k correctness, including boundary
changes; positional differences remain diagnostic evidence. See the [fuse audit](b19_lbi_fusion_v1_fuse_audit.md).
FP16 quantization error is reported independently, with equal-quantization round trips checked exactly.
The isolated module SGD test uses standard CUDA AMP scaling within its original three-attempt budget; see the
[B32/P3 module update evidence and failure receipts](b19_lbi_fusion_v1_module_amp.md).

Length, channels, batch/spatial dimensions, devices and non-floating inputs give explicit errors. When Cs=Cl,
tensor metadata cannot identify a semantic reversal; the complete ordered graph and distinguishable-input tests
establish S/L provenance. The module never guesses a source from tensor values.

## Evaluation, diagnostics and packaging

Validation selects best using native b19 fitness. Fixed post-selection evaluations go into `evaluation/val` and
`evaluation/test`; original b19 re-evaluation goes into `baseline_comparison/b19_val` and `b19_test`.
The same B32/640/device0 FP32 settings, conf0.001, IoU0.7, max_det300 and no TTA apply to both. AP75 is extracted
from `all_ap[:,5]`, not inferred from mAP. Precision, recall, mAP50, mAP50-95, AP75, split, counts, weight hashes and
complete evaluation args are retained. Test is never used to select epoch, tune thresholds or modify the model.

Diagnostics use the first 16 lexically sorted validation image IDs, listed and hashed before inference. They record
all four weight norms, u/v channel-RMS distributions and near-zero ratios (fixed threshold1e-4), un/vn/interaction/t
moments, residual/L RMS and L2, and per-image residual/L and residual/cat ratios with denominator epsilon1e-6.
Live feature hooks are checked against the formula; all parameter/buffer hashes remain unchanged.
A single checkpoint cannot establish long-term gradient collapse: this is explicitly NOT MEASURED. The staged
preflight gradient summary is attached separately where present. Interaction responses are not probabilities.

Package requires complete training, preflight, comparison, diagnostics and evaluation artifacts. It includes both
weights and a Git source archive containing the complete dependency closure, source patch and file hashes, plus
manifest/read-back verification and a SHA256 sidecar. Attempt paths and symlinks are excluded with reasons; regular
hardlinks are materialized. Optional missing old console.log is recorded. Required artifacts cannot be skipped.
Only the selected run is walked, never other runs or datasets. Local package dry-run uses labeled fixture bytes;
Windows without symlink permission skips real-link integration explicitly.

## Source inventory and execution

Required implementation files are the module, minimal registry/parser edits, single-point YAML,
`run_b19_lbi_fusion.py`, `verify_b19_lbi_fusion.py`, `finish_b19_lbi_fusion.py`,
`server_b19_lbi_fusion_v1.sh`, `test_lbi_fusion_v1.py` and these two documents.
Additional dependencies are `b19_common.py`, `diagnose_b19_lbi_fusion.py`, `b19_reference.json`,
`b19_test_reference.json`, `b19_dataset_manifest.json` and `b19_launcher_expanded.txt`.
The sanitized reference, utility-source and local validation summaries are under `docs/experiments`.

```bash
python -m pytest tests/test_lbi_fusion_v1.py -q
python -m ruff check ultralytics/nn/modules/lbi_fusion.py tools/experiments tests/test_lbi_fusion_v1.py
bash -n tools/experiments/server_b19_lbi_fusion_v1.sh
```

Use a fresh ignored directory for each local audit; keep failed evidence. The local CLI requires `--local`,
`--baseline-root`, the exact archived `--baseline-args`, and `--output`. See the
[server procedure](b19_lbi_fusion_v1_server.md) for deployment and the automatic preflight entry.
Full commit/remote SHA reports are generated from Git into ignored `runs/lbi_local/deployment_report.json` and `.txt`
after committing; they are not committed recursively into the commit they identify.

Deleted: the experiment's layer15 native Concat YAML entry is replaced; no canonical b19 implementation is deleted.
Reused: native parser channel table, AMP helper, trainer/loss/MuSGD, class adaptation, lifecycle and audited neutral tools.
New lines are necessary for the specified residual and independent experimental evidence; adding a shared training
framework or modifying old experiment worktrees was unnecessary.

## References and boundaries

Low-dimensional cross-layer multiplicative interactions have prior art, including
[Yu et al., Hierarchical Bilinear Pooling](https://arxiv.org/abs/1807.09915).
LBI does not reproduce HBP's global classification head or borrow its classification improvements. Elementwise
multiplication is bilinear in normalized u/v, but the complete RMS/SiLU-containing mapping is not strictly bilinear.
Legal AMP observation follows the [PyTorch 2.8 AMP examples](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html):
unscale belongs at the actual accumulated optimizer boundary and only once per step. No accuracy, priority or
statistical-significance claim is made before the detection experiment.
