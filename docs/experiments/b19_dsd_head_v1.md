# b19 + DSD-Head v1

This is one YOLO26n crack detection experiment based on commit
`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`. The only architecture change is layer 23
`Detect -> DSDDetect`. The backbone, SPPF, C2PSA, neck connections, losses, assigners,
`reg_max=1`, decoding and NMS-free postprocessing are native b19. No SIR, DCR, RPCA or RSC module is used.

## Identity and source evidence

- Branch: `exp-yolo26n-b19-dsd-head-v1`.
- Server worktree: `/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DSD_HEAD_v1`.
- Run: `runs/detect/yolo26n_b19_dsd_head_v1` inside that worktree.
- tmux session: `y26_dsd_v1`.
- Initial weight: the baseline project's original `yolo26n.pt`, SHA256
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- Actual b19 args read locally:
  `E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153/train_run/args.yaml`.
  Its SHA256 is `b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`.
  The adjacent original training log confirms 2,504,190 parameters, 606/708 transferred items,
  MuSGD and the full recorded recipe. The preserved expanded launcher comes from the earlier
  user-provided b19 launch record; it is checked against every effective argument.
- The UUID attachment zip was not located. Its supplied `/workspace/scratch/...` path is unavailable on this host.
  The accessible extracted reference was read at
  `D:/7.21yolo26改/YOLO26缝合/ultralytics/nn/newsAddmodules/DEGConv_CVPR2026.py`, SHA256
  `a1eb76f9c20ef8ef07e153e03fd718db10cc1dc8123b6a92768df2c3c5cc374d`.
  Its spatial four-patch split, HOG hard bins, GroupNorm and strip-convolution stack were not copied.

[MixerCSeg/DEGConv](https://arxiv.org/abs/2603.01361) motivates direction and local edge detail.
[PiDiNet](https://arxiv.org/abs/2108.07009) and
[RIS-PiDiNet](https://github.com/yuhua666/RIS-PiDiNet) provide difference-convolution and symmetry precedents.
Classification/regression separation also predates this experiment. The research candidate is the specified
dynamic constrained P3 regression adaptation, with no claim to first invent the underlying operators.
Actual source paths, hashes and provenance are in `tools/experiments/dsd_sources.json`.

## Computation and ownership

Layer 16 supplies P3: B x 64 x 80 x 80 at square 640 input. Detect inputs remain `[16,19,22]`.
P3 native regression remains `Conv(64,16,3) -> Conv(16,16,3) -> Conv2d(16,4,1)` at the same state keys.

Each adapter partitions 64 channels into eight contiguous groups of eight. Directions `(dy,dx)` are
`(0,1), (1,0), (1,1), (1,-1)`. For a direction e:

```text
D_e(X)(p) = [X(p+e) + X(p-e) - 2X(p)] / (dy²+dx²)
DeltaX_c = 0.025 * sum_e tanh(z[group(c),e]) * D_e(X_c)
X_box = X + DeltaX
```

Only positions with a complete 3x3 neighborhood receive a correction. The exterior pixel ring is exact identity;
H < 3 or W < 3 returns the input. Rectangles and odd feature sizes are supported. Zero padding is applied only to
the already computed correction, never to features used in differences. The coefficient network's requested
depthwise 3x3 uses padding=1.

The coefficient network is biased `64->16 1x1`, SiLU, biased depthwise `16->16 3x3`, SiLU, biased `16->32 1x1`.
The last weight and bias start at zero. Earlier layers retain native initialization. No BN, extra attention,
auxiliary loss, trainable forward-time state or detached feature computation is added.
Coefficient prediction follows native AMP; tanh, differences, accumulation and residual addition use FP32,
followed by conversion to the feature dtype.

Measured parameter counts: 1040 + 160 + 544 = **1744 per adapter**, **3488 added during training**,
**1744 added after fusion**. The nc=1 model has **2,507,678 unfused parameters**.
CPU RNG isolation protects every shared parameter and subsequent random draw. `reg_adapter` and
`one2one_reg_adapter` have identical initial values and independent storage.

`Detect.forward` still owns the input detach before the one-to-one head. The one-to-one adapter's own
parameters receive gradients, while its P3 input is already detached. `forward_head` feeds the correction
only into P3 regression and returns original `feats`; classification and P4/P5 use their original inputs.
Fusion removes `reg_adapter` with the one-to-many head and preserves the active one-to-one adapter.

## Fixed recipe and preflight

`tools/experiments/server_b19_dsd_head_v1.sh` supports exactly:
`preflight`, `train`, `test`, `diagnose`, `package`.

It requires the real baseline args at
`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml`.
Missing original args do not silently become a reconstructed recipe. The data remains
`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml`.
The runner checks the full recipe and launcher, data/class/count manifests, original checkpoint hash,
architecture and every shared and transferred initialization tensor.

The recipe retains epochs=200, patience=60, imgsz=640, batch=32, workers=8, device=0, seed=42,
deterministic=True, amp=True, optimizer=MuSGD and cache=False. Learning rates, warmup, augmentation,
loss weights and all other fields are inherited. All differences and excluded output metadata are printed
and written into `resolved.json` and the run's provenance. Unknown/version-dependent defaults fail explicitly.

The reused infrastructure derives from the fixed RPCA implementation at `6f5f1e2`.
Obsolete SIR structure/three-batch gradient checks were removed from the imported runner.
DSD preflight follows native AMP, GradScaler, warmup, accumulation, unscale, clip=10, MuSGD and EMA ordering.
It records attempted versus actually completed optimizer steps, overflow skips, unscaled gradients,
each adapter's first last-weight update and subsequent nonzero finite gradients for all twelve new tensors.
Zero first-layer gradients before the zero final projection learns are expected. Failure to establish the real
task path within 64 batches fails with evidence. Native early warmup may still use accumulation=1; the exact
observed accumulation is logged, without accelerating the schedule.

Preflight also checks one actual validation batch=32 through the real FP32 Validator/AutoBackend with the
numerical controls below. A separate process reloads an FP16 EMA
snapshot, checks exact state and raw outputs under matched CPU FP32 conditions, fuses it and runs prediction.
Near-tied top-k outputs are compared by anchor identity in the small reload probe.

### Validator numerical controls

The failure reported at `1e9ad5aae54e5d7e849bf403f33ca6fa2586ab54` compared two independent copies of the
**same post-update DSD EMA**, not an updated DSD against its initial native baseline. Both were FP32/eval,
but the check did not own the arithmetic policy during GPU fusion and forward. PyTorchBackend fuses a supplied
GPU module in place; native Conv/BN folding uses `torch.mm`. The usual CPU checkpoint route folds before GPU
transfer. A tensor's FP32 dtype alone does not rule out TF32 in either matrix multiplication or convolution.
See the [PyTorch CUDA precision documentation](https://docs.pytorch.org/docs/2.8/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).
The server's raw-box error cannot be explained by decode cancellation or top-k ordering. The old log alone
does not uniquely establish whether weight folding or convolution execution caused it.

`dsd_validator.py` now owns the comparison lifecycle. It removes the old first-mismatch-only hook and reuses
the existing reference arithmetic context and tensor comparison reporter. The added controls are necessary
to distinguish state/implementation errors from backend arithmetic; deleting the assertion cannot provide
that evidence. No model, native fusion implementation, training option or evaluation recipe is changed.

- One native b19 Detect graph receives the exact common EMA parameters **and BN buffers**. It is a numerical
  control, not an independently trained b19 or an adapter-off training ablation. Initial equality still uses
  fresh, untrained native/DSD models in the separate initialization checks.
- Native and DSD each run the real Validator with automatic GPU fusion, an unfused reference and a separate
  CPU-fused reference transferred to the GPU. Every comparison uses the same complete FP32 batch=32, verified
  by the input hash. Model state hashes, modes, gradient flags and independent storage are checked.
- All controls run under ambient precision and again under strict FP32 with autocast and both TF32 switches
  disabled **before fusion**. The context restores precision, deterministic settings, threads and RNG on success
  or failure. Formal AMP/MuSGD training remains unchanged and starts in its own freshly seeded process.
- `validator_check.json` retains every scale's actual feature shape, box element count, boxes/scores/features
  errors, GPU-versus-CPU fused state errors and graph/head layer errors. Layer traces retain the first and worst
  box/score images from complete batch forwards to limit diagnostic memory. No batch is reduced. The report
  is also written on failure, within the new preflight evidence directory; previous evidence is preserved.
- Acceptance keeps the original raw `atol=rtol=1e-4` for **both** strict native and strict DSD comparisons.
  Dense box tolerances are propagated through the existing affine decoder. Repeated unfused forwards and
  real Validator/fused forwards must match exactly, including in ambient mode; native postprocessing must
  match its own dense predictions exactly. Ambient fusion discrepancies are retained as failed numerical
  comparisons, never relabelled equivalent or accepted via a larger tolerance.
- The fused one-to-one adapter is observed on the actual Validator call and its repeat. Existing detach,
  independent optimizer/EMA, effective-update gradient and fresh-process save/reload checks remain mandatory.

Local synthetic regression injects a deliberate P4 fusion error: strict validation must reject it, leave the
source EMA/settings unchanged and retain the scale reports. The Windows RTX2060 lacks TF32 tensor cores;
local success cannot establish the cause of the RTX4090 failure or stand in for the user's real-data preflight.

Formal training uses a separate process from disposable preflight state and native trainer seeding is repeated.
Only this experiment's lock is acquired. Other GPU jobs are reported and remain running. The server must match
b19's Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.98 and RTX 4090 environment. Preflight requires 8 GiB initial
free memory, measures actual CUDA peaks, and requires 2 GiB remaining headroom. Formal startup rechecks the
measured peak plus reserve. Native automatic batch reduction is disabled at the retry owner: OOM is re-raised
before args or the loader change. Existing runs and failed attempts are preserved.

## Deployment and operation

Deploy the exact full SHA from the handoff, using the existing b19 environment; do not upgrade the framework:

```bash
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DSD_HEAD_v1
SHA=<FULL_COMMIT_FROM_HANDOFF>
git -C "$BASE" fetch origin exp-yolo26n-b19-dsd-head-v1
git -C "$BASE" worktree add --detach "$WORK" "$SHA"
test "$(git -C "$WORK" rev-parse HEAD)" = "$SHA"
cd "$WORK"
bash tools/experiments/server_b19_dsd_head_v1.sh preflight
```

Create an independent tmux shell first, enable window retention, then start training. `train` automatically
launches a separate preflight process when matching successful evidence is absent:

```bash
tmux new-session -d -s y26_dsd_v1 -c "$WORK"
tmux set-option -t y26_dsd_v1 remain-on-exit on
tmux send-keys -t y26_dsd_v1 'bash tools/experiments/server_b19_dsd_head_v1.sh train' C-m
tmux attach -t y26_dsd_v1
```

Progress and exit state:

```bash
RUN="$WORK/runs/detect/yolo26n_b19_dsd_head_v1"
tail -n 5 "$RUN/results.csv"
tail -n 80 "$WORK/runs/detect/yolo26n_b19_dsd_head_v1_train.console.log"
cat "$WORK/runs/detect/yolo26n_b19_dsd_head_v1_train.exit_status"
cat "$WORK/runs/detect/yolo26n_b19_dsd_head_v1_train.process_status.json"
tmux capture-pane -pt y26_dsd_v1 -S -80
```

During a current attempt an exit-status file is absent until the process finishes. Previous statuses are kept in
the new attempt directory; `*_current_attempt` identifies that directory. Python and tee statuses are separate.

After successful training:

```bash
cd "$WORK"
bash tools/experiments/server_b19_dsd_head_v1.sh test
bash tools/experiments/server_b19_dsd_head_v1.sh diagnose
bash tools/experiments/server_b19_dsd_head_v1.sh package
```

`test` starts separate FP32 val and test processes with imgsz=640, batch=32, conf=0.001, iou=0.7, max_det=300,
rect=True, augment=False and quantize=None. Reports include split, actual dtype, weight SHA256, code SHA,
unrounded AP50/AP75/mAP50-95/recall and prediction JSON/text. Evaluation does not select a test checkpoint.

`diagnose` uses the first sixteen sorted validation images with content hashes. It records signed tanh coefficients
per direction and per group (histogram, quantiles, saturation), DeltaX norm/input norm for both adapters, and
fused one-to-one calls and adapter-on/off effects. It includes the matching full validation metrics.
Adapter-off results are inference diagnostics of the same trained checkpoint, never an independent training ablation.
Affine-feature zero response does not imply illumination or box-position invariance.

`package` requires completed training plus matching valid metrics/diagnostics and checks all report artifact hashes.
It bundles best/last weights, args, CSV, curves, predictions, diagnostics, provenance, source archive/patch,
execution logs and a per-file checksum manifest, and re-reads the archive to verify it. The printed record contains
the actual path, byte size and SHA256. Expected download files are:

```text
/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DSD_HEAD_v1/runs/detect/yolo26n_b19_dsd_head_v1.tar.gz
/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DSD_HEAD_v1/runs/detect/yolo26n_b19_dsd_head_v1.tar.gz.sha256
```

No server training, AP gain, trained weights or completed result archive is claimed by preparing this implementation.
Development checks on the Windows RTX 2060 are distinct from the fixed server recipe. In particular,
`DSD_RUN_NATIVE_SMOKE=1` enables a labelled synthetic batch=2/128 CUDA integration test for the real trainer;
it cannot create a successful server preflight receipt.
