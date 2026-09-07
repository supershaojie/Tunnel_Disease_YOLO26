# b19 + RSC-C2PSA v1

This independent YOLO26n crack experiment starts at b19 source commit
`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`. Only backbone layer 10 changes from `C2PSA` to `C2PSA_RSC`.
Native SPPF, QKV, PE(V), projection, FFN, residuals, Detect and P5 `[-1, 10]` remain in the graph.
No DSD-Head, SIR-SPPF, DCR, RPCA, linear attention or regional pooling is included.

The actual model audit reports 256 input/output channels, 128 attention-branch channels, one PSA repeat,
two heads, Q/K dimension 32 and Value dimension 64 per head. Total parameters change from 2,504,190 to 2,504,192.
The only new state key is `model.10.m.0.attn.theta`, shape `[2]`.

## Computation and interpretation

The implementation keeps the native operation order `(q * self.scale).transpose(-2, -1) @ k` and native softmax.
For scaled scores `S` and native probabilities `A`, it computes:

```python
log_a = S.float().log_softmax(dim=-1)
R = (0.5 * (log_a + log_a.transpose(-2, -1))).softmax(dim=-1)
beta = 0.2 * theta.float().sigmoid()
A_new = ((1 - beta) * A.float() + beta * R).to(V.dtype)
```

Each head starts with `theta = log(0.05 / 0.95)`, so `beta = 0.01`. This is a small initial perturbation.
The reference and mixing paths use FP32 without detach. Native Value aggregation, PE(V) and projection follow.
Construction transfers existing Attention submodules after native C2PSA initialization; constant theta consumes no RNG.
The trainer audits every shared parameter and buffer, including all unmatched single-class Detect initializations.
It checks the actual optimizer and EMA after native AMP setup, not just a standalone model construction.
Native MuSGD assigns theta to its ordinary weight-decay group; no special optimizer group is added.

Symmetric row-normalized `A` is a fixed point. `R` has unit row sums, but generally is neither symmetric nor doubly
stochastic. The mathematical row bound is `||A_new - A||_1 <= 2 * beta`; floating-point checks use rounding tolerance.
Dense attention still costs quadratic token memory/computation and RSC adds work. No linear or zero-overhead claim is made.

`Attention_RSC.enabled = False` explicitly calls inherited native Attention. The `bypass(model)` context in the verification
module restores the flag after exceptions. Post-training disabling is an inference diagnostic, not an independently trained
ablation. No quality gain or statistical significance is asserted before the experiment is run.

## Baseline and sources

The actual archived b19 args and startup log were read at:

```text
E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153/train_run/args.yaml
E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153/logs/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42.log
```

Args SHA256: `b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`.
Original `yolo26n.pt` SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
The original checkpoint has 80 COCO classes; the native single-class load matches 606/708 baseline items and 606/709
candidate items. Trained `best.pt`/`last.pt` cannot be used as initialization.

The complete args snapshot, startup excerpt and historical expanded CLI record are in `tools/experiments/`.
The original shell script itself was not in the b19 archive. The expanded record was retained by the prior experiment
infrastructure and is checked against every resolved args field. The server must supply the actual baseline `args.yaml`;
there is no recipe fallback if it is missing. All effective differences are printed and saved in `resolved.json`.

The requested ZIP was not found. Corresponding extracted references were found and read at:

```text
D:/7.21yolo26改/YOLO26缝合/ultralytics/nn/newsAddmodules/MALA_2025ICCV.py
D:/7.21yolo26改/YOLO26缝合/ultralytics/nn/newsAddmodules/MultipoleAttention_2025ICCV.py
```

Their actual SHA256 values and provenance limitations are in `tools/experiments/rsc_sources.json`. Their identity against
the missing ZIP cannot be established. MALA uses positive ELU Q/K, RoPE and reassociated K-transpose-V attention;
MultipoleAttention uses hierarchical local windows with down/up sampling. Those implementations are mechanism references
only and are not copied or imported.

Geometric symmetrization has established precedent in
[Monotonicity properties of weighted geometric symmetrizations](https://arxiv.org/html/2408.04357v1).
Reciprocal attention also has precedent in
[Story2Board](https://arxiv.org/html/2508.09983v1), whose value-mixing selection mechanism differs from this experiment.
RSC is an independently implemented application of the specified calibration formula, not a claim to invent the operator.

Experiment-independent provenance, equality audits, fixed-batch OOM ownership and shell logging reuse the prior
`codex/fix-rpca-v1-amp-preflight` infrastructure. All SIR/RPCA-specific gates and assertions were excluded. The preflight uses
the native training loop itself, eliminating a separately maintained approximation of warmup and optimizer stepping.

## Validation boundary

Local environment: Windows, Python 3.11.15, PyTorch 2.7.1+cu118, RTX 2060 6 GB. Eight local tests passed, covering complete
640-square and 640x512 forward/backward and shape checks; shared initialization and real original pretrained values;
probability invariants and underflow; exact native bypass in FP32 and CUDA AMP; native MuSGD/GradScaler/clipping/EMA attention
probes; zero-gradient fixed points; separate-process checkpoint reload and fuse; real FP32 Validator on two synthetic labeled
images; and fixed-batch OOM behavior. These small local probes are not server batch-32 validation or accuracy measurements.

The local server-preflight invocation audited all baseline fields and real dataset counts and then exited nonzero for the
documented runtime mismatch. No passing server receipt was created. The b19 server environment is Python 3.12.3,
PyTorch 2.8.0+cu128, Ultralytics 8.4.98 and RTX 4090. No dependencies are installed/upgraded by the server script.

Pending server validation: real 32-image training batches, available VRAM alongside DSD, complete native AMP Validator,
complete checkpoint FP32 Validator, 200-epoch training (patience 60), full FP32 val/test, diagnostics and result packaging.

Preflight runs 32 actual batches through unchanged `BaseTrainer._do_train` and `optimizer_step`, observing AMP, GradScaler
overflow skips, native warmup LR/momentum, accumulation, gradient clipping, MuSGD and EMA. The first 32 b19 warmup batches
have accumulation 1; subsequent accumulation policy remains native. Aggregate finite task-gradient and actual theta updates
are required, not nonzero gradients on every batch. No GradScaler scale is manually lowered. After the disposable loop,
native training validation and an independent full checkpoint FP32 validation both run. Memory usage and failed attempts
remain logged. Preflight allocations are released before the checkpoint evaluation child starts.

Formal training always executes a fresh preflight child, even after a manual preflight. It waits for that child to exit,
checks evidence and current memory, resets seed 42 and constructs a fresh native trainer. Batch remains 32; the first native
OOM retry request re-raises its original exception before batch mutation. Only this experiment's lock is acquired; other
experiments are neither locked nor stopped. Existing run directories are preserved and cannot be resumed or overwritten.

## Server entrypoints

Use the exact full commit SHA delivered with the branch. No `pip install -U`, primary-checkout switch or upstream merge is
needed. Deploy as an independent worktree:

```bash
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_RSC_C2PSA_v1
REV=<full-commit-SHA-from-delivery>
git -C "$BASE" fetch origin exp-yolo26n-b19-rsc-c2psa-v1
git -C "$BASE" worktree add --detach "$WORK" "$REV"
cd "$WORK"
test "$(git rev-parse HEAD)" = "$REV"
bash tools/experiments/server_b19_rsc_c2psa_v1.sh preflight
```

`B19_PYTHON` defaults to `/root/miniconda3/bin/python`; it may point to an existing interpreter with the exact b19 runtime.
All stages use the same script:

```bash
tmux new-session -d -s y26_rsc_v1 -c "$WORK"
tmux set-option -t y26_rsc_v1 remain-on-exit on
tmux send-keys -t y26_rsc_v1 'bash tools/experiments/server_b19_rsc_c2psa_v1.sh train; rc=$?; printf "RSC training exit=%s\n" "$rc"' C-m
tmux attach -t y26_rsc_v1
```

The interactive shell remains after failure. Detach with Ctrl-b then d. No tmux session is killed or replaced.

```bash
PROJECT="$WORK/runs/detect"
RUN=yolo26n_b19_rsc_c2psa_v1
tail -f "$PROJECT/${RUN}_train.console.log"
cat "$PROJECT/${RUN}_train.current_attempt"
cat "$PROJECT/${RUN}_train.exit_status"
cat "$PROJECT/${RUN}_train.process_status.json"
tail -n 5 "$PROJECT/$RUN/results.csv"
bash "$WORK/tools/experiments/server_b19_rsc_c2psa_v1.sh" test
bash "$WORK/tools/experiments/server_b19_rsc_c2psa_v1.sh" diagnose
bash "$WORK/tools/experiments/server_b19_rsc_c2psa_v1.sh" package
```

Exit-status files appear when a stage finishes. Each attempt has separate logs/status, and previous receipts are retained.
`test` runs fresh processes for val and test: FP32, imgsz 640, batch 32, conf 0.001, iou 0.7, max_det 300, rect True,
augment False, workers 8. Each report includes split, exact weight SHA256, code commit, actual precision, metrics/AP by IoU,
curves/confusion matrices and predictions (JSON and labels). `diagnose` records the first 16 sorted validation images with
image hashes, per-head beta/entropy/diagonal mass and original/new differences; it does not save attention matrices.

The archive is created only after successful train/test/diagnose evidence checks, at:

```text
/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_RSC_C2PSA_v1/artifacts/experiments/yolo26n_b19_rsc_c2psa_v1_<12-character-commit>.tar.gz
```

It includes weights, args, results, curves, evaluations/predictions, diagnostics, source archive, reference provenance,
execution logs and a per-file checksum manifest. Payload hashes are read back from the archive. The package command prints
its actual path, bytes and SHA256 and writes an adjacent `.tar.gz.json` receipt. Download both from this directory.
The package command's final console/status stays beside the run because it is written after the archive has closed.
