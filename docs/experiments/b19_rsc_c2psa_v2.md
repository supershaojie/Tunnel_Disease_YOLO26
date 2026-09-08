# b19 + RSC-C2PSA v2

Independent branch: `exp-yolo26n-b19-rsc-c2psa-v2`, based on v1 commit
`47a81c2e9da27b8f371ce22bd3d72c81b7b35e83`. This is the repository's YOLO26n crack detector.
Only layer 10 changes from native `C2PSA` to `C2PSA_RSC_V2`; layer 9 SPPF, QKV, PE(V), projection,
FFN, residuals, CSP bypass/concat, neck, P5 `[-1, 10]` and Detect remain native.

Actual dimensions: 256 input/output channels, 128 attention-branch channels, one PSA repeat,
two heads, 32 Q/K dimensions and 64 Value dimensions per head. Total parameters: **2,504,192**,
exactly **2** more than b19. The only added state key is `model.10.m.0.attn.theta`, shape `[2]`.
The original `Attention_RSC` and `C2PSA_RSC` classes and v1 YAML remain available for old checkpoints.

## Calculation

```python
S = (q * native_scale).transpose(-2, -1) @ k
L = S.float().log_softmax(dim=-1)
D = (0.5 * (L.transpose(-2, -1) - L)).tanh()
beta = 0.2 * theta.float().sigmoid()
delta_logits = beta.view(1, num_heads, 1, 1) * D
A_v2 = (S.float() + delta_logits).softmax(dim=-1)
# Cast to v.dtype, then native V aggregation, PE(V) and projection.
```

Each theta starts at `log(0.05/0.95)`, so beta starts at 0.01 and is bounded above by 0.2.
The 0.5 coefficient is fixed. `D` is antisymmetric with zero diagonal; each absolute logit change is at most
the corresponding beta. Row softmax generally produces asymmetric attention. Uniform attention has zero
correction and may have zero theta gradient. Nonzero initial beta is a perturbation, so enabled output is not
required to equal b19 at initialization. There is no entropy-decrease assertion.

Native Q scaling and matmul order/precision are preserved. Only correction operations and their softmax
explicitly use FP32; probabilities are cast to Value's dtype before aggregation. All correction paths remain
differentiable. Theta initialization reuses v1's constant allocation without consuming RNG, and native MuSGD
assigns it to the ordinary weight group without a special learning rate. `enabled=False` directly invokes
`Attention.forward`. Bypass comparisons use identical weights, device, precision and fusion state.

This is an engineering hypothesis awaiting experiments, with no guaranteed improvement or established academic
novelty claim. Reciprocal attention has prior art; dense N-squared attention and additional FP32 work remain.
Source history, attachment limitations and paper references are retained in `rsc_sources.json`; the v2 formula
and its user-specified origin are recorded in `rsc_v2_sources.json`. No linear or regional attention code was imported.

## Baseline and lifecycle

The server reads the actual b19 file, compares every field with its historical 112-field record, checks the expanded
launch record, and rejects any unapproved configuration difference:

```text
/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml
/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml
```

Args SHA256: `b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`.
Original `yolo26n.pt` SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Initialization must use that original 80-class checkpoint. The shared 708 state tensors, 606 matching pretrained
items, unmatched single-class head initialization, names, shapes and values are audited. Neither v1 best/last
nor preflight-updated state is a training initializer.

Both preflight and training retain batch=32, epochs=200, patience=60, imgsz=640, MuSGD, seed=42, AMP=True,
workers=8, device=0, deterministic=True, cache=False, and every b19 learning-rate, warmup, augmentation and loss field.
Only model and necessary data/weight/output paths change. Full resolved configuration and differences are printed
and saved in each audit's `resolved.json` and the training run's `provenance/resolved.json`.

`train` runs **one separate preflight process**, then reconstructs formal training with a fresh seed=42 and original
weights. The structural/initialization probes belong to the child; the parent does not repeat them. Preflight observes
32 real batches through unmodified `BaseTrainer._do_train`, including native AMP, GradScaler, warmup, accumulation,
clipping, MuSGD and EMA. It requires finite effective gradients, aggregate nonzero task gradients, at least two actual
optimizer steps, theta updates and EMA changes. Individual degenerate batches may have zero theta gradients.
It then checks the native training Validator, checkpoint serialization/reload/fuse and an independent FP32 Validator.
Fusion equivalence uses independent copies of the same complete checkpoint, including updated shared parameters.

Runtime requirements remain those inherited from v1: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.98 and
NVIDIA GeForce RTX 4090. Differences fail explicitly. A real memory failure exits before native batch reduction.
No global training lock, process termination, automatic batch reduction or replacement accumulation is used.
The lock is `runs/detect/yolo26n_b19_rsc_c2psa_v2.lock`, shared only by this experiment's stages.

Deleted: the duplicated shell lifecycle in the v1 entry and repeated structural/initialization probes in the training
parent. Reused: v1 configuration/weight/source audits, native training observer, evaluation and archive verification.
The two fixed shell entries now dispatch to one shared lifecycle with explicit version identity.

## Local validation and server checks

Local checks passed: 21 tests across `tests/test_rsc_c2psa.py` and `tests/test_rsc_c2psa_v2.py`.
The current local environment is Python 3.11.15, torch 2.7.1+cu118, RTX 2060 6GB. CUDA is available locally,
but this is not the b19 server environment. Coverage includes CPU/CUDA formula invariants, finite FP16 extremes,
640-square/640x512 full-model forward/backward, original weight and shared initialization audits, FP32/AMP
native/fused bypass, labeled native E2ELoss gradients and real MuSGD/GradScaler/clipping/EMA updates,
updated full EMA checkpoint reload/fuse, real FP32 Validator and the fixed 16-image diagnostic/report path.
The labeled local optimizer probes use batch=1 and are explicitly integration tests, **not** batch=32 preflight.
Synthetic labeled images verify API/loss/report behavior and make no detection-quality claim.

**待服务器验证**: actual b19 runtime batch=32 peak memory and concurrent GPU occupancy; 32 real native batches
with unchanged warmup/accumulation; full native and independent FP32 validation; formal training; final val/test,
trained-weight diagnosis, complete result packaging and shell flock/tmux behavior. No successful server receipt
or trained result package is claimed locally.

## Server commands

Use the final full commit SHA from delivery as `COMMIT`; do not deploy a moving branch head. No SSH is required.
Run the following in a shell on the server; existing worktrees and results are preserved:

```bash
export GIT_TERMINAL_PROMPT=0
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_RSC_C2PSA_v2
BRANCH=exp-yolo26n-b19-rsc-c2psa-v2
# COMMIT=<full SHA supplied in delivery>
git -C "$BASE" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
  fetch --progress --no-tags origin "$BRANCH"
git -C "$BASE" worktree add --detach "$WORK" "$COMMIT"
git -C "$WORK" rev-parse HEAD
```

For the normal launch, invoke `train` only; it runs preflight once before formal training:

```bash
tmux new-session -d -s y26_rsc_v2 -c "$WORK"
tmux set-option -t y26_rsc_v2 remain-on-exit on
tmux send-keys -t y26_rsc_v2:0.0 'B19_PYTHON=/root/miniconda3/bin/python bash tools/experiments/server_b19_rsc_c2psa_v2.sh train; rc=$?; printf "\nRSC v2 train exit=%s\n" "$rc"' C-m
tmux attach -t y26_rsc_v2
```

The interactive shell remains available on failure. Do not run `preflight && train`. For an optional diagnostic
preflight without training, the standalone command is:

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26_RSC_C2PSA_v2
B19_PYTHON=/root/miniconda3/bin/python bash tools/experiments/server_b19_rsc_c2psa_v2.sh preflight
```

Inspect attempts, logs, progress and exit status (status files appear when their stage exits):

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_RSC_C2PSA_v2
PROJECT="$WORK/runs/detect"
RUN=yolo26n_b19_rsc_c2psa_v2
cat "$PROJECT/${RUN}_train.current_attempt"
tail -n 80 -f "$PROJECT/${RUN}_train.console.log"
# Run the remaining commands in another shell, or stop tail with Ctrl-C.
tail -n 5 "$PROJECT/$RUN/results.csv"
cat "$PROJECT/${RUN}_train.exit_status" "$PROJECT/${RUN}_train.process_status.json"
cat "$(cat "$PROJECT/${RUN}_train.current_attempt")/exit_status"
tmux capture-pane -pt y26_rsc_v2:0.0 -S -80
```

After training exits successfully:

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26_RSC_C2PSA_v2
bash tools/experiments/server_b19_rsc_c2psa_v2.sh test
bash tools/experiments/server_b19_rsc_c2psa_v2.sh diagnose
bash tools/experiments/server_b19_rsc_c2psa_v2.sh package
```

`test` launches separate FP32 val and test processes with imgsz=640, batch=32, conf=0.001, iou=0.7 and
max_det=300, retaining unified evaluation settings. Metrics include P/R/AP50/AP75/mAP50-95, per-IoU AP,
predictions, curves, split, measured parameter dtype, one-to-one path invocation counts, code/data/weight hashes
and environment. `val_fp32.json` / `test_fp32.json` point to completed immutable report folders.

`diagnose` uses the first 16 sorted validation images and v1's 640-square LetterBox preprocessing. Per-head
theta/beta, D/delta-logit distributions (min/max/mean/std/p05/p50/p95/max-abs), original/corrected entropy,
row L1 and diagonal mass are recorded, with dense one-to-one box and score changes against native bypass.
Full attention matrices are not saved. Post-training bypass is an inference diagnosis, not a trained ablation.
Use validation to choose subsequent structure; avoid repeated test-set tuning.

`package` verifies the preflight actually used by training, stage exits, complete run files and bound report hashes.
It includes weights, args, results.csv, plots, val/test metrics and predictions, diagnostics, source archive/version,
source origins, execution logs and a per-file SHA256/size manifest; it rereads and verifies archive payload before
publishing. It prints actual archive path, byte size and SHA256 and writes a sibling `.tar.gz.json` receipt.

Expected download directory:

```text
/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_RSC_C2PSA_v2/artifacts/experiments/
yolo26n_b19_rsc_c2psa_v2_<12-character-commit>.tar.gz
```

Local test evidence is retained under the v2 worktree's `runs/local_v2_validation/` and summarized in
`tools/experiments/rsc_v2_local_validation.json`. Formal results are under
`runs/detect/yolo26n_b19_rsc_c2psa_v2/`; no result or weight is copied from v1 or another experiment.
