# b19 + MSI-C2PSA v1

This branch starts from native b19 `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`. Only model layer 10 changes to
`C2PSA_MSI`. Native SPPF, C3k2, Neck, Detect `[16,19,22]`, losses and one-to-one feature detach remain unchanged.
No RSC, RPCA, SIR-SPPF, DCR, DSD or MPDF model code is imported. Other worktrees and results are preserved.

Deleted: the adapted lifecycle's RSC version selectors, attention calibration diagnostics and full-dataset preflight
evaluation. Reused: `b19_common.py` configuration/initialization/optimizer audits and the existing attempt, FP32 evaluation
and archive lifecycle. The new module adds the requested independent branch; deleting native layers would violate b19.

## Model and initialization

`ultralytics/nn/modules/msi_c2psa.py` registers `C2PSA_MSI` in both base and repeat parser sets. The YAML is
`ultralytics/cfg/models/26/yolo26n-msi-c2psa-v1.yaml`; n scaling is applied exactly once. Native outer C2PSA forward is
inherited, with its native cv1/cv2 construction order. One PSABlock has c=128, two heads, and 256 expanded FFN channels.

```text
z = x + Attention(x)                     # shortcut=True
U = ffn[0](z)                            # exactly once, native Conv/BN/SiLU
Ua, Ub = split(U, [c,c], dim=1)
V = ffn[1](U)                            # native Conv/BN, no activation
Delta = Wo(GELU(DW3(Ua)) * DW5(Ub))
y = (z + V) + Delta
```

For shortcut=False, z=Attention(x) and y=V+Delta. DW3/DW5 are full, bias-free depthwise 3x3/5x5 convolutions with
unit stride and padding 1/2. GELU uses approximate="none". Wo is biased 1x1 convolution without BN or activation.
Each depthwise channel is explicitly initialized to identity; Wo weight/bias start at zero. New convolution construction
uses `torch.random.fork_rng(devices=[])` on CPU, preserving all native parameter initialization and subsequent RNG state.
Initialization only occurs in constructors; loading, EMA and fuse do not zero trained parameters.

Measured unfused single-class counts: native **2,504,190**, candidate **2,525,054**, added **20,864**. Added convolution
MACs at 640 are 8,294,400 per image (not a latency/memory measurement). Layer 10 output is 256x20x20 at 640.
The only extra state keys relative to the native nc=1 model are:

```text
model.10.m.0.msi.dw3.weight
model.10.m.0.msi.dw5.weight
model.10.m.0.msi.project.weight
model.10.m.0.msi.project.bias
```

All 708 shared parameter/buffer keys are checked exactly, including BN running state. Actual native pretrained loading
matches 606 tensors in both models. COCO nc=80 to crack nc=1 head shape mismatches are separately enumerated and follow
native initialization/class remapping; they are not MSI missing keys. The experiment never loads b19 best.pt as initialization.

## Sources and b19 evidence

`tools/experiments/msi_sources.json` records actual full paths, SHA256 hashes, relevant source sections and differences.
The read ZIP `D:/7.21yolo26改/YOLO26缝合.zip` has SHA256
`a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`.
Both inspected FeedForward files were verified byte-for-byte against their ZIP members. AAFM supplies only the
DW3/DW5 -> GELU(first)\*second operator idea. Its title/authors/DOI are not asserted. HMHA's FeedForward uses a shared 3x3
depthwise convolution on both halves; its associated HINT paper is
[ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Devil_is_in_the_Uniformity_Exploring_Diverse_Learners_within_Transformer_ICCV_2025_paper.html),
despite the filename. [TransNeXt, Dai Shi, CVPR 2024](https://arxiv.org/abs/2311.17132), section 3.2 and appendix D.4,
is a conceptual reference for local convolution before a GLU activation; this experiment does not copy its full block or
inherit its performance claims. Native Attention already includes positional encoding; only its FFN's two 1x1 projections
do not directly mix neighboring spatial positions.

The actual archived b19 args, startup log, git_state, original weight and dataset configuration were read locally.
`b19_reference.json`, `b19_archived_args.yaml`, `b19_startup_excerpt.txt` and `b19_launcher_expanded.txt` preserve that evidence.
The launch record was supplied in an earlier b19 task; the original shell script was absent from the results package.
The actual original weight SHA256 is `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

Formal startup requires the actual baseline run's args.yaml and original weight. Missing/conflicting inputs cause failure.
All effective b19 fields are compared, not just headline settings. Permitted differences are model/pretrained path identity,
data path relocation to identical split configuration, project and name; save_dir/cfg are regenerated metadata.
Every runtime saves the complete effective config, differences, source hashes and dataset manifest. In native trainer
setup the requested config is checked again, along with optimizer membership/settings and initial EMA state.

The pinned recipe includes epochs=200, patience=60, imgsz=640, train batch=32, MuSGD, seed=42, AMP=True, nbs=64,
lr0=0.01, lrf=0.003, momentum=0.937, weight_decay=0.0005 and the original augmentations/warmup/accumulation.
Runtime checks require the recorded b19 Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.98 and RTX 4090 environment;
no dependency upgrade is performed. B19_PYTHON chooses the existing matching interpreter.
Native formal training validation retains its own loader policy (detection uses twice the training batch); this is distinct
from train batch=32 and the explicit preflight/evaluation batch=32. No formal trainer behavior is changed for diagnostics.

## Validation and stage ownership

`tests/test_msi_c2psa.py` covers square 640 and rectangular 384x672 shapes, both shortcut formulas, exact shared
initialization/RNG, all identity channels, one FFN expansion/BN update, actual pretrained loading, native class compatibility,
data gradients, native MuSGD membership, EMA, fresh-process YOLO reload, fuse, AutoBackend, real Validator, actual one-to-one
outputs and preserved detach. Local batch=1 labeled probes and a CPU/64px synthetic Trainer lifecycle are integration tests,
not the formal preflight. `tools/experiments/msi_local_validation.json` records measured local results and limitations.

`train` always launches one independent preflight process. Preflight verifies untrained native/candidate batch=32 FP32
outputs with sequential GPU placement/release, then runs the unchanged native training loop at batch=32/imgsz=640/AMP,
with the formal optimizer and augmentations. At most 32 real batches are allowed; success needs at least two actual
optimizer.step calls, EMA updates, nonzero Wo update and nonzero finite DW3/DW5 data gradients after Wo updates.
GradScaler-skipped attempts are recorded separately; zero initial DW gradients are expected. No batch reduction is allowed.

The native training Validator uses a fixed list of 32 val images; a separate process checks the saved complete nonzero
checkpoint through FP32 AutoBackend/Validator on the same list. Fuse comparisons use independent copies of the same
quantized checkpoint, eval/FP32 CPU with recorded numerical conditions, and compare stable raw anchor order. Tolerances
are fixed at 1e-4 absolute/relative for fused raw outputs; exact initialization/loading use zero tolerance. No formal TF32
setting is changed. A failure records shape, dtype, maximum/mean error and counts outside tolerance.

Only after the preflight process exits does formal training reset seed and reconstruct from original yolo26n.pt. It does
not reuse preflight weights, BN, optimizer or EMA. The entry locks only MSI and does not stop other experiments.
Each shell stage owns a new attempt directory, current_attempt, running/exited state, independent console log and final
exit_status. Old receipts are moved into the new attempt's previous.\* files before it runs. No current success code exists
while a new attempt runs. Existing formal run directories are refused without starting another train attempt.
SIGKILL/power loss cannot write an exit receipt; use the saved PID to distinguish an interrupted attempt from live work.

`test` loads the native validation-selected best.pt and runs full FP32 val and test in separate processes, batch=32,
imgsz=640, conf=0.001, iou=0.7, max_det=300, rect=True, augment=False, workers=8. It records P/R/AP50/AP75/mAP50-95,
curves, predictions, split, actual dtype, exact weight/source hashes, settings and executed MSI/one-to-one hooks.
`diagnose` requires those saved evaluations and inspects the first 16 sorted val images. It measures per-sample L2 over
C,H,W with epsilon=1e-12 for Delta/V and Delta/(z+V), quantiles, depthwise center/off-center norms and identity changes,
Wo norms, preflight task gradients and actual raw one-to-one boxes/scores changes under a temporary output hook.
The hook is removed afterward and no trained tensor is altered. This is post-training inference diagnosis, not a separate
trained ablation. Responses/gradients/norms cannot prove AP gains. Archived b19 metrics are not claimed to be same-condition
FP32 reevaluations.

`package` requires completed train/test/diagnose receipts, verified preflight, best/last, args/results/curves, all evaluation
and diagnostic evidence, then checks artifact hashes and creates a verified archive under `artifacts/experiments`.
It includes the exact Git source tree, own reproducibility source/audits, stage logs/statuses and checksum manifest.
Raw datasets and credentials are excluded. Missing stages fail explicitly. The package command's final exit state remains
beside the stage attempt; the archive and JSON sidecar report the actual full path, bytes and SHA256. Download that printed
archive path using the existing AutoDL/FileZilla connection; no fabricated public download URL is produced.

## Fixed-commit deployment

Set SHA to the full commit in the delivery message. These commands preserve existing worktrees and refuse conflicting state.

```bash
set -euo pipefail
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MSI_C2PSA_v1
BRANCH=exp-yolo26n-b19-msi-c2psa-v1
SHA=FULL_COMMIT_FROM_DELIVERY
if ! git -C "$BASE" cat-file -e "$SHA^{commit}" 2>/dev/null; then
    GIT_TERMINAL_PROMPT=0 git -C "$BASE" -c http.version=HTTP/1.1 \
        -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
        fetch --progress --no-tags origin "$BRANCH"
fi
git -C "$BASE" cat-file -e "$SHA^{commit}"
if [ -e "$WORK" ]; then
    test "$(git -C "$WORK" rev-parse HEAD)" = "$SHA"
    test -z "$(git -C "$WORK" status --porcelain --untracked-files=no)"
else
    git -C "$BASE" worktree add --detach "$WORK" "$SHA"
fi
git -C "$WORK" rev-parse HEAD
```

## Independent tmux launch

Normal startup calls only train; do not prepend another preflight. Create an interactive shell first so failures remain
visible; the session-existence check prevents a second start.

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MSI_C2PSA_v1
if tmux has-session -t y26_msi_v1 2>/dev/null; then
    echo 'Existing MSI session retained; attach to inspect it.'
else
    tmux new-session -d -s y26_msi_v1 -c "$WORK"
    tmux set-option -t y26_msi_v1 remain-on-exit on
    tmux send-keys -t y26_msi_v1:0.0 'B19_PYTHON=/root/miniconda3/bin/python bash tools/experiments/server_b19_msi_c2psa_v1.sh train; rc=$?; printf "\nMSI train exit=%s\n" "$rc"' C-m
fi
tmux attach -t y26_msi_v1
```

## Progress and other stages

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MSI_C2PSA_v1
P="$WORK/runs/detect"
N=yolo26n_b19_msi_c2psa_v1
tmux capture-pane -pt y26_msi_v1:0.0 -S -80
STAGE=train
cat "$P/${N}_${STAGE}.current_attempt"
A=$(cat "$P/${N}_${STAGE}.current_attempt")
cat "$A/process_status.json"
ps -p "$(cat "$A/shell.pid")" -o pid,ppid,etime,stat,args
pgrep -af '[r]un_b19_msi_c2psa.py|[f]inish_b19_msi_c2psa.py'
if [ -f "$A/exit_status" ]; then cat "$A/exit_status"; else echo 'No final exit receipt; inspect PID/state.'; fi
tail -n 5 "$P/$N/results.csv" 2>/dev/null || true
tail -f "$A/console.log"
```

The automatic preflight child has `preflight.log`, `preflight_process.json` and `preflight.exit_status` inside the audit
directory named in the train attempt's `audit_path.txt`. Use STAGE=test/diagnose/package to inspect their receipts.

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MSI_C2PSA_v1
export B19_PYTHON=/root/miniconda3/bin/python
bash "$WORK/tools/experiments/server_b19_msi_c2psa_v1.sh" preflight  # optional standalone check
bash "$WORK/tools/experiments/server_b19_msi_c2psa_v1.sh" test
bash "$WORK/tools/experiments/server_b19_msi_c2psa_v1.sh" diagnose
bash "$WORK/tools/experiments/server_b19_msi_c2psa_v1.sh" package
ls -lh "$WORK/artifacts/experiments/"
```

Server execution, real batch=32 GPU resource checks, formal 200-epoch training, final AP comparison and complete result
archive remain for the user to execute. Local validation does not claim those stages succeeded.
