# b19 + RPCA-C2PSA v1

This is one independent Region-Probability Calibrated Attention experiment, starting from engineering commit
`905297080aaff29c463824fc6070ebff4b005d16`. The baseline is b19 only. SIR v2 and DCR code/history are retained;
their features are not present in this model. No second training, seed sweep, or test-driven hyperparameter search is scheduled.

## Identity and source evidence

- Branch: `exp-yolo26n-b19-rpca-c2psa-v1`.
- Server worktree: `/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_RPCA_C2PSA_v1`.
- Run: `runs/detect/yolo26n_b19_e1_rpca_c2psa_v1`.
- tmux: `y26_rpca_v1`.
- Initial checkpoint: `/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/yolo26n.pt`.
- Required initial SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

Directly read on this Windows host: native `block.py` Attention/PSABlock/C2PSA, `tasks.py` parsing/checkpoint loading,
module exports, native `yolo26.yaml`, SIR v1/v2 runners, evaluation/reload/packaging code, their tests, the b19 recipe
snapshot and expanded launcher. The local SIR v2 worktree was only inspected; all edits are in the new RPCA worktree.

The supplied module directory exists at `D:/7.21yolo26改/YOLO26缝合/`. The following files were directly read and hashed:

| File relative to that directory                                     | SHA256                                                             |
| ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `ultralytics/nn/newsAddmodules/MultipoleAttention_2025ICCV.py`      | `4384b51d42ad84b443181c02b3f6ab898118a1a286ae96a3d1d5f7bc4c7317cb` |
| `ultralytics/cfg/models/add26/yolo26_C2PSA_MultipoleAttention.yaml` | `2297014a36b37c642f36b7e190348efb8e9c90001d3186440a58216bbe9634a1` |

The adjacent `YOLO26缝合.zip` exists but was not extracted. The attachment named
`a43ad862-97c5-4751-9a8b-cbed0f9827c3.zip` was not located in the accessible attachments; no temporary ChatGPT path is claimed.
The package wrapper indeed ends with `permute(0,3,2,1)` (H/W exchanged), and its P5 Concat uses `[-1,9]`.
Neither code nor its YAML was copied. Avoiding those bugs is engineering correctness, not a research contribution.

[Multipole Attention](https://arxiv.org/abs/2507.02748) is accepted at the **ECLR Workshop at ICCV 2025**, not the main
conference. It motivates hierarchical interaction. [BiFormer](https://arxiv.org/abs/2303.08810) is the region-routing
related-work comparison. RPCA keeps every region, fine-grained V, and dense attention; it is not an implementation of
either paper's efficient attention. No claim of linear complexity, guaranteed speedup, proven novelty, or measured
detection improvement is made.

## Fixed implementation

Only layer 10 changes from `C2PSA` to `C2PSA_RPCA`. Layer 4 is native C3k2, layer 9 native SPPF, layer 21 Concat
remains `[-1,10]`, and Detect inputs stay `[16,19,22]`. The YAML is native b19 with `nc=1` and the sole class-name
replacement. Repeats/channels/scales/end2end/reg_max are unchanged. Registration includes both base and repeat modules.

`a,b=cv1(x).split((c,c),1)`: a is this deep layer's bypass half, not P3. Each PSA block reads the same a through its own
gate and processes b in order. Native QKV, heads, dimensions, scale, PE, projection, FFN and shortcut order remain.

```text
L = (Q / sqrt(dk))^T @ K
A = softmax_keys(L)
B = softmax_regions(mean_valid_region(L) + log(valid_region_area))
C = softmax_valid_keys_within_region(L)
gamma = 0.5 * sigmoid(Conv1x1 -> SiLU -> DWConv3x3 -> SiLU -> Conv1x1(a))
P = sum_region(A); M = (1-gamma)*P + gamma*B
A' = (1-gamma)*A + gamma*B*C
output = proj((original_V @ A'^T).to(V.dtype) + pe(original_V_spatial))
```

True 2x2 spatial grouping supports dynamic/odd/rectangular inputs and singleton axes. Temporary padding has an explicit
validity mask and contributes to neither means nor probabilities nor outputs. Q and V are not pooled. No hard routing,
clamp, epsilon division, detached gate/probability, extra BN/dropout/loss or second QKV projection is introduced.
The regional computation and V aggregation explicitly disable autocast and use FP32; QKV/gate/projection retain AMP.

Gate convolutions are `c->16` 1x1 (2064 parameters), 16-channel depthwise 3x3 (160), and `16->heads` 1x1 (34 for nano).
All have bias. The last weight is zero and bias is `-ln(9)`, giving gamma=0.05. Earlier gate gradients may be zero at the
first backward; later updates must make them nonzero. Nano adds **2258** parameters: **2504190 -> 2506448** unfused.
Dense NxN matrices remain; resource cost must be measured, not inferred from parameter count.

Serializable classes live in `ultralytics/nn/modules/rpca_c2psa.py`. Native state paths are preserved, with new
`model.10.m.<i>.gate.*` tensors only. Native modules are constructed first, replacement state is transferred and extra
construction occurs inside CPU `fork_rng(devices=[])`. Audit compares all common keys, shapes and values, including
buffers, and verifies source-weight migration. Local nano counts are 708 common tensors and 606 migrated tensors.

The internal `Attention_RPCA.enabled=False` path invokes native Attention. `run.bypass(model)` restores every flag even
on failure. This is an inference diagnostic, not a retrained ablation; normally enabled gamma=0.05 is not an identity test.
“Calibration” means attention mass redistribution, not calibrated detection confidence or foreground probability B.

## Training and preflight

`run_b19_rpca_c2psa.py` reuses the SIR engineering runner through explicit trainer/model/entrypoint/check dependencies.
Hard-coded shared layer, new-parameter and reload contracts were replaced with explicit arguments or trainer attributes;
SIR defaults are retained. The server wrapper reuses its stage/lock/PIPESTATUS implementation with RPCA-specific identity.

Original b19 args/launcher are checked against the archived full recipe before get_cfg. Missing source records use the
explicitly identified historical snapshot, never new library defaults. Any unapproved effective field difference stops.
Only model, same-weight loading path, independent project/name, and data path relocation are allowed. Full differences,
original source hash, class adaptation, import path and environment are recorded. nbs=64, end2end and schedules remain b19.

The sole formal run keeps epochs=200, patience=60, batch=32, imgsz=640, workers=8, seed=42, deterministic=True, AMP=True,
device=0, cache=False, original MuSGD, losses and every augmentation field. No environment upgrade, resume, batch
reduction, epoch extension or output overwrite is performed.

`train` automatically invokes missing preflight in a child process; the formal process builds a fresh native trainer
and seed. Receipts bind model/code/config/initial weight/data manifest/environment and hashed checks. Three disposable
real augmented batch=32 updates test QKV/gate/FFN gradients, actual MuSGD groups/updates and EMA; unit-scale AMP backward
is a gradient probe, while formal training keeps the native GradScaler. The preflight records per-step timing, allocated
and reserved peaks. It initially requires 8 GiB free and checks 2 GiB remaining reserve after probing; starting formal
training rechecks measured reserved peak plus 2 GiB. Other GPU jobs are only listed, not stopped; complete GPU idleness
is not required. Memory availability can change concurrently; an OOM stops without changing the recipe.

Each stage owns an independent attempt directory and records Python and tee statuses after exit. Previous current
statuses move into the new attempt before launch, so an active retry cannot display an old success. A tmux window alone
does not prove training started: inspect epoch log/actual process. Preflight success, formal loop, completed.json plus
successful train exit, and separate evaluation receipts are distinct evidence.

Saving uses the native FP16 EMA snapshot with an independent same-snapshot FP32 reference. Parent/child CPU thread and
backend conditions match. All 714 tensors and unfused outputs are exact. Fusion compares retained raw tensors, every
dense decoded anchor and all final detections aligned by native anchor identity at the original 1e-4 absolute/relative
tolerance; native top-k replay is also exact. This avoids comparing different anchors when near-equal scores change
sort order. No final box, gate parameter or buffer is dropped from the check.

## Evaluation, diagnosis and package

The shell entry accepts `preflight`, `train`, `test`, `diagnose`, `package`. `test` evaluates val-selected best.pt in
independent FP32 val and test at imgsz=640, batch=32, workers=8, device=0, conf=.001, iou=.7, max_det=300, rect=True,
augment=False, plots=True, save_json=True, quantize=None. Actual parameter dtype and split are asserted; image/instance
counts must be 2404/2985 and 1202/1477 respectively. AP75 is column 5 of `metrics.box.all_ap`; all ten AP thresholds and
unrounded P/R/AP50/AP75/mAP50-95, args, output path, source/environment and weight hashes are saved.

Only matching code/weight/data/split/settings and intact artifacts can reuse reports. `evaluation.json` selects current
reports and package does not rerun test. Main comparison is independent FP32 **val mAP50-95**, then AP75/Recall; test
reports generalization only. No historical scalar is automatically treated as a matched b19 evaluation. Compare b19 only
with matching FP32 split/parameters/data provenance. One training run cannot prove statistical significance.

`diagnose` uses the first 16 sorted real val images on a separate enabled model. It reports gamma distribution, head and
position variation, mean absolute P-to-M change, normalization/region errors, parameter count and resource measurements.
Only two samples' per-query gamma maps are plotted; no dataset-sized NxN arrays are saved. Timed normal FP32 batch=1
inference excludes diagnostic recomputation; reported diagnosis memory includes that recomputation.

`package` requires completed training/current val/test/diagnosis. It includes best.pt **and last.pt**, args/results/core
plots, predictions JSON, diagnostics, provenance/config differences/initialization and preflight evidence, logs/status,
exact YAML/module/source archive, environment/pip freeze, this source record, and per-file SHA256 manifest. It excludes
data, downloads, other experiments, preflight tensor dumps and epoch checkpoints. The 150 MiB input ceiling bounds
packaging; every archived member and gzip CRC are verified. Existing archives are preserved.

Expected server archive (not produced until server stages complete):
`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_RPCA_C2PSA_v1/runs/detect/yolo26n_b19_e1_rpca_c2psa_v1.tar.gz`.
Packaging prints its actual size and SHA256 and writes a companion `.sha256` file.

## Local evidence and remaining server work

Local validation runs on Windows, Python 3.11.15, PyTorch 2.7.1+cu118, RTX 2060 (6 GiB), Ultralytics 8.4.98.
`runs/rpca_development/` holds disposable local evidence. Local batch=2/640 real augmented AMP checks do **not** certify
the server batch=32 requirement. Probability references cover 20x20, 19x21, 20x16, 1x7, 7x1 and 1x1; full-model checks
cover 640x640 and 640x512. Tests also cover repetition/head variation, native RNG/state, exact bypass, gradient linkage,
new-process reload/fusion/predict, corrupted state rejection, diagnostic summaries, package and shell contracts.

No SSH configuration or running SSH connection was found. The actual server preflight, one formal run, independent
FP32 val/test, trained diagnostics and final result archive remain to be executed on AutoDL. No training metric or
performance improvement is claimed by local implementation checks.
