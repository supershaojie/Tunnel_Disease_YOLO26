# b19 + BDF-Fusion

Training status: **NOT_STARTED**. This is an unvalidated accuracy hypothesis, not evidence of improvement over b19.

The experiment is based on `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`. Only layer 18 changes in
`ultralytics/cfg/models/26/yolo26n-bdf-fusion.yaml`: `[17, 13, 6] -> BDF_Fusion`. Its output remains 192 channels;
the 24 original layer indices, downstream parameter paths, and Detect inputs `[16, 19, 22]` are preserved.

`BDF_Fusion` lives beside `Concat` in `ultralytics/nn/modules/conv.py`. It uses the exact signed difference
`P(C) - B`, with an identity-initialized trainable projection and a zero-initialized last gate. Hidden width 16,
eight contiguous channel groups, and amplitude 0.5 are fixed in the module, so its YAML arguments are empty.
There is no normalization or additional activation after Reduce. Construction restores the surrounding CPU/device RNG;
forward, checkpoint loading, EMA, and fuse never reinitialize learned parameters. C already influences B through the
native network; this module changes how those features interact, rather than introducing a previously absent source.

## Configuration and entry point

`train.py` accepts `--data`, `--weights`, `--project`, `--name`, `--device`, `--baseline-args`, and `--dry-run`.
The first three are required. Name defaults to `bdf_fusion_b19_e200_i640_b32_s42`, device to `0`.
Dry-run checks construction, original weights, and configuration through native Trainer rebuild methods without creating
the requested output directory. It reports environment differences; actual training requires the verified server core
environment and refuses existing experiment output directories. It never upgrades packages or changes batch/AMP/optimizer.

`b19.yaml` contains only the 18 differences from the locked source defaults. These were recovered from the local archived
`b19_yolo26n_e200_train_val_test_20260823_224153/train_run/args.yaml` and cross-checked with its b19 training log.
Optional `--baseline-args` checks the server's historical configuration against this complete effective configuration;
old model, data, project, name, device, resume, save_dir, pretrained, and cfg fields are excluded.
Unmodified defaults include `nbs=64`, `amp=True`, `workers=8`, `imgsz=640`, `deterministic=True`, `cache=False`,
`lr0=0.01`, `momentum=0.937`, `weight_decay=0.0005`, and `exist_ok=False`.
The unchanged native E2ELoss uses O2M gain 0.8 decreasing to 0.1 across 200 epochs, with complementary O2O gain;
the original O2O feature detach and MuSGD parameter grouping remain intact.

Both dry-run and training require the original weight SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Actual training additionally checks the same `yolo26n.pt` in the working directory for native AMP reuse and always uses
`resume=False`. No smoke checkpoint, dataset, training product, or environment package is committed.

## Finite validation, 2026-09-27

`verify.py` accepts `--weights`, `--dataset` (local b19 dataset root), and `--report` (local JSON output).
It exercises native Trainer rebuild/load methods and disposable model copies, with exactly three FP32 MuSGD steps on two
nonempty training samples resized to 256. It does not run formal training or dataset validation/test, and is not a benchmark.

Local environment: Windows, Python 3.11.15, torch 2.7.1+cu118, CUDA runtime 11.8, RTX 2060 6 GB, source Ultralytics 8.4.98.
Albumentations is installed locally, so this is not a reproduction of the server package environment. The smoke reads two
images directly and applies no Albumentations transforms. Shared packages were not changed.

- Native nc=1 state: all **708** same-name, same-shape parameters/buffers equal, including random detection-head state.
  **606** items inherit original weights; **102** shape gaps belong to native nc80-to-nc1 adaptation. The BDF model adds
  five state entries. Unexpected native missing items: **0**. Both CPU and CUDA-default-device RNG preservation passed.
- Model-level 640x640 and 384x640 CPU FP32 forwards: raw O2M/O2O and decoded output maximum absolute error **0** at Gate=0.
  Actual 640 layer-18 inputs: `[1,64,40,40]`, `[1,128,40,40]`, `[1,128,40,40]`; output `[1,192,40,40]`.
- Nonzero signed gates: explicit formula and contiguous eight-group broadcast error **0**; maximum output correction
  **0.1853646**. Inputs unchanged; invalid channel/spatial wiring rejected.
- Three real MuSGD updates: finite nonzero Gate task gradients on step 1; P/Reduce/DW task gradients zero on step 1 and
  finite/nonzero on steps 2-3, measured before optimizer decay. Every new parameter belongs to exactly one native group,
  with lr 0.01; no custom group or learning rate. CUDA FP16 autocast forward/backward also finite on a separate copy.
- Learned nonzero state: deepcopy, EMA, save/reload, fresh-process default YOLO.predict, and native model.fuse passed.
  Learned correction magnitude **0.0011653**; deepcopy/reload raw error **0**; EMA maximum raw error **4.8161e-5**;
  fused O2O raw error **5.6267e-5** (atol 2e-4, rtol 1e-4). P and Gate state preserved exactly through reload/fuse/predict.
- Dry-run passed with historical b19 args and created no formal output directory. Ruff and diff whitespace checks passed.
  Reference docs were regenerated; unrelated generated navigation reordering was excluded.

| nc=1 parameters   | Native b19 |       BDF |
| ----------------- | ---------: | --------: |
| Before fuse       |  2,504,190 | 2,525,974 |
| After native fuse |  2,375,031 | 2,396,815 |

The increment is **21,784** in both cases. Native fuse folds BN and removes the O2M inference branch, explaining the lower
totals. Added convolution work at 640 is **34,841,600 MACs**, or **0.0696832 GFLOPs** at two FLOPs/MAC; this excludes
elementwise operations and memory traffic.

Unverified: server Python 3.12.3 / torch 2.8.0+cu128 execution, full batch 32, full training, dataset metrics, ONNX/TensorRT,
and sustained performance. No server connection, formal training, parameter search, or complete val/test was performed.
