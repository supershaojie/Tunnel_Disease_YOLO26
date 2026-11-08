# DRSC-YOLO detection visualization

This tool compares the unchanged 1202-image, 1477-box test split. It reuses official low-confidence prediction caches;
only a missing or incomplete model is exported with its existing Python and native evaluator. Training, main-table
metrics, CAM and speed measurements are outside this tool.

Run from this compatible source worktree. Use a dedicated tool virtual environment with access to the existing native
Torch installation (`python -m venv --system-site-packages .venv-vis`). Install only plotting dependencies there:

```bash
.venv-vis/bin/python -m pip install numpy 'Pillow>=9.1' matplotlib PyYAML tqdm
.venv-vis/bin/python -m experiments.visualization.drsc_detection all --device 0
```

Defaults contain the seven official runs, checkpoint SHA256 values, dataset and existing source/Python paths in
`config.py`. `--base`, `--data` and `--out` relocate the server directories. For individual changes, `--overrides paths.json`
accepts only per-model `source`, `python`, `weights`, `run`, `cache` paths, for example:

```json
{
  "yolov8n": {
    "cache": "/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/comparison/yolov8n/compare_yolov8n_b19_e200_i640_b32_sgd_s42_retry1/evaluate_test"
  }
}
```

`prepare` only reads/exports caches; `candidates` only computes statistics and previews from `bundle.json`; `package`
rebuilds the candidate ZIP. `--reuse-only` stops with the concrete model/file reason if inference would be needed.
Inference runs sequentially in separate existing environments. Input rectangles, EMA, precision and postprocessing stay
with their evaluator. RT-DETR keeps native query probabilities and reverses letterboxing once; no visualization NMS is
applied. Checkpoint bytes, cache identity, collection threshold, test coverage and GT correspondence are checked before
use. Canonical GT comes from original labels; native inverse-rounding differences are recorded.

At display confidence 0.25, `stats.csv` records native class-aware one-to-one TP/FP/FN at IoU 0.50 and 0.75, exact matched
IoUs, redundant predictions and predictions without a qualifying match. The latter are not automatically called background
false positives. `ranking.csv` and `candidates.json` explain numerical differences against YOLOv26n, YOLOv13n and
RT-DETR-R18. Brightness/box-shape cues remain hypotheses. Parent IDs and image similarity reduce near duplicates. Up to
80 supported candidates are proposed; fewer are retained when necessary. No automatic shortlist is a final paper choice.

Review `review/overview/` and enlarge `review/previews/C001.png`. `candidates.zip` contains candidate originals once,
candidate-only numeric data, provenance, all-test statistics, previews and a proposed case file. Full prediction caches,
weights, environments and the full dataset stay outside the archive. Repackage after the run to include the completed log.

After inspecting real cases, render exactly five IDs:

```bash
.venv-vis/bin/python -m experiments.visualization.drsc_detection render --ids C001,C005,C012,C023,C037
```

Alternatively pass `--cases selected.json` with `{"status":"user_selected","cases":[...]}`; each of its five entries has
`case_id`, canonical `image_id`, and optional `crop: [x1,y1,x2,y2]` in original pixels. A crop is shared by all eight rows.
Rendering reads cached predictions and preserves image proportions, brightness, true scores and fixed row order.
`final/` contains 600 dpi PNG, SVG/PDF with embedded photos and vector boxes/text, and `cases.json` for reproduction.
`final_figures.zip` contains only the three figure formats. Photograph sampling is matched to export resolution, while
originals remain available unchanged.

An extracted candidate archive is portable: use `render --bundle /path/to/review/bundle.json --out /path/to/new-output
--ids C001,C005,C012,C023,C037` from the tool checkout. Candidate IDs are stable within the saved candidate list; changing
selection or crops never reruns inference.
