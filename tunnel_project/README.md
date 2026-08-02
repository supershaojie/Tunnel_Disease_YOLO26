# Tunnel Crack Dataset Experiments

This directory contains project-specific, reproducible dataset preparation code for the tunnel-crack experiments.
The authoritative input for the AugFirst Diverse5x experiment is only
`datasets/Tunnel_Crack_Original_NoAug_7_2_1_seed42`; the builder reads its current repaired labels and never reads
labels from the original 20,206-image collection.

## AugFirst Diverse5x RandomSplit

The experiment intentionally reproduces and diagnoses the legacy file-level random-split protocol:

- `split_policy=file_level_random`
- `parent_id_grouping=false`
- `independent_test_set=false`

All five output files for a parent (`orig`, `geo`, `light`, `degrade`, and `compound`) enter one globally sorted file
pool. The builder performs exactly one `random.Random(42).shuffle` and does not group by `parent_id` or stratify by
variant. Consequently, related files may cross train, validation, and test. This experiment must not be described as
a strict leakage-free independent test set.

The safe default processes exactly 50 deterministically selected source images and produces a 250-file dry-run:

```powershell
$env:NO_ALBUMENTATIONS_UPDATE = '1'
& 'D:\miniconda3\envs\yolo26\python.exe' -B `
    'tunnel_project\scripts\02_build_crack_augfirst_diverse5x_randomsplit.py' `
    --source 'datasets\Tunnel_Crack_Original_NoAug_7_2_1_seed42' `
    --output 'datasets_dryrun_Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42' `
    --seed 42
```

The complete 2,404-parent build is locked behind the explicit `--full` option. Do not use that option during the
dry-run and manual-review stage. The builder refuses an existing output, writes to a same-disk temporary directory,
validates the entire result, verifies the source fingerprint before and after generation, and only then atomically
renames the temporary directory to its final name.

Generated manifests contain full SHA-256 hashes, stable SHA-256-derived sample seeds, actual applied transform
parameters, pre/post box counts, and file-level split positions. The audit reports retain the existing 8,149 source
64-bit dHash candidate pairs and compute dry-run exact/near-duplicate statistics without automatically deleting or
regrouping any sample.

### Exhausted-retry safety fallbacks

The original 12 seeded attempts, parameter ranges, order, acceptance checks, and quality thresholds remain the normal
path. A normal candidate that passes is returned immediately and records `fallback_used=false`. Only a `light` or
`compound` target for which all 12 normal attempts fail enters the existing deterministic luminance fallback stage.
A `geo` target that exhausts all 12 normal candidates enters a separate deterministic full-frame axis-flip fallback.
The safe geometry uses no padding or interpolation, maps boxes analytically in their original order, and is shared by
standalone `geo` and the geometry component of `compound` fallback. Standalone `geo` also applies the existing bounded
luminance adjustment after the flip. Every fallback still has to preserve a one-to-one source/output box mapping, pass
all existing quality checks, and pass the nonidentity check; otherwise the formal builder remains fail-fast.

Before another formal build, scan all 9,616 `geo`, `light`, `degrade`, and `compound` targets without persisting a JPEG
dataset:

```powershell
$env:NO_ALBUMENTATIONS_UPDATE = '1'
& 'D:\miniconda3\envs\yolo26\python.exe' -B `
    'tunnel_project\scripts\02_build_crack_augfirst_diverse5x_randomsplit.py' `
    --source 'datasets\Tunnel_Crack_Original_NoAug_7_2_1_seed42' `
    --output 'datasets\_preflight_Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42' `
    --seed 42 `
    --preflight-collect-all
```

This audit-only mode calls the production augmentation path in the same stable order as a formal build, continues after
individual failures, and writes only a JSON report plus CSV details. It never enables `--full`, never writes the formal
dataset, and never relaxes a quality threshold.
