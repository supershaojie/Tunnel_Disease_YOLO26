"""Independent FP32 val/test, fixed-val attention diagnosis, and verified light RPCA result packaging."""

# ruff: noqa: E402 -- Select this worktree before importing its package.

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import finish_b19_sir_sppf as common
from tools.experiments import finish_b19_sir_sppf_v2 as finish
from tools.experiments import run_b19_rpca_c2psa as experiment
from tools.experiments import run_b19_sir_sppf as shared

import cv2
import matplotlib.pyplot as plt
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.modules import C2PSA_RPCA
from ultralytics.nn.modules.rpca_c2psa import calibrated_probabilities, region_group
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML


def distribution(x):
    """Summarize a bounded diagnostic tensor, never serialize dense attention matrices."""
    x = x.detach().float().flatten()
    return dict(
        mean=x.mean().item(),
        std=x.std(unbiased=False).item(),
        min=x.min().item(),
        max=x.max().item(),
        quantiles=dict(
            zip(
                ("0", "0.05", "0.25", "0.5", "0.75", "0.95", "1"),
                x.quantile(torch.tensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1], device=x.device)).tolist(),
            )
        ),
    )


@torch.no_grad()
def attention_statistics(module, x, gamma):
    """Recompute only the current small sample and return summaries plus per-query scalar maps."""
    batch, _, height, width = x.shape
    q, k, _ = (
        module.qkv(x)
        .view(batch, module.num_heads, 2 * module.key_dim + module.head_dim, height * width)
        .split((module.key_dim, module.key_dim, module.head_dim), dim=2)
    )
    logits = (q.float() * module.scale).transpose(-2, -1) @ k.float()
    calibrated, detail = calibrated_probabilities(logits, gamma, height, width, diagnostics=True)
    change = (detail["M"] - detail["P"]).abs()
    row = dict(
        gamma=distribution(gamma),
        gamma_by_head=[distribution(gamma[:, i]) for i in range(module.num_heads)],
        gamma_position_std_by_head=gamma.squeeze(-1).std(-1, unbiased=False).mean(0).tolist(),
        region_mass_mean_absolute_change=change.mean().item(),
        probability_row_sum_max_error=(calibrated.sum(-1) - 1).abs().max().item(),
        region_mass_max_error=(region_group(calibrated, height, width).sum(-1) - detail["M"]).abs().max().item(),
    )
    maps = gamma[0, :, :, 0].reshape(module.num_heads, height, width).cpu()
    return row, maps


@torch.no_grad()
def diagnose(run, data):
    """Inspect the first 16 sorted val images in an independent model; preserve normal enabled evaluation."""
    evidence = finish.provenance(run, data, experiment)
    cfg = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(cfg["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    assert len(images) == 16
    evidence["samples"] = [[str(path), shared.sha256(path)] for path in images]
    output = finish.report_directory(run, "diagnostics", evidence)
    if not (output / "metrics.json").is_file():
        output.mkdir(parents=True, exist_ok=False)
        model, _ = load_checkpoint(evidence["weight"])
        assert type(model.model[10]) is C2PSA_RPCA
        model = model.float().eval().to("cuda:0")
        assert all(block.attn.enabled for block in model.model[10].m)
        rows, timings = [], []
        torch.cuda.reset_peak_memory_stats(0)
        for sample_index, path in enumerate(images):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable validation image: {path}")
            image = LetterBox((640, 640), auto=False, stride=32)(image=image)
            tensor = (
                torch.from_numpy(image[..., ::-1].transpose(2, 0, 1).copy()).unsqueeze(0).to("cuda:0").float() / 255
            )
            # Measure normal inference without diagnostic recomputation or plotting in the timed interval.
            torch.cuda.synchronize()
            started = time.perf_counter()
            model(tensor)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            blocks, maps, hooks = [], [], []

            def inspect(module, inputs):
                row, heatmap = attention_statistics(module, *inputs)
                blocks.append(row)
                maps.append(heatmap)

            for block in model.model[10].m:
                hooks.append(block.attn.register_forward_pre_hook(inspect))
            try:
                model(tensor)
            finally:
                for hook in hooks:
                    hook.remove()
            rows.append(dict(image=str(path), sha256=shared.sha256(path), blocks=blocks))
            if sample_index < 2:
                for block_index, heatmaps in enumerate(maps):
                    figure, axes = plt.subplots(1, len(heatmaps), squeeze=False, figsize=(4 * len(heatmaps), 3))
                    for head, heatmap in enumerate(heatmaps):
                        plot = axes[0, head].imshow(heatmap.numpy(), vmin=0, vmax=0.5, cmap="viridis")
                        axes[0, head].set_title(f"gamma / block {block_index} / head {head}")
                        figure.colorbar(plot, ax=axes[0, head])
                    figure.tight_layout()
                    figure.savefig(output / f"gamma_{sample_index}_block_{block_index}.png", dpi=130)
                    plt.close(figure)
        shared.write_json(
            output / "metrics.json",
            dict(
                evidence=evidence,
                images=rows,
                actual_parameters=sum(p.numel() for p in model.parameters()),
                resources=dict(
                    batch=1,
                    imgsz=640,
                    precision="FP32",
                    inference_seconds=timings,
                    warm_seconds_mean=sum(timings[1:]) / len(timings[1:]),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                    peak_scope="includes one-sample diagnostic recomputation; training costs are in preflight checks.json",
                ),
                mode="16 fixed real val images; no augmentation; enabled attention; independent checkpoint instance",
                interpretation="Attention mass redistribution; B is not foreground probability or detection confidence calibration. "
                "Single-run differences do not establish statistical significance. No test-driven tuning.",
                artifacts={p.name: shared.sha256(p) for p in output.glob("*.png")},
            ),
        )
    shared.write_json(
        run / "diagnostics.json",
        dict(path=(output / "metrics.json").relative_to(run).as_posix(), sha256=shared.sha256(output / "metrics.json")),
    )


def package(run, data, output):
    """Include best AND last, current reports, source and initialization evidence without running evaluation."""
    finish.package(
        run,
        data,
        output,
        experiment=experiment,
        include_last=True,
        source_files=(
            "ultralytics/nn/modules/rpca_c2psa.py",
            "ultralytics/cfg/models/26/yolo26n-rpca-c2psa-v1.yaml",
            "tests/test_rpca_c2psa.py",
            "docs/experiments/b19_rpca_c2psa_v1.md",
        ),
    )


def main(argv=None):
    """Expose only the single candidate's test, diagnose, and package stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test", "diagnose", "package"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    assert run == ROOT / "runs/detect" / experiment.NAME
    common.completed_run(run)
    data = Path(YAML.load(run / "args.yaml")["data"]).resolve()
    if args.stage == "test":
        finish.evaluate_best(run, data, experiment)
    elif args.stage == "diagnose":
        diagnose(run, data)
    else:
        package(run, data, (args.output or run.with_suffix(".tar.gz")).resolve())


if __name__ == "__main__":
    main()
