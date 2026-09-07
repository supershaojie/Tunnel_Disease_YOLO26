"""Resolve the recorded b19 recipe, audit one SPPF replacement, and run SIR in an isolated output."""

# ruff: noqa: E402 -- Resolve this worktree and offline settings before importing Ultralytics.

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# Match the native CLI: initialize Ultralytics environment settings before the torch thread pool.
import ultralytics

import torch
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA, C3k2, SPPF_SIR
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.torch_utils import autocast, init_seeds

MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-sir-sppf-v1.yaml"
REFERENCE = json.loads((Path(__file__).with_name("b19_reference.json")).read_text(encoding="utf-8"))
REFERENCE["data"]["names"] = {int(k): v for k, v in REFERENCE["data"]["names"].items()}
B19 = re.compile(r"(?<![a-z0-9])b19(?![a-z0-9])", re.IGNORECASE)
MODULE_CONFIG = dict(k=5, n=3, router_channels=16, correction=0.5, layer=9, scale="n")
PRETRAINED_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"


def sha256(path):
    """Hash a file without buffering weights or logs into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    """Write a readable provenance report."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def git(*args):
    """Run read-only Git queries in the imported worktree."""
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT, text=True
    ).strip()


def find_baseline(root, explicit=None):
    """Read an original b19 record, or recover the user-provided archived snapshot explicitly."""
    path = Path(explicit) if explicit else root / "runs/detect" / REFERENCE["args"]["name"] / "args.yaml"
    if not path.is_file():
        return None, copy.deepcopy(REFERENCE["args"])
    raw = YAML.load(path)
    if raw.get("mode") != "train" or raw.get("name") != REFERENCE["args"]["name"]:
        raise ValueError(f"Expected the original b19 args.yaml: {path}")
    return path.resolve(), raw


def resolve_original(value, root, original_root):
    """Resolve in b19's original working directory, then map the same relative path to baseline-root."""
    value = str(value).replace("\\", "/")
    old = str(original_root).rstrip("/")
    if value.startswith(old + "/"):
        mapped = root / value[len(old) + 1 :]
    else:
        mapped = Path(value) if Path(value).is_absolute() else root / value
    if not mapped.is_file():
        raise FileNotFoundError(f"Missing original file: {value}; expected local mapping: {mapped}")
    return mapped.resolve()


def baseline_architecture():
    """Recover the unchanged b19 YAML; model scale is explicit and validated after construction."""
    cfg = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    cfg["scale"] = "n"
    cfg["nc"] = 1
    return cfg


def architecture_signature(cfg):
    """Compare actual graph semantics, normalizing checkpoint string None values."""
    values = {k: cfg.get(k) for k in ("backbone", "head", "scales", "scale", "end2end", "reg_max")}
    return json.dumps(values, sort_keys=True).replace('"None"', "null")


def resolve_recipe(options, model=MODEL):
    """Validate the archived b19 recipe and classify every allowed candidate difference."""
    root = options.baseline_root.resolve()
    if Path(ultralytics.__file__).resolve().parent != ROOT / "ultralytics":
        raise RuntimeError(f"Ultralytics import is outside the experiment worktree: {ultralytics.__file__}")
    path, raw = find_baseline(root, options.baseline_args)
    recovered = path is None
    if recovered:
        path = (options.project or ROOT / "runs/detect") / f"{options.name}_preflight/b19_recovered_args.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        YAML.save(path, raw)
    expected = REFERENCE["args"]
    metadata_keys = {"model", "pretrained", "data", "project", "name", "save_dir", "cfg"}
    mismatches = {
        k: [expected.get(k), raw.get(k)] for k in expected if k not in metadata_keys and raw.get(k) != expected[k]
    }
    missing = sorted(set(expected) - set(raw))
    unknown = sorted(set(raw) - set(DEFAULT_CFG_DICT) - {"save_dir"})
    if unknown or missing or mismatches:
        raise ValueError(
            f"b19 record/version conflict: unknown fields={unknown}; missing fields={missing}; recipe differences={mismatches}"
        )
    if raw.get("resume"):
        raise ValueError(
            "b19 was resumed; its initialization and comparable remaining budget require separate evidence."
        )
    git("merge-base", "--is-ancestor", REFERENCE["source_commit"], "HEAD")
    original_root = Path(expected["project"]).parent.parent.as_posix()
    data = resolve_original(raw["data"], root, original_root)
    data_cfg = YAML.load(data)
    if {k: data_cfg.get(k) for k in REFERENCE["data"]} != REFERENCE["data"]:
        raise ValueError(f"Dataset split/class configuration differs from recorded b19: {data}")
    initial = raw["pretrained"] if isinstance(raw.get("pretrained"), str) else raw["model"]
    if Path(initial).suffix != ".pt":
        raise ValueError("The verified b19 used an initial .pt; pretrained=True alone cannot locate that file.")
    try:
        source = resolve_original(initial, root, original_root)
    except FileNotFoundError:
        if not options.pretrained or not options.pretrained_sha256:
            raise FileNotFoundError(
                "Original initial weight is missing. Relocated --pretrained needs --pretrained-sha256 "
                "from the original b19 initial file; a new file's self-computed hash is not historical evidence."
            ) from None
        source = options.pretrained.resolve()
    if options.pretrained:
        override = options.pretrained.resolve()
        if sha256(override) != sha256(source):
            raise ValueError("--pretrained is not byte-identical to b19's resolved initial file.")
        source = override
    if options.pretrained_sha256.lower() != PRETRAINED_SHA256 or sha256(source) != PRETRAINED_SHA256:
        raise ValueError("Initial weight does not match the supplied original b19 SHA-256.")
    if source.name.lower() in {"best.pt", "last.pt"}:
        raise ValueError("Do not initialize an experiment from a b19 trained checkpoint.")
    # Native check_amp resolves this literal filename in cwd. Seed its cache with the verified original,
    # so a fresh worktree never downloads a different auxiliary checkpoint during native trainer setup.
    amp_weight = ROOT / "yolo26n.pt"
    if not amp_weight.exists():
        shutil.copyfile(source, amp_weight)
    if sha256(amp_weight) != PRETRAINED_SHA256:
        raise ValueError("The worktree AMP-check checkpoint differs from the original b19 weight")
    weights, checkpoint = load_checkpoint(source)
    if len(weights.names) != 80 or architecture_signature(weights.yaml) != architecture_signature(
        baseline_architecture()
    ):
        raise ValueError(
            "Initial checkpoint architecture/scale/classes conflict with b19's recorded COCO YOLO26n source."
        )
    if "crack" in str(weights.names).lower():
        raise ValueError("The initial checkpoint already contains the target crack class.")
    project = (options.project or ROOT / "runs/detect").resolve()
    name = options.name
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("--name must be a single directory name.")
    effective = {k: v for k, v in raw.items() if k not in {"save_dir", "cfg"}}
    effective.update(
        model=str(model),
        pretrained=str(source),
        data=str(data),
        project=str(project),
        name=name,
        resume=False,
        exist_ok=False,
    )
    effective = vars(get_cfg(overrides=effective))
    differences = {k: {"b19": raw.get(k), "candidate": v} for k, v in effective.items() if raw.get(k) != v}
    illegal = set(differences) - metadata_keys
    # New default fields must be exposed rather than silently accepted across source versions.
    if illegal:
        raise ValueError(f"Unexpected effective configuration changes: { {k: differences[k] for k in illegal} }")
    evidence = dict(
        args_path=str(path),
        args_sha256=sha256(path),
        args_source="user-provided archived snapshot recovered from b19_reference.json"
        if recovered
        else "original b19 args.yaml",
        results_sha256=sha256(path.parent / "results.csv") if (path.parent / "results.csv").is_file() else None,
        initial_path=str(source),
        initial_sha256=sha256(source),
        historical_initial_sha256=options.pretrained_sha256,
        checkpoint_metadata={k: checkpoint.get(k) for k in ("date", "version", "epoch")},
        initial_architecture=weights.yaml,
        initial_nc=len(weights.names),
        initial_scale=weights.yaml.get("scale"),
        data_path=str(data),
        data_sha256=sha256(data),
        config_differences=differences,
        excluded_fields={"save_dir": "run output, regenerated", "cfg": "already resolved into args"},
        source_commit=REFERENCE["source_commit"],
        commit=git("rev-parse", "HEAD"),
        import_path=ultralytics.__file__,
        python=sys.version,
        executable=sys.executable,
        numerical_backend=computation_conditions(),
        ultralytics=ultralytics.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_identity=(
            dict(
                logical_device=0,
                name=torch.cuda.get_device_name(0),
                uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
            )
            if torch.cuda.is_available()
            else None
        ),
        launcher="not present in archived b19 package; startup log records native trainer and effective args",
    )
    for item in (data.parent / "metadata").iterdir():
        evidence.setdefault("dataset_metadata_sha256", {})[item.name] = sha256(item)
    dataset = check_det_dataset(str(data), autodownload=False)
    manifest = {}
    for split, expected_count in REFERENCE["dataset_counts"].items():
        folder = Path(dataset[split])
        images = sorted(p for p in folder.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
        if len(images) != expected_count:
            raise ValueError(f"{split} image count {len(images)} != recorded b19 {expected_count}")
        entries = []
        targets = 0
        for image in images:
            label = Path(str(image).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(label)
            targets += len(label.read_text(encoding="utf-8").splitlines())
            entries.append(
                [
                    image.relative_to(data.parent).as_posix(),
                    image.stat().st_size,
                    image.stat().st_mtime_ns,
                    sha256(label),
                ]
            )
        if split in {"val", "test"} and targets != {"val": 2985, "test": 1477}[split]:
            raise ValueError(f"{split} label count differs from b19: {targets}")
        manifest[split] = dict(
            images=len(images),
            targets=targets,
            image_stat_label_content_sha256=hashlib.sha256(json.dumps(entries).encode()).hexdigest(),
        )
    evidence["dataset_manifest"] = manifest
    return raw, effective, evidence


def audit_weights(baseline, candidate, weights, new_prefix="model.9.router.", layer=9):
    """Require every common initialized tensor and every matching source tensor to be identical."""
    bsd, csd = baseline.state_dict(), candidate.state_dict()
    source = weights.float().state_dict() if weights is not None else {}
    changed = [k for k, v in bsd.items() if k not in csd or not torch.equal(v, csd[k])]
    if changed:
        raise AssertionError(f"Common baseline initialization changed: {changed}")
    loaded, unmatched = [], {}
    for key, tensor in source.items():
        if key not in csd:
            unmatched[key] = (
                "absent in both target-nc baseline and candidate" if key not in bsd else "unexpected missing key"
            )
        elif tensor.shape != csd[key].shape:
            if not key.startswith("model.23."):
                raise AssertionError(f"Non-head shape mismatch: {key}")
            unmatched[key] = dict(
                reason="same nc=80 to nc=1 Detect adaptation as native b19 initialization",
                source=list(tensor.shape),
                target=list(csd[key].shape),
            )
        else:
            if not torch.equal(tensor, csd[key]):
                raise AssertionError(f"Matching pretrained tensor was not loaded: {key}")
            loaded.append(key)
    new = {k: list(v.shape) for k, v in candidate.named_parameters() if k not in bsd}
    if not new or any(not k.startswith(new_prefix) for k in new):
        raise AssertionError(f"Unexpected added parameter locations: {new}")
    groups = {}
    for group, indices in (("backbone", range(11)), ("neck", range(11, 23)), ("head", range(23, 24))):
        keys = [k for k in bsd if int(k.split(".")[1]) in indices]
        matches = [k for k in loaded if k in keys]
        groups[group] = dict(
            common_keys=len(keys),
            common_elements=sum(bsd[k].numel() for k in keys),
            loaded_keys=len(matches),
            loaded_elements=sum(bsd[k].numel() for k in matches),
        )
    return dict(
        common_keys=len(bsd),
        common_elements=sum(v.numel() for v in bsd.values()),
        loaded_keys=loaded,
        loaded_elements=sum(source[k].numel() for k in loaded),
        groups=groups,
        unmatched_source_keys=unmatched,
        new_parameters=new,
        target_original_keys=[k for k in bsd if k.startswith(f"model.{layer}.")],
        common_tensor_keys=list(bsd),
        all_common_tensors_equal=True,
        rng_strategy="fork CPU RNG only for new CPU branch construction",
        baseline_parameters=sum(p.numel() for p in baseline.parameters()),
        candidate_parameters=sum(p.numel() for p in candidate.parameters()),
        added_parameters=sum(p.numel() for k, p in candidate.named_parameters() if k in new),
    )


class AuditedTrainer(DetectionTrainer):
    """Use native trainer reconstruction and optimizer construction, with equality audits at both boundaries."""

    block_type = SPPF_SIR
    layer = 9
    new_marker = ".router."
    new_parameters = 14896
    gradient_markers = ("model.9.router.",)

    def validate_new(self):
        """Check the candidate-specific initialization after native trainer setup."""
        router = self.model.model[self.layer].router
        assert torch.count_nonzero(router[-1].weight) == torch.count_nonzero(router[-1].bias) == 0
        assert all(p.requires_grad for p in router.parameters())

    def __init__(self, overrides, _callbacks=None):
        """Atomically own one output directory using the native explicit save_dir extension."""
        output = Path(overrides["project"]) / overrides["name"]
        output.mkdir(parents=True, exist_ok=False)
        super().__init__(
            cfg={**DEFAULT_CFG_DICT, "save_dir": str(output)}, overrides=overrides.copy(), _callbacks=_callbacks
        )

    @property
    def _oom_retries(self):
        """The fixed-batch experiment owns no memory-recovery retries."""
        return 0

    @_oom_retries.setter
    def _oom_retries(self, value):
        """Reject native retry requests inside its catch boundary, before batch or args mutation.

        The pinned b19 loop has no memory-handler hook. Resetting zero is harmless; its first increment
        requests a retry while the original exception is active. Re-raise that exact error locally.
        """
        if value:
            error = sys.exc_info()[1]
            if error is None:
                raise RuntimeError("Experiment fixed-batch training forbids memory recovery retries")
            raise error

    def get_dataset(self):
        """Use the same detection dataset checker while forbidding automatic replacement downloads."""
        return check_det_dataset(self.args.data, autodownload=False)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Audit the actual rebuilt model without consuming the caller's initialization stream twice."""
        with torch.random.fork_rng(devices=[]):
            baseline = super().get_model(copy.deepcopy(baseline_architecture()), weights, verbose=False)
        candidate = super().get_model(cfg, weights, verbose)
        self.weight_audit = audit_weights(baseline, candidate, weights)
        self.initial_common = {
            k: v.detach().cpu().clone() for k, v in candidate.state_dict().items() if ".router." not in k
        }
        assert self.weight_audit["baseline_parameters"] == 2504190
        assert self.weight_audit["candidate_parameters"] == 2519086
        assert type(candidate.model[4]) is C3k2 and type(candidate.model[9]) is self.block_type
        assert type(candidate.model[10]) is C2PSA
        # Native model construction performs a zero-image stride probe and updates BN identically in both models.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            baseline.eval()
            candidate.eval()
            x = torch.randn(1, 3, 64, 96)
            assert_close_tree(baseline(x), candidate(x), atol=0, rtol=0)
            candidate.train()
        return candidate


def optimizer_signature(optimizer):
    """Record actual optimizer settings, excluding parameter identities and group population counts."""
    return dict(
        name=type(optimizer).__name__,
        groups=[{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
    )


def audit_optimizer(trainer):
    """Check the final optimizer includes the router and all effective new parameters, and matches native b19 settings."""
    ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    missing = [
        k
        for k, p in trainer.model.named_parameters()
        if trainer.new_marker in k and (id(p) not in ids or not p.requires_grad)
    ]
    if missing:
        raise AssertionError(f"Router parameters missing/frozen in final optimizer: {missing}")
    with torch.random.fork_rng(devices=[]):
        baseline = trainer.set_model_names_for_load(
            DetectionModel(baseline_architecture(), nc=trainer.data["nc"], verbose=False)
        )
    iterations = (
        math.ceil(len(trainer.train_loader.dataset) / max(trainer.batch_size, trainer.args.nbs)) * trainer.epochs
    )
    decay = trainer.args.weight_decay * trainer.batch_size * trainer.accumulate / trainer.args.nbs
    baseline_optimizer = trainer.build_optimizer(
        baseline, trainer.args.optimizer, trainer.args.lr0, trainer.args.momentum, decay, iterations
    )
    actual, expected = optimizer_signature(trainer.optimizer), optimizer_signature(baseline_optimizer)
    # _setup_scheduler adds initial_lr to the actual groups; it does not change their effective starting lr.
    for group in actual["groups"]:
        group.pop("initial_lr", None)
    if actual != expected or actual["name"] != REFERENCE["observed_optimizer"]["name"]:
        raise AssertionError(f"Actual optimizer differs from b19: {actual} versus {expected}")
    actual["router_groups"] = [
        dict(
            index=i,
            settings={k: v for k, v in group.items() if k != "params"},
            parameters=[
                k
                for k, p in trainer.model.named_parameters()
                if trainer.new_marker in k and any(p is q for q in group["params"])
            ],
        )
        for i, group in enumerate(trainer.optimizer.param_groups)
    ]
    actual["batch"] = trainer.batch_size
    actual["accumulate"] = trainer.accumulate
    actual["nbs"] = trainer.args.nbs
    return actual


def assert_close_tree(a, b, atol=1e-5, rtol=1e-5, path="raw", report=None):
    """Compare complete trees, reporting finite errors against the reference scale, never near-zero ratios."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), f"{path}: expected tensor, got {type(b)}"
        assert (a.shape, a.dtype, a.device) == (b.shape, b.dtype, b.device), (
            f"{path}: reference shape/dtype/device={a.shape}/{a.dtype}/{a.device}; "
            f"actual={b.shape}/{b.dtype}/{b.device}"
        )
        finite = torch.isfinite(a) & torch.isfinite(b)
        delta = (a.double() - b.double()).abs()
        scale = a[finite].double().abs().max().item() if finite.any() else 0.0
        errors = delta[finite]
        outside = (a != b) if atol == rtol == 0 else ~torch.isclose(b, a, atol=atol, rtol=rtol)
        row = dict(
            path=path,
            shape=list(a.shape),
            dtype=str(a.dtype),
            device=str(a.device),
            finite=finite.all().item(),
            max_abs=errors.max().item() if errors.numel() else 0.0,
            mean_abs=errors.mean().item() if errors.numel() else 0.0,
            reference_max_abs=scale,
            scale_floor=1e-6,
            max_error_over_scale=(errors.max().item() if errors.numel() else 0.0) / max(scale, 1e-6),
            outside_tolerance=(outside | ~finite).sum().item(),
            atol=atol,
            rtol=rtol,
        )
        if report is not None:
            report.append(row)
        assert row["finite"] and row["outside_tolerance"] == 0, json.dumps(row)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{path}: dictionary keys differ"
        for key in a:
            assert_close_tree(a[key], b[key], atol, rtol, f"{path}.{key}", report)
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), f"{path}: sequence type/length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close_tree(x, y, atol, rtol, f"{path}[{i}]", report)
    else:
        assert a == b, f"{path}: reference={a!r}, actual={b!r}"


@torch.random.fork_rng(devices=[])
def structural_checks(directory, model=MODEL, block_type=SPPF_SIR, nc=80):
    """Check only layer 9 changed, nano dimensions, parameter counts, and complete zero-router outputs."""
    original = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    expected = copy.deepcopy(original)
    expected["backbone"][9][2] = block_type.__name__
    expected["nc"] = nc
    assert YAML.load(model) == expected
    torch.random.default_generator.manual_seed(42)
    baseline = DetectionModel(baseline_architecture(), verbose=False).eval()
    torch.random.default_generator.manual_seed(42)
    candidate = DetectionModel(str(model), nc=1, verbose=False).eval()
    audit = audit_weights(baseline, candidate, None)
    assert audit["baseline_parameters"] == 2504190 and audit["added_parameters"] == 14896
    assert audit["candidate_parameters"] == 2519086
    assert candidate.yaml["scale"] == "n"
    assert [i for i, m in enumerate(candidate.model) if isinstance(m, block_type)] == [9]
    assert type(candidate.model[4]) is C3k2
    assert type(candidate.model[10]) is C2PSA and candidate.model[-1].nc == 1
    assert candidate.model[9].n == 3 and candidate.model[9].add
    assert candidate.model[9].cv1.conv.in_channels == candidate.model[9].cv2.conv.out_channels == 256
    assert candidate.model[-1].f == baseline.model[-1].f == [16, 19, 22]
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    shapes = []
    for h, w in ((640, 640), (640, 960)):

        def record_shape(module, inputs, output):
            shapes.append([list(inputs[0].shape), list(output.shape)])

        hook = candidate.model[9].register_forward_hook(record_shape)
        try:
            with torch.no_grad():
                x = torch.randn(1, 3, h, w)
                assert_close_tree(baseline(x), candidate(x), atol=0, rtol=0)
        finally:
            hook.remove()
        assert shapes[-1] == [[1, 256, h // 32, w // 32]] * 2
    report = dict(audit, layer9_shapes=shapes, detection_strides=[8, 16, 32], full_network_zero_router_equal=True)
    write_json(Path(directory) / "structural.json", report)
    return report


def gradient_check(trainer, batch, amp):
    """Check real labeled AMP loss and gradients, including the initially zero earlier router gradients."""
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    started = time.perf_counter()
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    batch = trainer.preprocess_batch(batch)
    if batch["cls"].numel() == 0:
        raise ValueError("Preflight batch contains no targets.")
    with autocast(enabled=amp, device=trainer.device.type):
        loss, items = trainer.model(batch)
        total = loss.sum()
    if not torch.isfinite(total):
        raise AssertionError("Nonfinite detection loss.")
    # Unit-scale AMP backward isolates the gradient path from GradScaler's initial overflow search.
    # Formal training retains the native GradScaler unchanged.
    total.backward()
    norms = {}
    for key, p in trainer.model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise AssertionError(f"Nonfinite gradient: {key}")
        if any(marker in key for marker in trainer.gradient_markers):
            if p.grad is None:
                raise AssertionError(f"Disconnected required gradient: {key}")
            norms[key] = p.grad.float().norm().item()
    if not all(norms[k] > 0 for k in norms if trainer.new_marker + "4." in k):
        raise AssertionError(f"Router output layer has no gradient: {norms}")
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 10.0)
    trainer.optimizer.step()
    trainer.ema.update(trainer.model)
    if not all(torch.isfinite(p).all() for p in trainer.model.parameters()):
        raise AssertionError("Nonfinite model after native optimizer step.")
    assert all(any(v > 0 for k, v in norms.items() if marker in k) for marker in trainer.gradient_markers)
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    return dict(
        loss=total.item(),
        components=items.detach().cpu().tolist(),
        new_gradient_norms={k: v for k, v in norms.items() if trainer.new_marker in k},
        gradient_norms=norms,
        seconds=time.perf_counter() - started,
        amp=amp,
        gradient_scale=1,
        optimizer_step=True,
        targets=batch["cls"].numel(),
        image_shape=list(batch["img"].shape),
    )


@contextmanager
def reload_context():
    """Own CPU reference/reload computation settings locally and restore the caller even on failure."""
    threads = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]), torch.no_grad(), autocast(
            False, device="cpu"
        ), torch.backends.mkldnn.flags(enabled=True, deterministic=True, allow_tf32=False), torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=True, allow_tf32=False
        ):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def reload_tensor_info(tensor):
    """Identify the actual input and non-state inference caches without printing their contents."""
    return dict(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        sha256=hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
    )


def reload_attributes(model):
    """Capture module identity, forward-affecting attributes and Detect caches outside state_dict."""
    fields = (
        "n",
        "enabled",
        "num_heads",
        "head_dim",
        "key_dim",
        "scale",
        "add",
        "inplace",
        "end2end",
        "dynamic",
        "export",
        "format",
        "max_det",
        "agnostic_nms",
        "xyxy",
        "nc",
        "reg_max",
        "shape",
        "stride",
        "anchors",
        "strides",
        "eps",
        "momentum",
    )
    result = {}
    for name, module in model.named_modules():
        attrs = dict(type=f"{type(module).__module__}.{type(module).__name__}", training=module.training)
        for key in fields:
            if hasattr(module, key):
                value = getattr(module, key)
                attrs[key] = reload_tensor_info(value) if isinstance(value, torch.Tensor) else value
        result[name] = attrs
    return result


def computation_conditions():
    """Describe reusable backend policy without model-specific tensors or transient inference mode."""
    return dict(
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        omp=os.environ.get("OMP_NUM_THREADS"),
        mkl=os.environ.get("MKL_NUM_THREADS"),
        mkldnn=torch.backends.mkldnn.enabled,
        mkldnn_deterministic=torch.backends.mkldnn.deterministic,
        mkldnn_allow_tf32=torch.backends.mkldnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def reload_conditions(model, x):
    """Record actual FP32 CPU conditions in both processes, including import paths and input digest."""
    return dict(
        python=sys.version,
        executable=sys.executable,
        torch=str(torch.__version__),
        torch_path=torch.__file__,
        ultralytics=ultralytics.__version__,
        ultralytics_path=ultralytics.__file__,
        **computation_conditions(),
        grad_enabled=torch.is_grad_enabled(),
        cpu_autocast=torch.is_autocast_enabled("cpu"),
        model_devices=sorted({str(t.device) for t in model.state_dict().values()}),
        model_float_dtypes=sorted({str(t.dtype) for t in model.state_dict().values() if t.is_floating_point()}),
        all_eval=all(not m.training for m in model.modules()),
        input=reload_tensor_info(x),
    )


def reload_in_process(path, block_type=SPPF_SIR, layer=9, new_marker=".router.", new_parameters=14896):
    """Audit the independently saved snapshot against native YOLO loading before comparing any outputs."""
    import numpy as np
    from ultralytics import YOLO

    path = Path(path)
    reference = torch.load(path.with_name("reload_reference.pt"), map_location="cpu", weights_only=False)
    report = dict(passed=False, state_exact=False, raw=[], fused=[])
    try:
        with reload_context():
            m = YOLO(path)
            assert type(m.model.model[layer]) is block_type
            assert sum(p.numel() for k, p in m.model.named_parameters() if new_marker in k) == new_parameters
            x = reference["x"]
            report["conditions"] = reload_conditions(m.model, x)
            assert_close_tree(reference["conditions"], report["conditions"], path="conditions")
            assert_close_tree(reference["state"], m.model.state_dict(), 0, 0, path="state")
            report.update(state_exact=True, state_keys=len(reference["state"]))
            assert_close_tree(reference["attributes_before"], reload_attributes(m.model), path="attributes_before")
            before = m.model(x)
            assert_close_tree(reference["state"], m.model.state_dict(), 0, 0, path="state_after_forward")
            assert_close_tree(reference["attributes_after"], reload_attributes(m.model), path="attributes_after")
            assert_close_tree(reference["raw"], before, 0, 0, report=report["raw"])
            m.fuse()
            after = m.model(x)
            # Native YOLO26 fuse removes one2many. Keep the entire retained raw branch and decoded output.
            assert after[1]["one2many"] == {}
            assert_close_tree(
                before[1]["one2one"], after[1]["one2one"], 1e-4, 1e-4, path="fused.one2one", report=report["fused"]
            )
            assert_fused_predictions(m.model.model[-1], before, after, report["fused"])
            result = m.predict(np.zeros((64, 96, 3), dtype=np.uint8), imgsz=96, device="cpu", verbose=False)
            assert len(result) == 1 and torch.isfinite(result[0].boxes.data).all()
            report["passed"] = True
    finally:
        write_json(path.with_name("reload_check.json"), report)
    print("fresh-process exact state/raw reload, fuse, prediction passed")


def assert_fused_predictions(head, before, after, report):
    """Compare all decoded anchors and all selected boxes by anchor identity, including near-tied scores."""
    decoded = [head._inference(output[1]["one2one"]).permute(0, 2, 1) for output in (before, after)]
    assert_close_tree(decoded[0], decoded[1], 1e-4, 1e-4, path="fused.dense_decoded", report=report)
    indices = [head.get_topk_index(value[..., 4:], head.max_det)[2].squeeze(-1) for value in decoded]
    # The 64x96 single-class reload probe retains all 126 anchors; no selection boundary is discarded.
    assert head.nc == 1 and decoded[0].shape[1] <= head.max_det
    for output, value, index in zip((before, after), decoded, indices):
        assert_close_tree(head.postprocess(value), output[0], 0, 0, path="fused.native_postprocess", report=report)
        assert torch.equal(index.sort(-1).values, torch.arange(value.shape[1]).expand_as(index))
    aligned = [
        output[0].gather(1, index.argsort(-1).unsqueeze(-1).expand_as(output[0]))
        for output, index in zip((before, after), indices)
    ]
    assert_close_tree(aligned[0], aligned[1], 1e-4, 1e-4, path="fused.decoded_by_anchor", report=report)


def save_reload_check(model, directory, block_type=SPPF_SIR, layer=9, new_marker=".router.", new_parameters=14896):
    """Serialize one FP16 EMA snapshot, retain its FP32 reference, then reload in a fresh process."""
    directory = Path(directory)
    saved = copy.deepcopy(model).cpu().half().eval()
    saved.criterion = None
    path = directory / "preflight.pt"
    torch.save(
        {
            "ema": saved,
            "model": None,
            "train_args": vars(model.args) if not isinstance(model.args, dict) else model.args,
        },
        path,
    )
    saved.float()  # This very snapshot, not a second file load, is the independent loader-FP32 reference.
    with reload_context():
        x = torch.randn(1, 3, 64, 96, device="cpu", dtype=torch.float32)
        reference = dict(
            x=x,
            state={k: v.clone() for k, v in saved.state_dict().items()},
            attributes_before=reload_attributes(saved),
            conditions=reload_conditions(saved, x),
        )
        reference["raw"] = saved(x)
        assert_close_tree(reference["state"], saved.state_dict(), 0, 0, path="snapshot_after_forward")
        reference["attributes_after"] = reload_attributes(saved)
        torch.save(reference, directory / "reload_reference.pt")
        write_json(directory / "reload_reference_conditions.json", reference["conditions"])
    code = """
from tools.experiments.run_b19_sir_sppf import reload_in_process
import sys, importlib
reload_in_process(sys.argv[1], getattr(importlib.import_module(sys.argv[2]), sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(path),
            block_type.__module__,
            block_type.__name__,
            str(layer),
            new_marker,
            str(new_parameters),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    (directory / "reload_process.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    return dict(
        path=str(path),
        sha256=sha256(path),
        fresh_process=True,
        fuse=True,
        prediction=True,
        checkpoint_precision="FP16 EMA",
        state_exact=True,
        reload_raw_equal=True,
        diagnostics=str(directory / "reload_check.json"),
    )


class TeeStream:
    """Keep progress and prints visible while appending them to the durable training log."""

    def __init__(self, console, logfile):
        """Pair the existing terminal stream with the log stream."""
        self.console, self.logfile = console, logfile

    def write(self, text):
        """Write terminal progress and ordinary prints to both destinations."""
        self.console.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        """Flush both streams."""
        self.console.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        """Preserve terminal capabilities expected by the progress display."""
        return getattr(self.console, name)


def runtime_issues(config):
    """Identify resource/environment blockers without changing the b19 recipe or touching other jobs."""
    issues = []
    if str(torch.__version__) != REFERENCE["environment"]["torch"]:
        issues.append(f"b19 torch={REFERENCE['environment']['torch']}; current torch={torch.__version__}")
    if ultralytics.__version__ != REFERENCE["environment"]["ultralytics"]:
        issues.append(f"b19 Ultralytics={REFERENCE['environment']['ultralytics']}; current={ultralytics.__version__}")
    if sys.version.split()[0] != REFERENCE["environment"]["python"]:
        issues.append(f"b19 Python={REFERENCE['environment']['python']}; current={sys.version.split()[0]}")
    if str(config["device"]) != "0" or not torch.cuda.is_available():
        issues.append("Recorded b19 CUDA device 0 is unavailable; CPU or another device is not substituted.")
    else:
        uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        if uuid is None:
            return issues + ["Cannot map the selected logical CUDA device to a physical GPU UUID."]
        gpu_id = str(uuid) if str(uuid).startswith("GPU-") else "GPU-" + str(uuid)
        if torch.cuda.get_device_name(0) != REFERENCE["environment"]["gpu"]:
            issues.append(f"b19 GPU={REFERENCE['environment']['gpu']}; current={torch.cuda.get_device_name(0)}")
        free, _ = torch.cuda.mem_get_info(0)
        if free < 8 * 1024**3:
            issues.append(
                f"Insufficient verified headroom: free={free / 1024**3:.2f} GiB; require 8 GiB for preflight."
            )
        try:
            processes = subprocess.check_output(
                ["nvidia-smi", "-i", gpu_id, "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
            other = [line for line in processes.splitlines() if line.split(",")[0].strip() != str(os.getpid())]
            print(f"Selected physical GPU {gpu_id}; concurrent compute jobs (unchanged): {other}")
        except (OSError, subprocess.CalledProcessError) as exc:
            issues.append(f"Cannot verify GPU occupancy: {exc}")
    return issues


def launcher_evidence(options, raw):
    """Inspect an original native CLI launch record; arbitrary custom launchers require a separate review."""
    path = options.baseline_launcher
    if path is None or not path.is_file():
        path = Path(__file__).with_name("b19_launcher_expanded.txt")
    import shlex

    path = path.resolve()
    content = path.read_text(encoding="utf-8").replace("\\\n", " ")
    commands = []
    for line in content.splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens and Path(tokens[0]).name == "yolo" and "train" in tokens[:3]:
            commands.append(tokens)
    if len(commands) != 1:
        raise ValueError(
            "Launch evidence must contain exactly one original native yolo [detect] train invocation. "
            "Custom Python trainers/callbacks require explicit implementation review."
        )
    command = commands[0]
    values = {}
    for token in command[command.index("train") + 1 :]:
        if token in {"|", ">", "2>&1", "&"}:
            break
        if "=" not in token:
            raise ValueError(f"Unresolved launch token: {token}")
        key, value = token.split("=", 1)
        if "$" in value or "`" in value:
            raise ValueError(f"Resolve the original runtime expansion in the launch record: {token}")
        if key == "cfg":
            raise ValueError("Provide the original expanded CLI configuration, including the values from cfg.")
        from ultralytics.cfg import smart_value

        values[key] = smart_value(value)
    if not B19.search(str(values.get("name", ""))):
        raise ValueError("Original launch record must identify b19 by name.")
    for key, value in values.items():
        if key not in raw or str(value) != str(raw[key]):
            raise ValueError(f"Original launch command differs from args.yaml: {key}={value!r}, args={raw.get(key)!r}")
    expanded = vars(get_cfg(overrides={**values, "task": "detect", "mode": "train"}))
    missing_overrides = {
        k: [expanded.get(k), v] for k, v in raw.items() if k != "save_dir" and str(expanded.get(k)) != str(v)
    }
    if missing_overrides:
        raise ValueError(f"The original command does not reproduce archived effective args: {missing_overrides}")
    return dict(verified=True, path=str(path), sha256=sha256(path), command=command, trainer="native DetectionTrainer")


def preflight(config, evidence, directory, block_type=SPPF_SIR, trainer_type=AuditedTrainer):
    """Use full native setup, then three disposable real batch=32 AMP updates; never run an epoch."""
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "environment.json", evidence)
    trainer = trainer_type(overrides=dict(config, project=str(directory), name="check"))
    trainer.add_callback("on_pretrain_routine_end", final_model_audit)
    trainer._setup_train()
    if len(trainer.train_loader.dataset) != REFERENCE["dataset_counts"]["train"]:
        raise ValueError("Training dataset count differs from archived b19.")
    if len(trainer.test_loader.dataset) != REFERENCE["dataset_counts"]["val"]:
        raise ValueError("Validation dataset count differs from archived b19.")
    write_json(directory / "weights.json", trainer.weight_audit)
    torch.cuda.reset_peak_memory_stats(trainer.device)
    report = dict(optimizer=audit_optimizer(trainer), execution_conditions=computation_conditions(), real_batches=[])
    loader = iter(trainer.train_loader)
    warmup = max(round(trainer.args.warmup_epochs * len(trainer.train_loader)), 100)
    for step in range(3):
        # Native first-epoch warmup: each of these first three steps has accumulate=1.
        for group in trainer.optimizer.param_groups:
            start = trainer.args.warmup_bias_lr if group.get("param_group") == "bias" else 0.0
            group["lr"] = start + (group["initial_lr"] * trainer.lf(0) - start) * step / warmup
            if "momentum" in group:
                group["momentum"] = (
                    trainer.args.warmup_momentum
                    + (trainer.args.momentum - trainer.args.warmup_momentum) * step / warmup
                )
        report["real_batches"].append(gradient_check(trainer, next(loader), bool(config["amp"])))
    norms = report["real_batches"][-1]["new_gradient_norms"]
    assert all(v > 0 for v in norms.values()), f"Earlier router layers did not learn: {norms}"
    assert type(trainer.ema.ema.model[trainer.layer]) is block_type
    assert set(trainer.model.state_dict()) == set(trainer.ema.ema.state_dict())
    assert trainer.ema.updates == 3
    assert all(torch.isfinite(t).all() for t in trainer.ema.ema.state_dict().values())
    report["ema"] = True
    report["reload"] = save_reload_check(
        trainer.ema.ema, directory, block_type, trainer.layer, trainer.new_marker, trainer.new_parameters
    )
    report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(trainer.device)
    report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(trainer.device)
    report["remaining_free_bytes"] = torch.cuda.mem_get_info(trainer.device)[0]
    write_json(directory / "checks.json", report)
    if report["remaining_free_bytes"] < 2 * 1024**3:
        raise RuntimeError("Preflight leaves less than 2 GiB GPU headroom; preserve other jobs and start later")
    for loader in (trainer.train_loader, trainer.test_loader):
        loader.close()
    return report


def final_model_audit(trainer):
    """Audit the actual post-setup optimizer/model before the first formal batch."""
    if trainer.amp != bool(trainer.args.amp):
        raise RuntimeError("Native AMP check changed b19 AMP; stop instead of silently changing the recipe.")
    changed = [k for k, v in trainer.initial_common.items() if not torch.equal(v, trainer.model.state_dict()[k].cpu())]
    if changed:
        raise AssertionError(f"Common tensors changed before the first formal batch: {changed}")
    trainer.validate_new()
    assert type(trainer.ema.ema.model[trainer.layer]) is trainer.block_type
    write_json(trainer.save_dir / "provenance/final_optimizer.json", audit_optimizer(trainer))
    write_json(trainer.save_dir / "provenance/final_weight_audit.json", trainer.weight_audit)
    del trainer.initial_common


def record_completion(trainer):
    """Record native best-checkpoint val metrics and completion only after the training loop returns."""
    validator = trainer.validator
    write_json(
        trainer.save_dir / "val_metrics.json",
        dict(
            split="val",
            results_dict=validator.metrics.results_dict,
            speed=validator.speed,
            images=validator.seen,
            targets=int(validator.metrics.nt_per_class.sum()),
            args=vars(validator.args),
            best_sha256=sha256(trainer.best),
            ap75=float(validator.metrics.box.all_ap[:, 5].mean()),
        ),
    )
    write_json(
        trainer.save_dir / "completed.json",
        dict(
            completed=True,
            epoch=trainer.epoch + 1,
            run=str(trainer.save_dir),
            best_sha256=sha256(trainer.best),
            commit=git("rev-parse", "HEAD"),
        ),
    )


def source_hashes(extra=()):
    """Fingerprint model code, configuration, and explicit experiment entry dependencies."""
    paths = (
        sorted((ROOT / "ultralytics").rglob("*.py"))
        + sorted((ROOT / "ultralytics/cfg").rglob("*.yaml"))
        + [Path(__file__), Path(__file__).with_name("b19_reference.json"), *map(Path, extra)]
    )
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def main(
    argv=None,
    *,
    model=MODEL,
    trainer_type=AuditedTrainer,
    entrypoint=Path(__file__).resolve(),
    name="yolo26n_b19_d1_sir_sppf_v1",
    source_files=(),
    structure_check=structural_checks,
    module_config=MODULE_CONFIG,
):
    """Resolve and run the sole candidate; missing server evidence never becomes a passing receipt."""
    block_type = trainer_type.block_type
    parser = argparse.ArgumentParser(description=f"Audited b19 experiment: {block_type.__name__}")
    parser.add_argument(
        "--baseline-root", type=Path, required=True, help="Original b19 project root; never the experiment worktree"
    )
    parser.add_argument("--baseline-args", type=Path, help="Explicit original b19 args.yaml (wins over discovery)")
    parser.add_argument("--pretrained", type=Path, help="Relocated copy of the SAME resolved initial checkpoint")
    parser.add_argument(
        "--pretrained-sha256", default=PRETRAINED_SHA256, help="Must equal the recorded original b19 SHA-256"
    )
    parser.add_argument("--baseline-launcher", type=Path, help="Original expanded native b19 CLI command/script")
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--name", default=name)
    parser.add_argument(
        "--project", type=Path, help="Independent experiment output project; defaults to this worktree/runs/detect"
    )
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return 0
    options = parser.parse_args(arguments)
    for attr in ("baseline_root", "baseline_args", "pretrained", "baseline_launcher", "project"):
        value = getattr(options, attr)
        if value is not None:
            setattr(options, attr, value.resolve())
    os.chdir(ROOT)
    project = (options.project or ROOT / "runs/detect").resolve()
    if options.stage == "train" and (project / options.name).exists():
        raise FileExistsError(f"Experiment output already exists; no second run or overwrite: {project / options.name}")
    check_root = project / f"{options.name}_preflight"
    check_root.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(check_root / f"{options.stage}.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    try:
        structure_check(check_root, model, block_type)
        raw, config, evidence = resolve_recipe(options, model)
        launcher = launcher_evidence(options, raw)
        evidence["launch_evidence"] = launcher
        # Fingerprint all package/entry source bytes, even before the final commit, and data manifest metadata.
        evidence["source_sha256"] = source_hashes(
            [
                entrypoint,
                Path(__file__).with_name("server_b19_sir_sppf_v1.sh"),
                Path(__file__).with_name("finish_b19_sir_sppf.py"),
                *source_files,
            ]
        )
        signature = hashlib.sha256(json.dumps([config, evidence], sort_keys=True).encode()).hexdigest()
        check_dir = check_root / signature[:20]
        write_json(check_root / "resolved.json", dict(config=config, evidence=evidence, fingerprint=signature))
        print(
            json.dumps(
                dict(config_differences=evidence["config_differences"], initial_sha256=evidence["initial_sha256"]),
                indent=2,
            )
        )
        # Always audit native Trainer.get_model locally, even if the server environment or launch evidence is missing.
        probe = object.__new__(trainer_type)
        probe.args = get_cfg(overrides=config)
        probe.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
        weights, _ = load_checkpoint(evidence["initial_path"])
        init_seeds(config["seed"], deterministic=config["deterministic"])
        probe.get_model(str(model), weights, verbose=False)
        write_json(check_root / "local_weight_audit.json", probe.weight_audit)
        del probe, weights
        issues = runtime_issues(config) if options.stage == "preflight" else []
        if not launcher["verified"]:
            issues.append(launcher["missing"])
        if issues:
            write_json(
                check_root / "missing.json",
                dict(passed=False, missing=issues, structural_passed=True, trainer_get_model_weights_passed=True),
            )
            print("Preflight incomplete:\n- " + "\n- ".join(issues), file=sys.stderr)
            return 2
        receipt = check_dir / "passed.json"
        if options.stage == "train" and not receipt.is_file():
            command = [
                "--baseline-root",
                str(options.baseline_root),
                "--stage",
                "preflight",
                "--project",
                str(project),
                "--name",
                options.name,
            ]
            for attr in ("baseline_args", "pretrained", "pretrained_sha256", "baseline_launcher"):
                value = getattr(options, attr)
                if value is not None:
                    command.extend(["--" + attr.replace("_", "-"), str(value)])
            result = subprocess.run([sys.executable, str(entrypoint), *command], cwd=ROOT)
            write_json(check_root / "preflight_process_status.json", dict(python=result.returncode, stage="preflight"))
            result.check_returncode()
        if options.stage == "preflight":
            if not receipt.is_file():
                if check_dir.exists():
                    # Preserve a failed attempt's logs; a new attempt never reuses its training state.
                    check_dir = Path(tempfile.mkdtemp(prefix=signature[:20] + "-retry-", dir=check_root)) / "attempt"
                preflight(config, evidence, check_dir, block_type, trainer_type)
                write_json(
                    check_dir / "passed.json",
                    dict(fingerprint=signature, passed=True, checks_sha256=sha256(check_dir / "checks.json")),
                )
                # A retry is referenced, not copied over earlier evidence.
                if check_dir / "passed.json" != receipt:
                    write_json(
                        receipt,
                        dict(
                            fingerprint=signature,
                            passed=True,
                            report_dir=str(check_dir),
                            checks_sha256=sha256(check_dir / "checks.json"),
                        ),
                    )
            passed = json.loads(receipt.read_text(encoding="utf-8"))
            checks = Path(passed.get("report_dir", check_dir)) / "checks.json"
            if (
                not passed.get("passed")
                or passed.get("fingerprint") != signature
                or passed.get("checks_sha256") != sha256(checks)
            ):
                raise RuntimeError("Preflight receipt or evidence mismatch")
            print(f"Preflight passed: {receipt}")
            return 0
        passed = json.loads(receipt.read_text(encoding="utf-8"))
        if passed.get("fingerprint") != signature or not passed.get("passed"):
            raise RuntimeError("Stale/mismatched preflight receipt.")
        checks = Path(passed.get("report_dir", check_dir)) / "checks.json"
        if passed.get("checks_sha256") != sha256(checks):
            raise RuntimeError("Preflight evidence checksum mismatch")
        issues = runtime_issues(config)
        required_memory = json.loads(checks.read_text(encoding="utf-8"))["peak_cuda_reserved_bytes"] + 2 * 1024**3
        if torch.cuda.is_available() and torch.cuda.mem_get_info(0)[0] < required_memory:
            issues.append(f"Need measured preflight peak plus 2 GiB reserve: {required_memory} bytes")
        if issues:
            raise RuntimeError("Resources/environment changed after preflight: " + "; ".join(issues))
        # Start from the b19 seed in a fresh trainer; no disposable batch/model/optimizer/EMA is reused.
        trainer = trainer_type(overrides=config)
        provenance = trainer.save_dir / "provenance"
        shutil.copytree(Path(passed.get("report_dir", check_dir)), provenance)
        shutil.copy2(evidence["args_path"], provenance / "b19_original_args.yaml")
        shutil.copy2(model, provenance / model.name)
        shutil.copy2(launcher["path"], provenance / "b19_launcher_expanded.txt")
        shutil.copy2(evidence["data_path"], provenance / "original_data.yaml")
        YAML.save(provenance / "effective.yaml", config)
        shutil.copy2(check_root / "structural.json", provenance / "structural.json")
        shutil.copy2(check_root / "local_weight_audit.json", provenance / "initialization.json")
        write_json(provenance / "module.json", module_config)
        write_json(provenance / "resolved.json", evidence)
        trainer.add_callback("on_pretrain_routine_end", final_model_audit)
        trainer.add_callback("on_train_end", record_completion)
        train_handler = logging.FileHandler(trainer.save_dir / "train.log", encoding="utf-8")
        LOGGER.addHandler(train_handler)
        try:
            with redirect_stdout(TeeStream(sys.stdout, train_handler.stream)), redirect_stderr(
                TeeStream(sys.stderr, train_handler.stream)
            ):
                trainer.train()
        except Exception:
            train_handler.stream.write(traceback.format_exc())
            train_handler.flush()
            raise
        finally:
            LOGGER.removeHandler(train_handler)
            train_handler.close()
        print(f"Experiment completed; validation-selected weights and provenance: {trainer.save_dir}")
        return 0
    except Exception:
        text = traceback.format_exc()
        with (check_root / f"{options.stage}.log").open("a", encoding="utf-8") as stream:
            stream.write(text)
        print(text, file=sys.stderr)
        return 1
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
