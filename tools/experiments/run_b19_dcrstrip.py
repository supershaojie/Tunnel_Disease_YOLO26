"""Resolve the recorded b19 recipe, audit one P3 replacement, and run A1 in an isolated output."""

# ruff: noqa: E402 -- Resolve this worktree and offline settings before importing Ultralytics.

from __future__ import annotations

import argparse
import copy
import csv
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
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch

import ultralytics
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C3k2_DCRStrip
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.torch_utils import ModelEMA, autocast, init_seeds

MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-dcrstrip-v1.yaml"
REFERENCE = json.loads((Path(__file__).with_name("b19_reference.json")).read_text(encoding="utf-8"))
REFERENCE["data"]["names"] = {int(k): v for k, v in REFERENCE["data"]["names"].items()}
B19 = re.compile(r"(?<![a-z0-9])b19(?![a-z0-9])", re.IGNORECASE)
MODULE_CONFIG = dict(
    k=7,
    r=1,
    reduction="max(8,ceil(C/32)*8)",
    alpha=0.05,
    enabled=True,
    use_contrast=True,
    adaptive_fusion=True,
    layer=4,
    scale="n",
)


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
    """Find independent b19 identifiers; reject different candidates instead of choosing by timestamp."""
    if explicit:
        candidates = [Path(explicit).resolve()]
    else:
        candidates = []
        for folder in ("runs", "records", "experiments", "artifacts", "logs"):
            base = root / folder
            if base.is_dir():
                candidates.extend(base.rglob("args.yaml"))
    found = {}
    for path in candidates:
        args = YAML.load(path)
        if args.get("mode") != "train" or not B19.search(str(args.get("name", ""))):
            if explicit:
                raise ValueError(f"Not a b19 training args.yaml: {path}")
            continue
        if not (path.parent / "results.csv").is_file():
            raise ValueError(f"Missing paired b19 results.csv: {path.parent}")
        with (path.parent / "results.csv").open(encoding="utf-8-sig") as stream:
            if not any(csv.DictReader(stream)):
                raise ValueError(f"Empty b19 results.csv: {path.parent}")
        signature = json.dumps(args, sort_keys=True)
        found.setdefault(signature, []).append(path)
    if len(found) != 1:
        paths = [str(p) for group in found.values() for p in group]
        raise ValueError(f"Expected one b19 recipe, found {len(found)}. Use --baseline-args. Candidates: {paths}")
    path = next(iter(found.values()))[0]
    return path, YAML.load(path)


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


def resolve_recipe(options):
    """Validate the archived b19 recipe and classify every allowed A1 difference."""
    root = options.baseline_root.resolve()
    if Path(ultralytics.__file__).resolve().parent != ROOT / "ultralytics":
        raise RuntimeError(f"Ultralytics import is outside the A1 worktree: {ultralytics.__file__}")
    path, raw = find_baseline(root, options.baseline_args)
    expected = REFERENCE["args"]
    metadata_keys = {"model", "pretrained", "data", "project", "name", "save_dir", "cfg"}
    mismatches = {
        k: [expected.get(k), raw.get(k)] for k in expected if k not in metadata_keys and raw.get(k) != expected[k]
    }
    unknown = sorted(set(raw) - set(DEFAULT_CFG_DICT) - {"save_dir"})
    if unknown or mismatches:
        raise ValueError(f"b19 record/version conflict: unknown fields={unknown}; recipe differences={mismatches}")
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
    if options.pretrained_sha256 and sha256(source) != options.pretrained_sha256.lower():
        raise ValueError("Initial weight does not match the supplied original b19 SHA-256.")
    if source.name.lower() in {"best.pt", "last.pt"}:
        raise ValueError("Do not initialize A1 from a b19 trained checkpoint.")
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
        model=str(MODEL),
        pretrained=str(source),
        data=str(data),
        project=str(project),
        name=name,
        resume=False,
        exist_ok=False,
    )
    effective = vars(get_cfg(overrides=effective))
    differences = {k: {"b19": raw.get(k), "a1": v} for k, v in effective.items() if raw.get(k) != v}
    illegal = set(differences) - metadata_keys
    # New default fields must be exposed rather than silently accepted across source versions.
    if illegal:
        raise ValueError(f"Unexpected effective configuration changes: { {k: differences[k] for k in illegal} }")
    evidence = dict(
        args_path=str(path),
        args_sha256=sha256(path),
        results_sha256=sha256(path.parent / "results.csv"),
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
        ultralytics=ultralytics.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        launcher="not present in archived b19 package; startup log records native trainer and effective args",
    )
    for item in (data.parent / "metadata").glob("*.json"):
        evidence.setdefault("dataset_metadata_sha256", {})[item.name] = sha256(item)
    return raw, effective, evidence


def audit_weights(baseline, candidate, weights):
    """Require every common initialized tensor and every matching source tensor to be identical."""
    bsd, csd = baseline.state_dict(), candidate.state_dict()
    source = weights.float().state_dict() if weights is not None else {}
    changed = [k for k, v in bsd.items() if k not in csd or not torch.equal(v, csd[k])]
    if changed:
        raise AssertionError(f"Common baseline initialization changed: {changed}")
    loaded, unmatched = [], {}
    for key, tensor in source.items():
        if key not in csd:
            unmatched[key] = "absent in both target-nc baseline and A1" if key not in bsd else "unexpected missing key"
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
    if not new or any(not k.startswith("model.4.dcr.") for k in new):
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
        target_original_keys=[k for k in bsd if k.startswith("model.4.")],
        all_common_tensors_equal=True,
        rng_strategy="fork CPU RNG only for new CPU branch construction",
        baseline_parameters=sum(p.numel() for p in baseline.parameters()),
        candidate_parameters=sum(p.numel() for p in candidate.parameters()),
        added_parameters=sum(p.numel() for k, p in candidate.named_parameters() if k in new),
        gflops="Not reported: generic profiler omits functional diagonal conv2d; each uses dense 7x7 work.",
    )


class AuditedTrainer(DetectionTrainer):
    """Use native trainer reconstruction and optimizer construction, with equality audits at both boundaries."""

    def __init__(self, overrides, _callbacks=None):
        """Atomically own one output directory using the native explicit save_dir extension."""
        output = Path(overrides["project"]) / overrides["name"]
        output.mkdir(parents=True, exist_ok=False)
        super().__init__(
            cfg={**DEFAULT_CFG_DICT, "save_dir": str(output)}, overrides=overrides.copy(), _callbacks=_callbacks
        )

    def _handle_train_memory_error(self, error, epoch):
        """Fixed-budget ablations propagate memory errors before any batch-size mutation."""
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
            k: v.detach().cpu().clone() for k, v in candidate.state_dict().items() if ".dcr." not in k
        }
        return candidate


def optimizer_signature(optimizer):
    """Record actual optimizer settings, excluding parameter identities and group population counts."""
    return dict(
        name=type(optimizer).__name__,
        groups=[{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
    )


def audit_optimizer(trainer):
    """Check the final optimizer includes alpha and all effective new parameters, and matches native b19 settings."""
    ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    missing = [
        k for k, p in trainer.model.named_parameters() if ".dcr." in k and (id(p) not in ids or not p.requires_grad)
    ]
    if missing:
        raise AssertionError(f"DCR parameters missing/frozen in final optimizer: {missing}")
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
    return actual


def assert_close_tree(a, b, atol=1e-5, rtol=1e-5):
    """Compare complete raw one-to-many/one-to-one output structures recursively."""
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.allclose(a, b, atol=atol, rtol=rtol), "Raw output tensor values differ"
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_close_tree(a[key], b[key], atol, rtol)
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_close_tree(x, y, atol, rtol)
    else:
        assert a == b


def structural_checks(directory):
    """Check full-model bypass and shape invariants using no dataset, weights, network, or GPU."""
    init_seeds(42)
    baseline = DetectionModel(baseline_architecture(), verbose=False).eval()
    init_seeds(42)
    candidate = DetectionModel(str(MODEL), verbose=False).eval()
    audit = audit_weights(baseline, candidate, None)
    assert candidate.yaml["scale"] == "n"
    assert [i for i, m in enumerate(candidate.model) if isinstance(m, C3k2_DCRStrip)] == [4]
    assert candidate.model[4].cv1.conv.in_channels == 64 and candidate.model[4].cv2.conv.out_channels == 128
    assert candidate.model[-1].f == baseline.model[-1].f == [16, 19, 22]
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    shapes = []
    candidate.model[4].dcr.enabled = False
    for h, w in ((640, 640), (640, 960)):
        observed = {}

        def record_shape(module, inputs, output):
            observed["p3"] = list(output.shape)

        hook = candidate.model[4].register_forward_hook(record_shape)
        with torch.no_grad():
            x = torch.randn(1, 3, h, w)
            assert_close_tree(baseline(x), candidate(x))
        hook.remove()
        assert observed["p3"] == [1, 128, h // 8, w // 8]
        shapes.append(observed["p3"])
    candidate.model[4].dcr.enabled = True
    with torch.no_grad():
        out = candidate(torch.randn(1, 3, 640, 960))
        assert isinstance(out, tuple)
    report = dict(audit, p3_shapes=shapes, detection_strides=[8, 16, 32], full_network_bypass=True)
    write_json(Path(directory) / "structural.json", report)
    return report


def gradient_check(trainer, batch, amp):
    """Run actual YOLO26 detection loss and one native optimizer step on a target-containing batch."""
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
    # A unit scale tests the gradient path without conflating initial GradScaler scale probing with model overflow.
    total.backward()
    norms = {}
    for key, p in trainer.model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise AssertionError(f"Nonfinite gradient: {key}")
        if ".dcr." in key:
            if p.grad is None or not torch.count_nonzero(p.grad):
                raise AssertionError(f"Inactive full-v1 gradient path: {key}")
            norms[key] = p.grad.float().norm().item()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 10.0)
    trainer.optimizer.step()
    if not all(torch.isfinite(p).all() for p in trainer.model.parameters()):
        raise AssertionError("Nonfinite model after native optimizer step.")
    return dict(
        loss=total.item(),
        components=items.detach().cpu().tolist(),
        new_gradient_norms=norms,
        amp=amp,
        optimizer_step=True,
        targets=batch["cls"].numel(),
        image_shape=list(batch["img"].shape),
    )


def save_reload_check(model, directory):
    """Save and reload through YOLO in a fresh Python process, then verify ordinary and fused prediction."""
    directory = Path(directory)
    saved = copy.deepcopy(model).cpu().eval()
    saved.criterion = None
    path = directory / "preflight.pt"
    torch.save(
        {"model": saved, "train_args": vars(model.args) if not isinstance(model.args, dict) else model.args}, path
    )
    code = """
import sys, torch, numpy as np
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2_DCRStrip
torch.set_num_threads(4)
m = YOLO(sys.argv[1])
assert isinstance(m.model.model[4], C3k2_DCRStrip)
assert m.model.model[4].dcr.enabled and m.model.model[4].dcr.use_contrast and m.model.model[4].dcr.adaptive_fusion
x = torch.randn(1, 3, 64, 96)
m.model.eval()
with torch.no_grad():
    before = m.model(x)[0]
    m.fuse()
    after = m.model(x)[0]
assert before.shape == after.shape and torch.allclose(before, after, atol=1e-4, rtol=1e-4)
result = m.predict(np.zeros((64, 96, 3), dtype=np.uint8), imgsz=96, device='cpu', verbose=False)
assert len(result) == 1 and torch.isfinite(result[0].boxes.data).all()
print('fresh-process YOLO reload, fuse, prediction passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    return dict(path=str(path), sha256=sha256(path), fresh_process=True, fuse=True, prediction=True)


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
            if other:
                issues.append(f"GPU already has compute jobs; no jobs were stopped: {other}")
        except (OSError, subprocess.CalledProcessError) as exc:
            issues.append(f"Cannot verify GPU occupancy: {exc}")
    return issues


def launcher_evidence(options, raw):
    """Inspect an original native CLI launch record; arbitrary custom launchers require a separate review."""
    if not options.baseline_launcher:
        return {
            "verified": False,
            "missing": "Original b19 launch command/script (not included in archived package). "
            "Supply --baseline-launcher with the original native yolo train command/script to rule out custom callbacks.",
        }
    import shlex

    path = options.baseline_launcher.resolve()
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


def preflight(config, evidence, directory):
    """Build the native trainer pipeline and perform disposable real-batch checks without running epochs."""
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "environment.json", evidence)
    # Only the output destination differs. No batch, resolution, AMP, seed, or training budget override is used here.
    options = dict(config, project=str(directory), name="check")
    trainer = AuditedTrainer(overrides=options)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.stride = int(trainer.model.stride.max())
    # Native b19 has freeze=None and reg_max=1, so there are no frozen DFL parameters to reproduce here.
    if trainer.args.freeze is not None:
        raise ValueError("The recorded b19 had freeze=None; a different freeze recipe needs separate validation.")
    trainer._build_train_pipeline()
    trainer.set_class_weights()
    if len(trainer.train_loader.dataset) != REFERENCE["dataset_counts"]["train"]:
        raise ValueError("Training dataset count differs from archived b19.")
    if len(trainer.test_loader.dataset) != REFERENCE["dataset_counts"]["val"]:
        raise ValueError("Validation dataset count differs from archived b19.")
    write_json(directory / "weights.json", trainer.weight_audit)
    optimizer = audit_optimizer(trainer)
    batch = next(iter(trainer.train_loader))
    if batch["cls"].numel() == 0:
        raise ValueError("First real augmented batch has no labels; inspect the b19 dataset.")
    report = dict(optimizer=optimizer, real_batch=gradient_check(trainer, batch, bool(config["amp"])))
    ema = ModelEMA(trainer.model)
    ema.update(trainer.model)
    assert isinstance(ema.ema.model[4], C3k2_DCRStrip)
    report["ema"] = True
    report["reload"] = save_reload_check(trainer.model, directory)
    with torch.no_grad():
        block = trainer.model.model[4].dcr.eval()
        feature = torch.randn(1, 128, 17, 23, device=trainer.device)
        delta, gates = block.residual(feature)
        report["diagnostics"] = dict(
            alpha=block.alpha.item(),
            gate_means=gates.mean((0, 2, 3)).cpu().tolist(),
            residual_main_norm_ratio=(block.alpha * delta).norm().item() / feature.norm().item(),
        )
    write_json(directory / "checks.json", report)
    # Dataloader workers and their RNG state die with the preflight subprocess before a fresh training process starts.
    return report


def final_model_audit(trainer):
    """Audit the actual post-setup optimizer/model before the first formal batch."""
    if trainer.amp != bool(trainer.args.amp):
        raise RuntimeError("Native AMP check changed b19 AMP; stop instead of silently changing the recipe.")
    changed = [k for k, v in trainer.initial_common.items() if not torch.equal(v, trainer.model.state_dict()[k].cpu())]
    if changed:
        raise AssertionError(f"Common tensors changed before the first formal batch: {changed}")
    write_json(trainer.save_dir / "provenance/final_optimizer.json", audit_optimizer(trainer))
    write_json(trainer.save_dir / "provenance/final_weight_audit.json", trainer.weight_audit)
    del trainer.initial_common


def main(argv=None):
    """Run isolated preflight or training; report missing evidence and fail without guessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root", type=Path, required=True, help="Original b19 project root; never the A1 worktree"
    )
    parser.add_argument("--baseline-args", type=Path, help="Explicit original b19 args.yaml (wins over discovery)")
    parser.add_argument("--pretrained", type=Path, help="Relocated copy of the SAME resolved initial checkpoint")
    parser.add_argument("--pretrained-sha256", help="Original b19 initial digest; required if its old path is missing")
    parser.add_argument("--baseline-launcher", type=Path, help="Original expanded native b19 CLI command/script")
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--name", default="yolo26n_b19_a1_dcrstrip_v1")
    parser.add_argument(
        "--project", type=Path, help="Independent A1 output project; defaults to this worktree/runs/detect"
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
    project = (options.project or ROOT / "runs/detect").resolve()
    check_root = project / f"{options.name}_preflight"
    check_root.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(check_root / f"{options.stage}.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    try:
        structural_checks(check_root)
        raw, config, evidence = resolve_recipe(options)
        launcher = launcher_evidence(options, raw)
        evidence["launch_evidence"] = launcher
        # Fingerprint all package/entry source bytes, even before the final commit, and data manifest metadata.
        source_paths = (
            sorted((ROOT / "ultralytics").rglob("*.py"))
            + sorted((ROOT / "ultralytics/cfg").rglob("*.yaml"))
            + [
                Path(__file__),
                Path(__file__).with_name("b19_reference.json"),
            ]
        )
        evidence["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
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
        probe = object.__new__(AuditedTrainer)
        probe.args = get_cfg(overrides=config)
        probe.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
        weights, _ = load_checkpoint(evidence["initial_path"])
        init_seeds(config["seed"], deterministic=config["deterministic"])
        probe.get_model(str(MODEL), weights, verbose=False)
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
            subprocess.run([sys.executable, str(Path(__file__)), *command], cwd=ROOT, check=True)
        if options.stage == "preflight":
            if not receipt.is_file():
                if check_dir.exists():
                    # Preserve a failed attempt's logs; a new attempt never reuses its training state.
                    check_dir = Path(tempfile.mkdtemp(prefix=signature[:20] + "-retry-", dir=check_root)) / "attempt"
                preflight(config, evidence, check_dir)
                write_json(check_dir / "passed.json", dict(fingerprint=signature, passed=True))
                # A retry is referenced, not copied over earlier evidence.
                if check_dir / "passed.json" != receipt:
                    write_json(receipt, dict(fingerprint=signature, passed=True, report_dir=str(check_dir)))
            print(f"Preflight passed: {receipt}")
            return 0
        passed = json.loads(receipt.read_text(encoding="utf-8"))
        if passed.get("fingerprint") != signature or not passed.get("passed"):
            raise RuntimeError("Stale/mismatched preflight receipt.")
        issues = runtime_issues(config)
        if issues:
            raise RuntimeError("Resources/environment changed after preflight: " + "; ".join(issues))
        # Start from the b19 seed in a fresh trainer; no disposable batch/model/optimizer/EMA is reused.
        trainer = AuditedTrainer(overrides=config)
        provenance = trainer.save_dir / "provenance"
        shutil.copytree(Path(passed.get("report_dir", check_dir)), provenance)
        shutil.copy2(evidence["args_path"], provenance / "b19_original_args.yaml")
        shutil.copy2(MODEL, provenance / MODEL.name)
        shutil.copy2(evidence["data_path"], provenance / "original_data.yaml")
        YAML.save(provenance / "a1_effective.yaml", config)
        write_json(provenance / "module.json", MODULE_CONFIG)
        write_json(provenance / "resolved.json", evidence)
        trainer.add_callback("on_pretrain_routine_end", final_model_audit)
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
        print(f"A1 completed; validation-selected weights and provenance: {trainer.save_dir}")
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
