"""B19 evidence and equality audits reused from AMP-preflight fix 6f5f1e2.

Only experiment-independent utilities are retained; no SIR/RPCA module or gate policy is imported.
"""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import ultralytics
import torch
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-rsc-c2psa-v1.yaml"
REFERENCE = json.loads(Path(__file__).with_name("b19_reference.json").read_text(encoding="utf-8"))
REFERENCE["data"]["names"] = {int(k): v for k, v in REFERENCE["data"]["names"].items()}
B19 = re.compile(r"(?<![a-z0-9])b19(?![a-z0-9])", re.IGNORECASE)
PRETRAINED_SHA256 = REFERENCE["historical_initial_weight_sha256"]


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
    """Require and read an original b19 record; never substitute an unverified recipe."""
    path = Path(explicit) if explicit else root / "runs/detect" / REFERENCE["args"]["name"] / "args.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Original b19 args.yaml is required: {path}")
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
    require_clean_source()
    root = options.baseline_root.resolve()
    if Path(ultralytics.__file__).resolve().parent != ROOT / "ultralytics":
        raise RuntimeError(f"Ultralytics import is outside the experiment worktree: {ultralytics.__file__}")
    path, raw = find_baseline(root, options.baseline_args)
    expected = REFERENCE["args"]
    metadata_keys = {"model", "pretrained", "data", "project", "name", "save_dir", "cfg"}
    mismatches = {k: [expected.get(k), raw.get(k)] for k in expected if raw.get(k) != expected[k]}
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
        args_source="original b19 args.yaml",
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
    evidence["dataset_manifest"] = dataset_manifest(data)
    return raw, effective, evidence


def dataset_manifest(data):
    """Bind fixed b19 splits to image stats and complete label content, including expected counts."""
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
    return manifest


def audit_weights(baseline, candidate, weights, new_prefix="model.10.m.", layer=10):
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
        rng_strategy="native modules constructed once; constant theta consumes no RNG",
        baseline_parameters=sum(p.numel() for p in baseline.parameters()),
        candidate_parameters=sum(p.numel() for p in candidate.parameters()),
        added_parameters=sum(p.numel() for k, p in candidate.named_parameters() if k in new),
    )


def optimizer_signature(optimizer):
    """Record actual optimizer settings, excluding parameter identities and group population counts."""
    return dict(
        name=type(optimizer).__name__,
        groups=[{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
    )


def audit_optimizer(trainer):
    """Check the final optimizer includes the new_parameter and all effective new parameters, and matches native b19 settings."""
    ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    missing = [
        k
        for k, p in trainer.model.named_parameters()
        if trainer.new_marker in k and (id(p) not in ids or not p.requires_grad)
    ]
    if missing:
        raise AssertionError(f"New parameters missing/frozen in final optimizer: {missing}")
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
    actual["new_parameter_groups"] = [
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


def computation_conditions():
    """Describe reusable backend policy without model-specific tensors or transient inference mode."""
    return dict(
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        omp=os.environ.get("OMP_NUM_THREADS"),
        mkl=os.environ.get("MKL_NUM_THREADS"),
        mkldnn=torch.backends.mkldnn.enabled,
        mkldnn_deterministic=torch.backends.mkldnn.deterministic,
        mkldnn_allow_tf32=getattr(torch.backends.mkldnn, "allow_tf32", None),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
    )


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
    return dict(
        verified=True,
        path=str(path),
        sha256=sha256(path),
        command=command,
        trainer="native DetectionTrainer",
        source="archived expanded launch record; original shell script was not included in the b19 archive",
    )


def source_hashes(extra=()):
    """Fingerprint model code, configuration, and explicit experiment entry dependencies."""
    paths = (
        sorted((ROOT / "ultralytics").rglob("*.py"))
        + sorted((ROOT / "ultralytics/cfg").rglob("*.yaml"))
        + sorted(p for p in (ROOT / "tools/experiments").iterdir() if p.is_file())
        + sorted((ROOT / "tests").glob("test_rsc_c2psa*.py"))
        + list(map(Path, extra))
    )
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def require_clean_source():
    """Bind executable source to HEAD, including ignored/untracked Python files in the existing source inventory."""
    tracked = set(git("ls-files", "-z").split("\0"))
    untracked = {name.replace("\\", "/") for name in source_hashes()} - tracked
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty or untracked:
        raise RuntimeError(
            f"Fixed-commit experiment requires clean tracked files and committed source: {dirty}; {sorted(untracked)}"
        )


def verify_preflight(directory):
    """Validate the preflight actually used by training, independently of optional manual shell invocations."""
    directory = Path(directory)
    receipt = json.loads((directory / "passed.json").read_text(encoding="utf-8"))
    checks = directory / "preflight/checks.json"
    report = json.loads(checks.read_text(encoding="utf-8"))
    if (
        not receipt["passed"]
        or not report["passed"]
        or receipt["commit"] != git("rev-parse", "HEAD")
        or receipt["checks_sha256"] != sha256(checks)
        or receipt["peak_reserved_bytes"] != report["peak_reserved_bytes"]
    ):
        raise RuntimeError(f"Mismatched training preflight evidence: {directory}")
    return receipt
