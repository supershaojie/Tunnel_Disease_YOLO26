"""Resolve the recorded b19 recipe, audit the first P5/P4 concatenation, and run CCA in an isolated output."""

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
import traceback
import warnings
from datetime import datetime, timezone
from contextlib import redirect_stderr, redirect_stdout
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
from ultralytics.nn.modules import C2PSA, C3k2, Concat_CCA_Fusion
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.torch_utils import autocast, init_seeds

NAME = "yolo26n_b19_cca_fusion_v1"
MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-cca-fusion-v1.yaml"
REFERENCE = json.loads((Path(__file__).with_name("b19_reference.json")).read_text(encoding="utf-8"))
REFERENCE["data"]["names"] = {int(k): v for k, v in REFERENCE["data"]["names"].items()}
B19 = re.compile(r"(?<![a-z0-9])b19(?![a-z0-9])", re.IGNORECASE)
MODULE_CONFIG = dict(
    layer=12,
    inputs=[11, 6, 10],
    scale="n",
    rank=16,
    kernel=3,
    temperature=0.25,
    formula="Y = cat(U + Wo(sum(a*(Vj-Vparent))), L)",
)
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
    """Require the original b19 args record; never synthesize missing source evidence."""
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
    """Validate the archived b19 recipe and classify every allowed CCA difference."""
    root = options.baseline_root.resolve()
    if Path(ultralytics.__file__).resolve().parent != ROOT / "ultralytics":
        raise RuntimeError(f"Ultralytics import is outside the CCA worktree: {ultralytics.__file__}")
    module_path = Path(sys.modules["ultralytics.nn.modules.cca_fusion"].__file__).resolve()
    if (
        module_path != ROOT / "ultralytics/nn/modules/cca_fusion.py"
        or Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT
    ):
        raise RuntimeError("CCA module or Git root is outside the current worktree")
    path, raw = find_baseline(root, options.baseline_args)
    if sha256(path) != REFERENCE["args_sha256"]:
        raise ValueError("Original b19 args.yaml SHA256 mismatch")
    results = path.parent / "results.csv"
    if sha256(results) != "44795068d0ebc687c29ff7b7b8cbf337469546b2d4dd515cb0616c270bdf9507":
        raise ValueError("Original b19 results.csv SHA256 mismatch")
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
        raise ValueError("Do not initialize CCA from a b19 trained checkpoint.")
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
    differences = {k: {"b19": raw.get(k), "cca": v} for k, v in effective.items() if raw.get(k) != v}
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
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        torch_num_threads=torch.get_num_threads(),
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
        module_path=str(Path(sys.modules["ultralytics.nn.modules.cca_fusion"].__file__).resolve()),
        config_comparison={
            k: dict(b19=raw.get(k), cca=effective.get(k), equal=raw.get(k) == effective.get(k))
            for k in sorted(set(raw) | set(effective))
        },
    )
    for item in (data.parent / "metadata").iterdir():
        evidence.setdefault("dataset_metadata_sha256", {})[item.name] = sha256(item)
    evidence["dataset_manifest"] = dataset_manifest(data)
    expected_manifest = json.loads(Path(__file__).with_name("b19_dataset_manifest.json").read_text(encoding="utf-8"))
    if evidence["dataset_manifest"] != expected_manifest:
        raise ValueError("Dataset image/label content differs from the independently read local b19 dataset archive")
    for split, expected in REFERENCE["dataset_counts"].items():
        actual = evidence["dataset_manifest"][split]
        if actual["images"] != expected or (
            split in {"val", "test"} and actual["targets"] != {"val": 2985, "test": 1477}[split]
        ):
            raise ValueError(f"Dataset counts differ from b19 for {split}: {actual}")
    return raw, effective, evidence


def dataset_manifest(data):
    """Use one content-based dataset identity for training, evaluation and packaging."""
    dataset = check_det_dataset(str(data), autodownload=False)
    manifest = {}
    for split in REFERENCE["dataset_counts"]:
        folder = Path(dataset[split])
        images = sorted(p for p in folder.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
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
                    sha256(image),
                    sha256(label),
                ]
            )
        manifest[split] = dict(
            images=len(images),
            targets=targets,
            image_stat_label_content_sha256=hashlib.sha256(json.dumps(entries).encode()).hexdigest(),
        )
    return manifest


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
            unmatched[key] = "absent in both target-nc baseline and CCA" if key not in bsd else "unexpected missing key"
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
    if not new or any(not k.startswith("model.12.") for k in new):
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
        target_original_keys=[k for k in bsd if k.startswith("model.12.")],
        all_common_tensors_equal=True,
        rng_strategy="fork CPU RNG only for new CPU branch construction",
        baseline_parameters=sum(p.numel() for p in baseline.parameters()),
        candidate_parameters=sum(p.numel() for p in candidate.parameters()),
        added_parameters=sum(p.numel() for k, p in candidate.named_parameters() if k in new),
    )


class AuditedTrainer(DetectionTrainer):
    """Use native trainer reconstruction and optimizer construction, with equality audits at both boundaries."""

    block_type = Concat_CCA_Fusion

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
                raise RuntimeError("CCA fixed-batch training forbids memory recovery retries")
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
            k: v.detach().cpu().clone() for k, v in candidate.state_dict().items() if not k.startswith("model.12.")
        }
        assert self.weight_audit["baseline_parameters"] == 2504190
        assert self.weight_audit["candidate_parameters"] == 2518526
        assert type(candidate.model[4]) is C3k2 and type(candidate.model[12]) is self.block_type
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
    """Check the final optimizer includes the cca and all effective new parameters, and matches native b19 settings."""
    for key, parameter in trainer.model.named_parameters():
        if not key.startswith("model.12."):
            continue
        groups = [group for group in trainer.optimizer.param_groups for value in group["params"] if value is parameter]
        assert parameter.requires_grad and len(groups) == 1, f"Missing/frozen/duplicated CCA parameter: {key}"
        group = groups[0]
        assert group["param_group"] == ("muon" if parameter.ndim >= 2 else "bias"), key
        assert bool(group.get("use_muon", False)) == (parameter.ndim >= 2), key
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
    actual["cca_groups"] = [
        dict(
            index=i,
            settings={k: v for k, v in group.items() if k != "params"},
            parameters=[
                k
                for k, p in trainer.model.named_parameters()
                if k.startswith("model.12.") and any(p is q for q in group["params"])
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
def structural_checks(directory, model=MODEL, block_type=Concat_CCA_Fusion, nc=1):
    """Check only layer 12 changed, nano dimensions, parameter counts, and complete zero-cca outputs."""
    original = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    expected = copy.deepcopy(original)
    expected["head"][1] = [[11, 6, 10], 1, block_type.__name__, []]
    expected["nc"] = nc
    assert YAML.load(model) == expected
    torch.random.default_generator.manual_seed(42)
    baseline = DetectionModel(baseline_architecture(), verbose=False).eval()
    torch.random.default_generator.manual_seed(42)
    candidate = DetectionModel(str(model), nc=1, verbose=False).eval()
    audit = audit_weights(baseline, candidate, None)
    assert audit["baseline_parameters"] == 2504190 and audit["added_parameters"] == 14336
    assert audit["candidate_parameters"] == 2518526
    assert candidate.yaml["scale"] == "n"
    assert [i for i, m in enumerate(candidate.model) if isinstance(m, block_type)] == [12]
    assert type(candidate.model[4]) is C3k2
    assert type(candidate.model[10]) is C2PSA and candidate.model[-1].nc == 1
    assert candidate.model[12].Wq.in_channels == 128
    assert candidate.model[12].Wk.in_channels == candidate.model[12].Wo.out_channels == 256
    assert candidate.model[13].cv1.conv.in_channels == 384
    assert candidate.model[-1].reg_max == 1 and candidate.end2end
    assert candidate.model[-1].f == baseline.model[-1].f == [16, 19, 22]
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    shapes = []
    for h, w in ((640, 640), (640, 960)):

        def record_shape(module, inputs, output):
            shapes.append([*[list(t.shape) for t in inputs[0]], list(output.shape)])

        hook = candidate.model[12].register_forward_hook(record_shape)
        try:
            with torch.no_grad():
                x = torch.randn(1, 3, h, w)
                assert_close_tree(baseline(x), candidate(x), atol=0, rtol=0)
        finally:
            hook.remove()
        assert shapes[-1] == [
            [1, 256, h // 16, w // 16],
            [1, 128, h // 16, w // 16],
            [1, 256, h // 32, w // 32],
            [1, 384, h // 16, w // 16],
        ]
    report = dict(audit, layer12_shapes=shapes, detection_strides=[8, 16, 32], full_network_zero_cca_equal=True)
    write_json(Path(directory) / "structural.json", report)
    return report


def save_reload_check(model, directory, block_type=Concat_CCA_Fusion):
    """Compare an independent FP16 snapshot with fresh-process reload, then native-controlled FP32 fusion."""
    directory = Path(directory)
    saved = copy.deepcopy(model).cpu().half().eval()
    saved.criterion = None
    path = directory / "preflight.pt"
    torch.save(
        {
            "ema": saved,
            "model": None,
            "train_args": dict(vars(model.args)) if not isinstance(model.args, dict) else model.args,
        },
        path,
    )
    saved.float()
    # Own the same CPU backend settings in both processes without changing formal training settings.
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
            x = torch.randn(1, 3, 64, 96)
            torch.save(dict(x=x, state=saved.state_dict(), raw=saved(x)), directory / "reload_reference.pt")
    finally:
        torch.set_num_threads(threads)
    code = (
        "from tools.experiments.run_b19_cca_fusion import reload_in_process; import sys; reload_in_process(sys.argv[1])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    (directory / "reload_process.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        print(result.stdout + result.stderr, file=sys.stderr)
    result.check_returncode()
    return json.loads((directory / "reload_check.json").read_text(encoding="utf-8"))


def reload_in_process(path):
    """Load through the public YOLO facade and check every retained fused head output on independent copies."""
    from ultralytics import YOLO

    path = Path(path)
    reference = torch.load(path.with_name("reload_reference.pt"), map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
        model = YOLO(path).model.float().eval()
        assert type(model.model[12]) is Concat_CCA_Fusion
        assert_close_tree(reference["state"], model.state_dict(), 0, 0)
        x = reference["x"]
        assert_close_tree(reference["raw"], model(x), 0, 0)
        native = DetectionModel(baseline_architecture(), verbose=False).eval()
        native.load_state_dict(
            {k: v for k, v in model.state_dict().items() if not k.startswith("model.12.")}, strict=True
        )
        report = dict(
            passed=True, state_exact=True, raw_exact=True, precision="FP16 snapshot reloaded in FP32", fuse={}
        )
        for name, original in (("native_control", native), ("cca", model)):
            unfused, fused = copy.deepcopy(original), copy.deepcopy(original)
            calls = []
            handle = fused.model[12].register_forward_hook(lambda *args: calls.append(True)) if name == "cca" else None
            try:
                before = unfused(x)
                after = fused.fuse(verbose=False)(x)
                if handle is not None:
                    assert calls
                    report["fused_cca_calls"] = len(calls)
            finally:
                if handle is not None:
                    handle.remove()
            assert after[1]["one2many"] == {}
            errors = []
            assert_close_tree(before[1]["one2one"], after[1]["one2one"], 1e-4, 1e-4, report=errors)
            decoded_before = unfused.model[-1]._inference(before[1]["one2one"])
            decoded_after = fused.model[-1]._inference(after[1]["one2one"])
            assert_close_tree(decoded_before, decoded_after, 1e-4, 1e-4, path="decoded_by_anchor", report=errors)
            # At 64x96 all 126 anchors are retained. Nearly tied scores may reorder top-k rows after fusion.
            # Compare the same anchors, retaining the original tolerance and every selected box/score/class.
            orders = []
            indices = []
            for head, decoded in ((unfused.model[-1], decoded_before), (fused.model[-1], decoded_after)):
                index = head.get_topk_index(decoded[:, 4:].permute(0, 2, 1), head.max_det)[2].squeeze(-1)
                indices.append(index)
                orders.append(index.argsort(dim=1))
            assert_close_tree(indices[0].sort()[0], indices[1].sort()[0], 0, 0)
            aligned = [
                value[0].gather(1, order.unsqueeze(-1).expand(-1, -1, 6))
                for value, order in zip((before, after), orders)
            ]
            assert_close_tree(*aligned, 1e-4, 1e-4, path="selected_by_anchor", report=errors)
            report.setdefault("topk_reordered_rows", {})[name] = (indices[0] != indices[1]).sum().item()
            report["fuse"][name] = errors
    write_json(path.with_name("reload_check.json"), report)


PREFLIGHT_MAX_BATCHES = 128
CCA_PARAMETERS = ["Wq.weight", "Wk.weight", "Wv.weight", "Wo.weight"]


def observation_report(names=CCA_PARAMETERS, max_batches=PREFLIGHT_MAX_BATCHES):
    """Create the bounded observation ledger before any Trainer setup can fail."""
    return dict(
        passed=False,
        stage="initialization",
        max_batches=max_batches,
        batches_observed=0,
        attempted_steps=0,
        overflow_skips=0,
        successful_steps=0,
        real_batches=[],
        first_finite_nonzero_gradient_step={k: None for k in names},
        first_effective_update_step={k: None for k in names},
        missing_effective_parameters=list(names),
    )


def record_failure(report, exc):
    """Attach the active failure boundary and distinguish the evidence missing for each parameter."""
    reasons = {}
    for name in report["missing_effective_parameters"]:
        rows = [batch["parameters"][name] for batch in report["real_batches"] if name in batch.get("parameters", {})]
        if any(row.get("registrations") != 1 for row in rows):
            reason = "parameter group missing or duplicated"
        elif any(row.get("gradient_before_clip", {}).get("is_none") for row in rows):
            reason = "disconnected computation graph"
        elif report["successful_steps"] == 0 and report["overflow_skips"]:
            reason = "AMP overflow skipped all optimizer steps"
        elif report["first_finite_nonzero_gradient_step"][name] is None:
            reason = "no finite nonzero task gradient yet; inspect upstream zero-projection unlocking and failure stage"
        else:
            reason = "task gradient observed but no representable task-attributable update; inspect LR, clipping, scale and zero-task replay"
        reasons[name] = reason
    report.update(
        passed=False,
        failure_stage=report["stage"],
        exception=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()),
        missing_reasons=reasons,
    )


def tensor_stats(value):
    """Summarize detached diagnostics without emitting nonstandard JSON NaN/Infinity."""
    if value is None:
        return dict(is_none=True, finite=None, norm=None, max_abs=None, nonzero=0, dtype=None)
    value = value.detach()
    finite = bool(torch.isfinite(value).all())
    return dict(
        is_none=False,
        finite=finite,
        norm=value.double().norm().item() if finite else None,
        max_abs=value.abs().max().item() if finite else None,
        nonzero=torch.count_nonzero(value).item(),
        dtype=str(value.dtype),
    )


def parameter_change(before, after):
    """Measure exact element changes, including updates smaller than allclose tolerances."""
    return dict(
        changed=not torch.equal(before, after),
        changed_elements=(before != after).sum().item(),
        max_abs_delta=(after.detach().double() - before.double()).abs().max().item(),
    )


def observed_optimizer_step(trainer, params, report, row):
    """Observe the native unscale/clip/step/update/zero/EMA sequence without changing its arithmetic."""
    optimizer = trainer.optimizer
    report["attempted_steps"] += 1
    step = report["attempted_steps"]  # One-based attempted optimizer steps; batch indices remain zero-based.
    row.update(attempted_step=step, scale_before=trainer.scaler.get_scale(), parameters={})
    report["stage"] = "optimizer_membership"
    groups = {}
    for name, p in params.items():
        matches = [(i, g) for i, g in enumerate(optimizer.param_groups) for q in g["params"] if q is p]
        row["parameters"][name] = info = dict(
            registrations=len(matches), dtype=str(p.dtype), requires_grad=p.requires_grad
        )
        assert len(matches) == 1, f"{name}: expected exactly one optimizer registration, got {len(matches)}"
        assert p.dtype == torch.float32 and p.requires_grad, f"{name}: invalid dtype/requires_grad"
        index, groups[name] = matches[0]
        info["optimizer_group"] = dict(
            index=index,
            **{
                k: groups[name].get(k)
                for k in ("param_group", "lr", "initial_lr", "momentum", "weight_decay", "use_muon", "nesterov")
            },
        )
    report["stage"] = "unscale_clip"
    trainer.scaler.unscale_(optimizer)
    finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in trainer.model.parameters())
    row["finite_gradients"] = finite
    before = {k: p.detach().clone() for k, p in params.items()}
    for name, p in params.items():
        row["parameters"][name]["gradient_before_clip"] = tensor_stats(p.grad)
    norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 10.0)
    coefficient = torch.clamp(10.0 / (norm + 1e-6), max=1.0)
    row["total_gradient_norm_before_clip"] = norm.item() if torch.isfinite(norm) else str(norm.item())
    row["clip_coefficient"] = coefficient.item() if torch.isfinite(coefficient) else str(coefficient.item())
    no_task = {}
    for name, p in params.items():
        info = row["parameters"][name]
        info["gradient_after_clip"] = stats = tensor_stats(p.grad)
        pre = info["gradient_before_clip"]
        if pre["finite"] and pre["nonzero"] and report["first_finite_nonzero_gradient_step"][name] is None:
            report["first_finite_nonzero_gradient_step"][name] = step
        if finite and stats["finite"]:
            replica = torch.nn.Parameter(p.detach().clone())
            replica.grad = torch.zeros_like(p)
            shadow = type(optimizer)([dict(groups[name], params=[replica])], muon=optimizer.muon, sgd=optimizer.sgd)
            shadow.state[replica] = copy.deepcopy(optimizer.state.get(p, {}))
            shadow.step()
            no_task[name] = replica.detach()
    report["stage"] = "optimizer_step"
    updates = []
    hook = optimizer.register_step_post_hook(lambda *args: updates.append(True))
    try:
        trainer.scaler.step(optimizer)
        row["optimizer_step"] = bool(updates)
        report["successful_steps"] += int(bool(updates))
        trainer.scaler.update()
        row["scale_after"] = trainer.scaler.get_scale()
        overflow = not updates and row["scale_after"] < row["scale_before"]
        report["overflow_skips"] += int(overflow)
        row["overflow_skip"] = overflow
        optimizer.zero_grad()
        if trainer.ema:
            trainer.ema.update(trainer.model)
    finally:
        hook.remove()
    for name, p in params.items():
        info = row["parameters"][name]
        info.update(parameter_change(before[name], p))
        pre, post = info["gradient_before_clip"], info["gradient_after_clip"]
        eligible = bool(updates) and finite and pre["finite"] and pre["nonzero"] and post["finite"] and post["nonzero"]
        info["different_from_zero_task_replay"] = name in no_task and not torch.equal(no_task[name], p)
        if (
            eligible
            and info["changed"]
            and info["different_from_zero_task_replay"]
            and report["first_effective_update_step"][name] is None
        ):
            report["first_effective_update_step"][name] = step
    report["missing_effective_parameters"] = [k for k, v in report["first_effective_update_step"].items() if v is None]
    row["missing_effective_parameters"] = list(report["missing_effective_parameters"])
    row["first_finite_nonzero_gradient_step"] = dict(report["first_finite_nonzero_gradient_step"])
    row["first_effective_update_step"] = dict(report["first_effective_update_step"])
    assert all(not info["gradient_before_clip"]["is_none"] for info in row["parameters"].values()), (
        "Disconnected CCA gradient"
    )
    assert finite or (not updates and overflow), "Nonfinite gradients were not skipped by GradScaler"
    assert updates or overflow, "No optimizer step without an AMP overflow"


def observe_batches(trainer, params, loader, report, checks_path):
    """Stop at the first complete task-update evidence or fail at the fixed batch budget."""
    import numpy as np

    try:
        report["stage"] = "epoch_setup"
        trainer.optimizer.zero_grad()
        trainer.epoch = trainer.start_epoch
        # Native _do_train steps the scheduler once before the first batch, then enters model train mode.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer.scheduler.step()
        trainer._model_train()
        warmup = (
            max(round(trainer.args.warmup_epochs * len(trainer.train_loader)), 100)
            if trainer.args.warmup_epochs > 0
            else -1
        )
        last_step = -1
        for ni in range(report["max_batches"]):
            row = dict(batch=ni, attempted=False)
            report["real_batches"].append(row)
            report["stage"] = "batch_forward_backward"
            if ni <= warmup:
                trainer.accumulate = max(
                    1, int(np.interp(ni, [0, warmup], [1, trainer.args.nbs / trainer.batch_size]).round())
                )
                for group in trainer.optimizer.param_groups:
                    start = trainer.args.warmup_bias_lr if group.get("param_group") == "bias" else 0.0
                    group["lr"] = float(
                        np.interp(ni, [0, warmup], [start, group["initial_lr"] * trainer.lf(trainer.epoch)])
                    )
                    if "momentum" in group:
                        group["momentum"] = float(
                            np.interp(ni, [0, warmup], [trainer.args.warmup_momentum, trainer.args.momentum])
                        )
            raw = next(loader)
            assert raw["img"].shape[0] == trainer.batch_size == 32
            report["batches_observed"] += 1
            with autocast(trainer.amp, device=trainer.device.type):
                batch = trainer.preprocess_batch(raw)
                loss, items = trainer.model(batch)
                total = loss.sum()
            row.update(
                shape=list(batch["img"].shape),
                loss=total.item(),
                components=items.detach().cpu().tolist(),
                scale_before=trainer.scaler.get_scale(),
                accumulate=trainer.accumulate,
            )
            assert list(batch["img"].shape) == [32, 3, 640, 640]
            assert batch["cls"].numel() and torch.isfinite(total)
            trainer.scaler.scale(total).backward()
            row["attempted"] = ni - last_step >= trainer.accumulate
            if row["attempted"]:
                observed_optimizer_step(trainer, params, report, row)
                last_step = ni  # Native Trainer advances this even when GradScaler skips the step.
            row["scale_after"] = trainer.scaler.get_scale()
            report["stage"] = "model_finiteness"
            assert all(torch.isfinite(v).all() for v in trainer.model.state_dict().values())
            write_json(checks_path, report)
            if not report["missing_effective_parameters"]:
                return
        report["stage"] = "observation_budget"
        raise AssertionError(
            f"Batch budget exhausted; no effective task update for: {report['missing_effective_parameters']}"
        )
    except BaseException as exc:
        record_failure(report, exc)
        raise
    finally:
        write_json(checks_path, report)


def preflight(config, evidence, directory, block_type=Concat_CCA_Fusion, trainer_type=AuditedTrainer):
    """Run bounded real batch32 native AMP/MuSGD updates and the unchanged downstream checks."""
    from ultralytics import YOLO

    directory.mkdir(parents=True, exist_ok=False)
    report = observation_report()
    trainer = None
    try:
        report["stage"] = "setup"
        write_json(directory / "environment.json", evidence)
        trainer = trainer_type(overrides=dict(config, project=str(directory), name="check"))
        trainer.add_callback("on_pretrain_routine_end", final_model_audit)
        trainer._setup_train()
        assert trainer.batch_size == trainer.args.batch == 32
        assert len(trainer.train_loader.dataset) == REFERENCE["dataset_counts"]["train"]
        assert len(trainer.test_loader.dataset) == REFERENCE["dataset_counts"]["val"]
        report["optimizer"] = audit_optimizer(trainer)

        def capture_numerics(module, args, value):
            with torch.no_grad():
                up, low, high = args[0]
                delta, weights, q, k = module.correspondence(low, high)
                latest = dict(
                    q=tensor_stats(q),
                    k=tensor_stats(k),
                    q_norm=tensor_stats(q.float().norm(dim=1)),
                    k_norm=tensor_stats(k.float().norm(dim=1)),
                    weights=tensor_stats(weights),
                    delta=tensor_stats(delta),
                    residual=tensor_stats(value[:, :256] - up),
                    center_weight=weights[:, 4].mean().item(),
                    entropy=(-(weights * weights.clamp_min(1e-30).log()).sum(1)).mean().item(),
                    shape=list(high.shape),
                    valid_counts="corner=4 edge=6 interior=9 (at standard P5 20x20)",
                )
                report["latest_numerics"] = latest
                if not all(
                    latest[name]["finite"] for name in ("q", "k", "q_norm", "k_norm", "weights", "delta", "residual")
                ):
                    torch.save(
                        dict(q=q[:1].cpu(), k=k[:1].cpu(), weights=weights[:1].cpu(), delta=delta[:1].cpu()),
                        directory / "numerical_failure.pt",
                    )
                    raise AssertionError("Nonfinite CCA values; preserved numerical_failure.pt")

        numeric_hook = trainer.model.model[12].register_forward_hook(capture_numerics)
        params = dict(trainer.model.model[12].named_parameters())
        initial = {k: p.detach().clone() for k, p in params.items()}
        loader = iter(trainer.train_loader)
        # Disposable independent model; FP32 check never modifies the AMP training candidate or optimizer.
        report["stage"] = "fp32"
        fp32 = copy.deepcopy(trainer.model).float().train()
        fp32.zero_grad(set_to_none=True)
        first_raw = next(loader)
        batch = trainer.preprocess_batch(copy.deepcopy(first_raw))
        assert list(batch["img"].shape) == [32, 3, 640, 640]
        with autocast(False, device=trainer.device.type):
            loss, _ = fp32(batch)
            loss.sum().backward()
        assert torch.isfinite(loss).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in fp32.parameters())
        report["fp32"] = dict(batch=32, loss=loss.detach().cpu().tolist(), finite=True)
        del fp32, batch, loss
        from itertools import chain

        loader = iter(chain([first_raw], loader))
        del first_raw
        observe_batches(trainer, params, loader, report, directory / "checks.json")
        numeric_hook.remove()
        report["stage"] = "ema"
        assert all(not torch.equal(initial[k], p) for k, p in params.items())
        assert type(trainer.ema.ema.model[12]) is block_type
        assert trainer.ema.updates > 0 and torch.count_nonzero(trainer.ema.ema.model[12].Wo.weight) > 0
        report["ema_updates"] = trainer.ema.updates
        report["stage"] = "reload_fuse"
        report["reload"] = save_reload_check(trainer.ema.ema, directory, block_type)
        # Native standalone Validator owns its model copy/fusion and evaluates the unchanged val split.
        report["stage"] = "validator"
        validator_model = YOLO(directory / "preflight.pt")
        observed = {}
        calls = []
        residuals = []

        def capture_residual(module, args, value):
            calls.append(True)
            residuals.append((value[:, :256] - args[0][0]).float().norm().item())

        validator_model.model.model[12].register_forward_hook(capture_residual)
        validator_model.add_callback("on_val_start", lambda v: (calls.clear(), residuals.clear()))
        validator_model.add_callback(
            "on_val_end",
            lambda v: observed.update(args=vars(v.args), images=v.seen, targets=int(v.metrics.nt_per_class.sum())),
        )
        metrics = validator_model.val(
            data=config["data"],
            split="val",
            imgsz=640,
            batch=32,
            workers=8,
            device=config["device"],
            quantize=None,
            conf=0.001,
            iou=0.7,
            max_det=300,
            rect=True,
            augment=False,
            project=str(directory),
            name="validator",
            exist_ok=False,
        )
        assert calls and max(residuals) > 0
        observed.update(cca_calls=len(calls), residual_max_norm=max(residuals))
        assert observed["images"] == 2404 and observed["targets"] == 2985
        assert observed["args"]["batch"] == 32 and observed["args"]["quantize"] is None
        report["validator"] = dict(observed, metrics=metrics.results_dict, scope="preflight only; not final accuracy")
        report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(trainer.device)
    except BaseException as exc:
        record_failure(report, exc)
        raise
    finally:
        try:
            if trainer is not None:
                for loader in (getattr(trainer, "train_loader", None), getattr(trainer, "test_loader", None)):
                    if loader is not None:
                        loader.close()
        except BaseException as exc:
            if "exception" not in report:
                report["stage"] = "loader_cleanup"
                record_failure(report, exc)
            raise
        finally:
            write_json(directory / "checks.json", report)
    report.update(passed=True, stage="complete")
    write_json(directory / "checks.json", report)
    return report


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
        raise FileNotFoundError(f"Original expanded b19 launcher is required: {path}")
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


def final_model_audit(trainer):
    """Audit the actual post-setup optimizer/model before the first formal batch."""
    if trainer.amp != bool(trainer.args.amp):
        raise RuntimeError("Native AMP check changed b19 AMP; stop instead of silently changing the recipe.")
    changed = [k for k, v in trainer.initial_common.items() if not torch.equal(v, trainer.model.state_dict()[k].cpu())]
    if changed:
        raise AssertionError(f"Common tensors changed before the first formal batch: {changed}")
    cca = trainer.model.model[12]
    assert torch.count_nonzero(cca.Wo.weight) == 0
    assert all(p.requires_grad for p in cca.parameters())
    assert type(trainer.ema.ema.model[12]) is trainer.block_type
    write_json(trainer.save_dir / "provenance/final_optimizer.json", audit_optimizer(trainer))
    write_json(trainer.save_dir / "provenance/final_weight_audit.json", trainer.weight_audit)
    del trainer.initial_common


def record_completion(trainer):
    """Record native best-checkpoint val metrics and completion only after the training loop returns."""
    validator = trainer.validator
    assert torch.count_nonzero(trainer.ema.ema.model[12].Wo.weight) > 0
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
        + [
            Path(__file__),
            Path(__file__).with_name("b19_reference.json"),
            Path(__file__).with_name("b19_dataset_manifest.json"),
            *map(Path, extra),
        ]
    )
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def main(
    argv=None,
    *,
    model=MODEL,
    trainer_type=AuditedTrainer,
    entrypoint=Path(__file__).resolve(),
    name="yolo26n_b19_cca_fusion_v1",
    source_files=(),
    structure_check=structural_checks,
    module_config=MODULE_CONFIG,
):
    """Resolve and run the sole CCA candidate; missing server evidence never becomes a passing receipt."""
    block_type = trainer_type.block_type
    parser = argparse.ArgumentParser(description=f"Audited b19 experiment: {block_type.__name__}")
    parser.add_argument(
        "--baseline-root", type=Path, required=True, help="Original b19 project root; never the CCA worktree"
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
        "--project", type=Path, help="Independent CCA output project; defaults to this worktree/runs/detect"
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
    check_root = project / f"{options.name}_preflight"
    check_root.mkdir(parents=True, exist_ok=True)
    invocation = Path(tempfile.mkdtemp(prefix=f"{options.stage}-invocation-", dir=check_root))
    status = observation_report()
    process = dict(
        pid=os.getpid(),
        started=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, str(entrypoint), *arguments],
        commit=git("rev-parse", "HEAD"),
        state="running",
        exit_code=None,
    )
    write_json(invocation / "process_status.json", process)
    (check_root / f"{options.stage}.current_invocation").write_text(str(invocation))
    handler = logging.FileHandler(invocation / "console.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    console_out, console_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = TeeStream(console_out, handler.stream), TeeStream(console_err, handler.stream)
    try:
        status["stage"] = "source_structure_recipe"
        if options.stage == "train" and (project / options.name).exists():
            raise FileExistsError(f"CCA output already exists; no second run or overwrite: {project / options.name}")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("Commit source changes before server preflight or training")
        structure_check(invocation, model, block_type)
        raw, config, evidence = resolve_recipe(options, model)
        write_json(invocation / "configuration.json", dict(raw=raw, effective=config, evidence=evidence))
        launcher = launcher_evidence(options, raw)
        evidence["launch_evidence"] = launcher
        # Fingerprint the committed package and experiment entry dependencies.
        evidence["source_sha256"] = source_hashes(
            [
                entrypoint,
                Path(__file__).with_name("server_b19_cca_fusion_v1.sh"),
                Path(__file__).with_name("finish_b19_cca_fusion.py"),
                *source_files,
            ]
        )
        signature = hashlib.sha256(json.dumps([config, evidence], sort_keys=True).encode()).hexdigest()
        check_dir = check_root / signature[:20]
        status.update(fingerprint=signature, commit=evidence.get("commit"))
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
        write_json(invocation / "local_weight_audit.json", probe.weight_audit)
        del probe, weights
        issues = runtime_issues(config) if options.stage == "preflight" else []
        if issues:
            write_json(
                invocation / "missing.json",
                dict(passed=False, missing=issues, structural_passed=True, trainer_get_model_weights_passed=True),
            )
            print("Preflight incomplete:\n- " + "\n- ".join(issues), file=sys.stderr)
            status["stage"] = "runtime_requirements"
            record_failure(status, RuntimeError("; ".join(issues)))
            process["exit_code"] = 2
            return 2
        receipt = check_root / f"{signature[:20]}.latest_passed.json"
        if options.stage == "train":
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
            status["stage"] = "preflight_subprocess"
            previous_invocations = set(check_root.glob("preflight-invocation-*"))
            result = subprocess.run([sys.executable, str(entrypoint), *command], cwd=ROOT)
            child_reports = set(check_root.glob("preflight-invocation-*")) - previous_invocations
            assert len(child_reports) == 1, "Expected exactly one fresh preflight invocation"
            child_path = child_reports.pop() / "checks.json"
            status["subprocess_checks"] = str(child_path)
            if child_path.is_file():
                child = json.loads(child_path.read_text(encoding="utf-8"))
                status.update(
                    {k: v for k, v in child.items() if k not in ("passed", "stage", "exception", "failure_stage")}
                )
            child_status = dict(
                python=result.returncode, stage="preflight", command=command, commit=git("rev-parse", "HEAD")
            )
            write_json(check_root / "preflight_process_status.json", child_status)
            result.check_returncode()
        if options.stage == "preflight":
            check_dir = Path(tempfile.mkdtemp(prefix=signature[:20] + "-", dir=check_root)) / "report"
            status.update(stage="real_preflight", report_dir=str(check_dir))
            preflight(config, evidence, check_dir, block_type, trainer_type)
            write_json(check_dir / "passed.json", dict(fingerprint=signature, passed=True))
            write_json(receipt, dict(fingerprint=signature, passed=True, report_dir=str(check_dir)))
        status["stage"] = "receipt_and_fresh_trainer"
        passed = json.loads(receipt.read_text(encoding="utf-8"))
        if passed.get("fingerprint") != signature or not passed.get("passed"):
            raise RuntimeError("Stale/mismatched preflight receipt.")
        status["report_dir"] = passed["report_dir"]
        if options.stage == "preflight":
            print(f"Preflight passed: {receipt}")
            status.update(passed=True, stage="complete", receipt=str(receipt))
            return 0
        issues = runtime_issues(config)
        if issues:
            raise RuntimeError("Resources/environment changed after preflight: " + "; ".join(issues))
        # Start from the b19 seed in a fresh trainer; no disposable batch/model/optimizer/EMA is reused.
        init_seeds(config["seed"], deterministic=config["deterministic"])
        trainer = trainer_type(overrides=config)
        provenance = trainer.save_dir / "provenance"
        shutil.copytree(Path(passed["report_dir"]), provenance)
        shutil.copy2(check_root / "preflight_process_status.json", provenance / "preflight_process_status.json")
        shutil.copytree(Path(status["subprocess_checks"]).parent, provenance / "preflight_invocation")
        shutil.copy2(evidence["args_path"], provenance / "b19_original_args.yaml")
        shutil.copy2(model, provenance / model.name)
        shutil.copy2(launcher["path"], provenance / "b19_launcher_expanded.txt")
        shutil.copy2(evidence["data_path"], provenance / "original_data.yaml")
        YAML.save(provenance / "cca_effective.yaml", config)
        write_json(provenance / "module.json", module_config)
        write_json(provenance / "resolved.json", evidence)
        trainer.add_callback("on_pretrain_routine_end", final_model_audit)
        trainer.add_callback("on_train_end", record_completion)
        train_handler = logging.FileHandler(trainer.save_dir / "train.log", encoding="utf-8")
        LOGGER.addHandler(train_handler)
        try:
            status["stage"] = "formal_train"
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
        print(f"CCA completed; validation-selected weights and provenance: {trainer.save_dir}")
        status.update(passed=True, stage="complete", receipt=str(receipt))
        return 0
    except BaseException as exc:
        record_failure(status, exc)
        text = traceback.format_exc()
        with (invocation / "console.log").open("a", encoding="utf-8") as stream:
            stream.write(text)
        print(text, file=sys.stderr)
        return 1
    finally:
        # Link/copy the detailed observation counters even when failure occurred inside the isolated subprocess.
        report_dir = status.get("report_dir")
        if report_dir and (Path(report_dir) / "checks.json").is_file():
            details = json.loads((Path(report_dir) / "checks.json").read_text(encoding="utf-8"))
            for key in (
                "attempted_steps",
                "overflow_skips",
                "successful_steps",
                "batches_observed",
                "first_finite_nonzero_gradient_step",
                "first_effective_update_step",
                "missing_effective_parameters",
            ):
                status[key] = details[key]
            if not details["passed"] and "missing_reasons" in details:
                status["missing_reasons"] = details["missing_reasons"]
        sys.stdout, sys.stderr = console_out, console_err
        write_json(invocation / "checks.json", status)
        process.update(
            state="finished",
            finished=datetime.now(timezone.utc).isoformat(),
            exit_code=process["exit_code"] if process["exit_code"] is not None else (0 if status["passed"] else 1),
        )
        write_json(invocation / "process_status.json", process)
        LOGGER.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
