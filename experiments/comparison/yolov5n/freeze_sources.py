"""Reproduce the committed private backend and standalone B19 algorithms from verified source trees."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SHARED = HERE.parent / "shared"
B19_SHA = "4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(root, revision, name):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "show", f"{revision}:{name}"]
    )


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one upstream patch target: {old[:90]}")
    return source.replace(old, new)


def freeze(upstream):
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={upstream.as_posix()}", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != "915bbf294bb74c859f0b41f1c23bc395014ea679":
        raise ValueError(f"Expected official v7.0, got {commit}")
    vendor = HERE / "vendor" / "yolov5"
    files = [
        "LICENSE",
        "requirements.txt",
        "train.py",
        "models/__init__.py",
        "models/yolo.py",
        "models/common.py",
        "models/experimental.py",
        "models/yolov5n.yaml",
        "utils/__init__.py",
        "utils/general.py",
        "utils/torch_utils.py",
        "utils/autoanchor.py",
        "utils/callbacks.py",
        "utils/dataloaders.py",
        "utils/downloads.py",
        "utils/augmentations.py",
        "utils/loss.py",
        "utils/metrics.py",
        "utils/plots.py",
        "utils/segment/__init__.py",
        "utils/segment/general.py",
        "data/hyps/hyp.scratch-low.yaml",
        "data/images/bus.jpg",
    ]
    upstream_hashes = {}
    for name in files:
        target = vendor / name
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = git_blob(upstream, commit, name)
        checked = (upstream / name).read_bytes()
        if name.endswith(".jpg"):
            if checked != blob:
                raise ValueError(f"Modified upstream asset: {name}")
        elif checked.replace(b"\r\n", b"\n") != blob.replace(b"\r\n", b"\n"):
            raise ValueError(f"Modified upstream source: {name}")
        target.write_bytes(blob if name.endswith(".jpg") else blob.replace(b"\r\n", b"\n"))
        upstream_hashes[name] = hashlib.sha256(blob).hexdigest()
    patches = []
    train = (vendor / "train.py").read_text(encoding="utf-8")
    train = replace(
        train, "import val as validate  # for end-of-epoch mAP", "from runtime import validation_run, HOOKS"
    )
    train = replace(train, "from utils.dataloaders import create_dataloader", "from data import create_dataloader")
    train = replace(train, "from utils.autobatch import check_train_batch_size\n", "")
    train = replace(
        train,
        "from utils.loggers import Loggers\nfrom utils.loggers.comet.comet_utils import check_comet_resume",
        "from comparison_io import trusted_load",
    )
    train = replace(
        train,
        "GIT_INFO = check_git_info()",
        "GIT_INFO = HOOKS.backend_identity  # verified source lock owns provenance",
    )
    start, end = train.index("    # Loggers\n"), train.index("    # Config\n")
    train = train[:start] + "    data_dict = opt.comparison_data\n\n" + train[end:]
    train = replace(
        train,
        "            weights = attempt_download(weights)  # download if not found locally",
        "            weights = str(Path(weights).resolve())  # outer preflight verifies source/hash",
    )
    train = replace(train, "torch.load(weights, map_location='cpu')", "trusted_load(weights, map_location='cpu')")
    train = replace(
        train,
        "        model.load_state_dict(csd, strict=False)  # load",
        "        missing = model.load_state_dict(csd, strict=False)\n        HOOKS.transfer(model, csd, missing, weights)",
    )
    train = replace(
        train,
        "    amp = check_amp(model)  # check AMP",
        "    amp = check_amp(model, opt.amp_probe) if opt.amp else False\n    HOOKS.amp(opt.amp, amp, device)",
    )
    start, end = train.index("    # Batch size\n"), train.index("    # Optimizer\n")
    train = train[:start] + train[end:]
    start, end = train.index("    # DP mode\n"), train.index("    # SyncBatchNorm\n")
    train = train[:start] + train[end:]
    train = replace(
        train, "last, best = w / 'last.pt', w / 'best.pt'", "last, best = w / 'last.pt', w / 'native_best.pt'"
    )
    train = replace(train, "deepcopy(de_parallel(model)).half()", "deepcopy(de_parallel(model)).float()")
    train = replace(train, "deepcopy(ema.ema).half()", "deepcopy(ema.ema).float()")
    train = replace(
        train,
        "    compute_loss = ComputeLoss(model)  # init loss class",
        "    compute_loss = ComputeLoss(model)  # init loss class\n    HOOKS.established(model, optimizer, hyp, accumulate, nbs, amp, dataset, ema, scheduler, scaler, stopper, nb)",
    )
    train = replace(
        train, "        model.train()\n", "        model.train()\n        HOOKS.epoch(epoch, dataset, train_loader)\n"
    )
    train = train.replace("validate.run(", "validation_run(")
    train = replace(
        train,
        "            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)",
        "            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)\n            HOOKS.fit(log_vals, epoch, stop, optimizer, accumulate)",
    )
    train = replace(
        train,
        "                torch.save(ckpt, last)",
        "                HOOKS.checkpoint(ckpt, epoch, model, ema, optimizer, scheduler, scaler, stopper)\n                torch.save(ckpt, last)\n                HOOKS.checksum(last)",
    )
    train = replace(
        train,
        "                    torch.save(ckpt, best)",
        "                    torch.save(ckpt, best)\n                    HOOKS.checksum(best)",
    )
    start, end = train.index("        for f in last, best:\n"), train.index("        callbacks.run('on_train_end'")
    train = train[:start] + train[end:]
    train = train[: train.index("\ndef main(opt,")]  # outer entry owns preparation, resume, status, and concurrency
    (vendor / "train.py").write_text(train.rstrip() + "\n", encoding="utf-8", newline="\n")
    patches.append(
        "train.py: outer entry owns datasets, logging, trusted loading/resume, AMP request, B19 schedule, audit and unified best; remove auto-batch, implicit multi-GPU, remote loggers, evolution/main, final FP16 revalidation and optimizer stripping; native loss/optimizer/warmup/EMA/anchor mechanism retained"
    )
    general = (vendor / "utils/general.py").read_text(encoding="utf-8")
    general = replace(general, "def check_amp(model):", "def check_amp(model, probe_weights):")
    general = replace(general, "DetectMultiBackend('yolov5n.pt', device)", "DetectMultiBackend(probe_weights, device)")
    general = replace(general, "import IPython\n", "")
    general = replace(
        general,
        "    ipython_type = str(type(IPython.get_ipython()))",
        "    ipython = sys.modules.get('IPython')\n    ipython_type = str(type(ipython.get_ipython())) if ipython else ''",
    )
    general = replace(
        general,
        "        if (time.time() - t) > time_limit:\n            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')\n            break  # time limit exceeded",
        "        # Every image is processed; elapsed time must never truncate an AP export.",
    )
    (vendor / "utils/general.py").write_text(general, encoding="utf-8", newline="\n")
    patches.append(
        "utils/general.py: AMP fallback uses verified cached same-family official checkpoint; remove NMS wall-clock truncation so every image is exported"
    )
    patches.append(
        "utils/general.py: remove eager notebook-only IPython dependency; inspect an already active notebook instead"
    )
    torch_utils = (vendor / "utils/torch_utils.py").read_text(encoding="utf-8")
    torch_utils = replace(
        torch_utils,
        "def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):",
        "def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5, beta2=0.999):",
    )
    torch_utils = torch_utils.replace("betas=(momentum, 0.999)", "betas=(momentum, beta2)")
    (vendor / "utils/torch_utils.py").write_text(torch_utils, encoding="utf-8", newline="\n")
    train = train.replace("hyp['weight_decay'])", "hyp['weight_decay'], beta2=opt.beta2)")
    train = train.replace("torch.save(ckpt, last)\n                HOOKS.checksum(last)", "HOOKS.save(ckpt, last)")
    train = train.replace("torch.save(ckpt, best)\n                    HOOKS.checksum(best)", "HOOKS.save(ckpt, best)")
    train = train.replace("torch.save(ckpt, w / f'epoch{epoch}.pt')", "HOOKS.save(ckpt, w / f'epoch{epoch}.pt')")
    (vendor / "train.py").write_text(train.rstrip() + "\n", encoding="utf-8", newline="\n")
    patches.append("utils/torch_utils.py: expose Adam beta2, preserving its original 0.999 default")
    dataloaders = (vendor / "utils/dataloaders.py").read_text(encoding="utf-8")
    dataloaders = replace(dataloaders, "(Albumentations, augment_hsv", "(augment_hsv")
    dataloaders = replace(
        dataloaders,
        "                      shuffle=False):",
        "                      shuffle=False,\n                      albumentations=None):",
    )
    dataloaders = replace(
        dataloaders,
        "            prefix=prefix)",
        "            prefix=prefix,\n            albumentations=albumentations)",
    )
    dataloaders = replace(
        dataloaders,
        "                 prefix=''):",
        "                 prefix='',\n                 albumentations=None):",
    )
    dataloaders = replace(
        dataloaders,
        "        self.albumentations = Albumentations(size=img_size) if augment else None",
        "        self.albumentations = albumentations",
    )
    dataloaders = replace(
        dataloaders,
        "            img, labels = self.albumentations(img, labels)\n            nl = len(labels)  # update after albumentations",
        "            if self.albumentations is not None:\n                img = self.albumentations({'img': img, 'cls': labels[:, :1]})['img']",
    )
    dataloaders = replace(
        dataloaders,
        "        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')",
        "        cache_root = Path(os.environ['YOLOV5_LABEL_CACHE'])\n        cache_root.mkdir(parents=True, exist_ok=True)\n        cache_path = cache_root / (get_hash(self.label_files + self.im_files) + '.cache')",
    )
    dataloaders = replace(
        dataloaders,
        "        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]",
        "        self.npy_files = [cache_root / ('image_' + get_hash([f]) + '.npy') for f in self.im_files]",
    )
    # JPEG repair belongs to dataset preparation, never to a comparison reader.
    dataloaders = replace(
        dataloaders,
        "                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)\n                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'",
        "                    raise ValueError('Corrupt JPEG: comparison datasets are read-only; repair separately')",
    )
    (vendor / "utils/dataloaders.py").write_text(dataloaders, encoding="utf-8", newline="\n")
    patches.append(
        "utils/dataloaders.py: relocate native variant label caches to experiment cache; remove write-back JPEG repair and ambient optional augmentation construction; adapter injects the explicit disabled/frozen policy before workers start"
    )
    augmentations = (vendor / "utils/augmentations.py").read_text(encoding="utf-8")
    start, end = augmentations.index("\nclass Albumentations:"), augmentations.index("\ndef normalize(")
    (vendor / "utils/augmentations.py").write_text(
        augmentations[:start] + augmentations[end:], encoding="utf-8", newline="\n"
    )
    patches.append(
        "utils/augmentations.py: delete unused ambient detection Albumentations wrapper; reuse the explicit frozen B19 optional provider"
    )
    for name in ("models/common.py", "models/experimental.py"):
        text = (vendor / name).read_text(encoding="utf-8")
        text = text.replace("torch.load(", "trusted_load(")
        text = text.replace(
            "from pathlib import Path", "from pathlib import Path\nfrom comparison_io import trusted_load"
        )
        if name == "models/common.py":
            text = replace(text, "from IPython.display import display\n", "")
            text = replace(
                text,
                "                display(im) if is_notebook() else im.show(self.files[i])",
                "                if is_notebook():\n                    from IPython.display import display\n                    display(im)\n                else:\n                    im.show(self.files[i])",
            )
        (vendor / name).write_text(text, encoding="utf-8", newline="\n")
        patches.append(f"{name}: only private checkpoint load sites use manifest-verified explicit weights_only=False")
    patches.append(
        "Private train checkpoint serialization retains FP32 model/EMA; lazy notebook display imports; source provenance comes from verified lock."
    )
    lock = {
        "repository": "https://github.com/ultralytics/yolov5",
        "tag": "v7.0",
        "commit": commit,
        "model": "original YOLOv5n P5 anchor detection",
        "license": "GPL-3.0 (v7.0 upstream LICENSE)",
        "upstream_files": upstream_hashes,
        "vendor_files": {n: sha(vendor / n) for n in files},
        "patches": patches,
    }
    (HERE / "backend.lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    import yaml

    source = (upstream / "train.py").read_text(encoding="utf-8")
    parse_opt = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "parse_opt")
    defaults = {}
    for node in ast.walk(parse_opt):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            key = ast.literal_eval(node.args[0]).lstrip("-").replace("-", "_")
            options = {k.arg: k.value for k in node.keywords}
            value = (
                eval(
                    compile(ast.Expression(options["default"]), "<locked parser>", "eval"),
                    {"ROOT": upstream, "__builtins__": {}},
                )
                if "default" in options
                else False
                if "action" in options and ast.literal_eval(options["action"]) == "store_true"
                else None
            )
            defaults[key] = "${vendor}/" + value.relative_to(upstream).as_posix() if isinstance(value, Path) else value
    snapshot = {
        "commit": commit,
        "source": "train.py::parse_opt + data/hyps/hyp.scratch-low.yaml + utils/torch_utils.py::smart_optimizer",
        "parser": defaults,
        "hyp": yaml.safe_load((upstream / "data/hyps/hyp.scratch-low.yaml").read_text(encoding="utf-8")),
        "optimizer_groups": ["bias (no decay)", "weight (decay)", "normalization (no decay)"],
        "adam_beta2": 0.999,
        "nominal_batch": 64,
        "schedule": "linear unless cos_lr explicitly true",
        "warmup_min_iterations": 100,
        "scaling": {
            "weight_decay": "batch*accumulate/64",
            "box": "3/nl",
            "cls": "nc/80*3/nl",
            "obj": "(imgsz/640)^2*3/nl",
        },
    }
    (HERE / "official_defaults.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    # Reuse the tested peer provider verbatim; do not generate a second augmentation core.
    shared_ref = "482e555248c0ba52b4bea2b474498cab61fd6f37"
    shared_root = HERE.parent / "shared"
    names = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "-C",
            str(ROOT),
            "ls-tree",
            "-r",
            "--name-only",
            shared_ref,
            "experiments/comparison/shared",
        ],
        text=True,
    ).splitlines()
    hashes = {}
    for name in names:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git_blob(ROOT, shared_ref, name))
        hashes[target.relative_to(shared_root).as_posix()] = sha(target)
    (HERE / "shared.lock.json").write_text(
        json.dumps(
            {
                "repository": "https://github.com/supershaojie/Tunnel_Disease_YOLO26",
                "reference_commit": shared_ref,
                "reference_branch": "codex/compare-yolov8n-b19",
                "files": hashes,
                "reuse": "Verbatim committed shared provider, data audit, matching, AP and prediction exchange. No source dependency on another worktree.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"backend_sha": commit, "b19_sha": B19_SHA, "vendor_files": len(files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Official checkout at the locked v7.0 commit")
    freeze(parser.parse_args().upstream.resolve())
