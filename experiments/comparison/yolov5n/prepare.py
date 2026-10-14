"""Verify locked source and atomically prepare/reuse official v7.0 YOLOv5n COCO weights."""

import copy
import os
from pathlib import Path

from common import HERE, UniqueStore, configuration, parser, verify_sources
from comparison_io import checksum, digest, read_json, verify_weights

URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
API = "https://api.github.com/repos/ultralytics/yolov5/releases/tags/v7.0"


def prepare(path, proxy=None):
    import requests

    path = Path(path).resolve()
    lock_path = HERE / "weights.lock.json"
    expected = read_json(lock_path) if lock_path.exists() else None
    if path.exists():
        if expected and digest(path) == expected["sha256"] and path.stat().st_size == expected["bytes"]:
            if not Path(str(path) + ".source.json").is_file():
                checksum(
                    path,
                    {
                        **expected,
                        "model": "yolov5n",
                        "kind": "official_pretrained",
                        "verification": "matches locally measured official-release lock; hash is not claimed as officially published",
                    },
                )
            return verify_weights(path)
        raise ValueError(f"Existing same-name weight does not match official lock: {path}; it will not be overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = path.with_name(path.name + ".prepare.lock")
    descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        session = requests.Session()
        session.headers["User-Agent"] = "comparison-b19-v1/yolov5n"
        if proxy:
            session.proxies.update(http=proxy, https=proxy)
        response = session.get(API, timeout=60)
        response.raise_for_status()
        release = response.json()
        assets = [a for a in release["assets"] if a["name"] == "yolov5n.pt" and a["browser_download_url"] == URL]
        if release["tag_name"] != "v7.0" or len(assets) != 1:
            raise ValueError("Exact official original yolov5n.pt asset not found in v7.0 release")
        asset = assets[0]
        download = session.get(URL, stream=True, timeout=(30, 60))
        download.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in download.iter_content(1024 * 1024):
                stream.write(chunk)
        size, actual = temporary.stat().st_size, digest(temporary)
        if size != asset["size"]:
            raise ValueError(f"Official asset size mismatch: expected {asset['size']}, downloaded {size}")
        if expected and (actual != expected["sha256"] or size != expected["bytes"]):
            raise ValueError("Downloaded official asset differs from committed measured weight lock")
        published = asset.get("digest")
        if published and published != "sha256:" + actual:
            raise ValueError("GitHub published asset digest mismatch")
        provenance = {
            "model": "yolov5n",
            "kind": "official_pretrained",
            "source_url": URL,
            "release_api": API,
            "release_page": "https://github.com/ultralytics/yolov5/releases/tag/v7.0",
            "tag": "v7.0",
            "asset_id": asset["id"],
            "asset_name": asset["name"],
            "bytes": size,
            "sha256": actual,
            "published_digest": published,
            "checksum_origin": "computed locally from downloaded official bytes; not an official checksum claim",
        }
        os.replace(temporary, path)
        checksum(path, provenance)
        print(f"Downloaded and verified {size} bytes, SHA256 {actual}")
        return provenance
    finally:
        temporary.unlink(missing_ok=True)
        guard.unlink(missing_ok=True)


def main():
    p = parser(__doc__)
    p.add_argument("--cache-dir", action=UniqueStore, type=Path, help="Override only the pretrained weight directory")
    p.add_argument("--proxy", action=UniqueStore, help="Optional explicit HTTP(S) proxy for this preparation request")
    args = p.parse_args()
    source = verify_sources()
    planning = copy.copy(args)
    planning.dry_run = True  # preparation intentionally does not require server training data/args
    cfg, _ = configuration(planning)
    path = args.cache_dir / "yolov5n.pt" if args.cache_dir else Path(cfg["weights"])
    if args.dry_run:
        print(
            {
                "state": "DRY_RUN",
                "backend_commit": source["backend"]["commit"],
                "weights": str(path),
                "URL": URL,
                "downloaded": False,
            }
        )
        return
    value = prepare(path, args.proxy)
    print(value)


if __name__ == "__main__":
    main()
