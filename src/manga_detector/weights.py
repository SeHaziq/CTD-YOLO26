"""Explicit, checksum-verified weight acquisition for GitHub Release assets."""
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


def _metadata():
    return json.loads(Path(__file__).with_name("model.json").read_text())


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_weights_dir(directory=None):
    if directory is not None:
        return Path(directory).expanduser().resolve()
    if os.environ.get("MANGA_DETECTOR_WEIGHTS"):
        return Path(os.environ["MANGA_DETECTOR_WEIGHTS"]).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2] / "weights"
    if (checkout / "ctd.pt").exists():
        return checkout
    from platformdirs import user_cache_path
    return user_cache_path("manga-detector") / "weights"


def verify_weights(directory, *, mode="hybrid", check_hash=True):
    required = ["ctd.pt"] + (["yolo26n-manga.pt"] if mode == "hybrid" else [])
    metadata = _metadata()["files"]
    for filename in required:
        path = Path(directory) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Supply weights_dir or use download_weights(release_base_url=...).")
        if check_hash and _sha(path) != metadata[filename]["sha256"]:
            raise RuntimeError(f"Weight checksum mismatch: {path}")


def download_weights(weights_dir=None, *, release_base_url=None):
    """Download official CTD plus the selected fine-tune from your release URL.

    release_base_url: e.g. https://github.com/OWNER/REPO/releases/download/v1.0.0
    Inference itself never invokes this function or downloads files.
    """
    directory = resolve_weights_dir(weights_dir)
    files = _metadata()["files"]
    missing = [name for name, meta in files.items() if not (directory / name).is_file() or _sha(directory / name) != meta["sha256"]]
    if "yolo26n-manga.pt" in missing and not release_base_url:
        raise ValueError("Provide release_base_url for the selected YOLO26 fine-tune, or copy the local weights manually.")
    directory.mkdir(parents=True, exist_ok=True)
    for name in missing:
        url = files[name]["url"] or release_base_url.rstrip("/") + "/" + name
        if not url.startswith("https://"):
            raise ValueError("Weight downloads require HTTPS")
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            urllib.request.urlretrieve(url, temporary)
            if _sha(temporary) != files[name]["sha256"]:
                raise RuntimeError(f"Downloaded checksum mismatch for {name}")
            temporary.replace(directory / name)
        finally:
            temporary.unlink(missing_ok=True)
    verify_weights(directory)
    return directory


def main():
    parser = argparse.ArgumentParser(description="Download the frozen manga detector weights")
    parser.add_argument("--release-base-url", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(download_weights(args.output, release_base_url=args.release_base_url))
