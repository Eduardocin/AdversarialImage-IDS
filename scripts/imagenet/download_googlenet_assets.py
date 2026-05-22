"""Download BVLC GoogLeNet Caffe assets used by the ImageNet track."""

from __future__ import print_function

import argparse
import hashlib
import shutil
from pathlib import Path

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "models" / "imagenet" / "googlenet"

ASSETS = (
    {
        "filename": "deploy.prototxt",
        "urls": [
            "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt",
        ],
        "sha1": None,
    },
    {
        "filename": "bvlc_googlenet.caffemodel",
        "urls": [
            "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel",
            "https://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel",
        ],
        "sha1": "405fc5acd08a3bb12de8ee5e23a96bec22f08204",
    },
    {
        "filename": "ilsvrc_2012_mean.npy",
        "urls": [
            "https://raw.githubusercontent.com/BVLC/caffe/master/python/caffe/imagenet/ilsvrc_2012_mean.npy",
        ],
        "sha1": None,
    },
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files again even when they already exist.",
    )
    parser.add_argument(
        "--caffemodel-url",
        default=None,
        help="Optional mirror URL for bvlc_googlenet.caffemodel.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def sha1sum(path: Path) -> str:
    """Return the SHA1 digest for a local file."""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, path: Path, timeout: int) -> None:
    """Download a URL to a local path."""
    with urlopen(url, timeout=timeout) as response:
        with path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def download_asset(asset: dict, output_dir: Path, force: bool = False, timeout: int = 120) -> Path:
    """Download one asset and verify it when a SHA1 is known."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / str(asset["filename"])
    if path.is_file() and not force:
        print("exists={0}".format(path))
    else:
        last_error = None
        for url in asset["urls"]:
            try:
                print("download={0}".format(url))
                download_url(str(url), path, timeout=timeout)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if path.exists():
                    path.unlink()
                print("download_failed={0}: {1}".format(url, exc))
        if last_error is not None:
            raise last_error

    expected_sha1 = asset.get("sha1")
    if expected_sha1:
        actual_sha1 = sha1sum(path)
        if actual_sha1 != expected_sha1:
            raise IOError(
                "SHA1 mismatch for {0}: expected {1}, got {2}".format(
                    path,
                    expected_sha1,
                    actual_sha1,
                )
            )
        print("sha1_ok={0}".format(path))
    return path


def main() -> int:
    """Download all configured GoogLeNet assets."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    assets = [dict(asset) for asset in ASSETS]
    if args.caffemodel_url:
        for asset in assets:
            if asset["filename"] == "bvlc_googlenet.caffemodel":
                asset["urls"] = [args.caffemodel_url] + list(asset["urls"])
    for asset in assets:
        download_asset(asset, output_dir, force=bool(args.force), timeout=int(args.timeout))
    print("googlenet_assets_dir={0}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
