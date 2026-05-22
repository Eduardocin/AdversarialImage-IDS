"""Download Caffe ImageNet model assets used by the reproduction."""

from __future__ import print_function

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from urllib.request import Request, urlopen
except ImportError:
    from urllib import urlopen
    from urllib2 import Request


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "models" / "imagenet"
MODEL_CHOICES = ("alexnet", "googlenet")


@dataclass(frozen=True)
class AssetSpec:
    """Describe one local asset and its download metadata."""

    filename: str
    urls: Tuple[str, ...]
    sha1: Optional[str] = None
    manual_urls: Tuple[str, ...] = ()
    note: Optional[str] = None


@dataclass(frozen=True)
class ModelSpec:
    """Describe one Caffe model asset directory."""

    key: str
    display_name: str
    output_subdir: str
    assets: Tuple[AssetSpec, ...]
    source_note: str


COMMON_CAFFE_MEAN = AssetSpec(
    filename="ilsvrc_2012_mean.npy",
    urls=(
        "https://raw.githubusercontent.com/BVLC/caffe/master/python/caffe/imagenet/ilsvrc_2012_mean.npy",
    ),
)

MODEL_SPECS: Dict[str, ModelSpec] = {
    "alexnet": ModelSpec(
        key="alexnet",
        display_name="BVLC AlexNet",
        output_subdir="alexnet",
        assets=(
            AssetSpec(
                filename="deploy.prototxt",
                urls=(
                    "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt",
                ),
            ),
            AssetSpec(
                filename="bvlc_alexnet.caffemodel",
                urls=(
                    "https://nvidia.box.com/shared/static/5j264j7mky11q8emy4q14w3r8hl5v6zh.caffemodel",
                    "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel",
                    "https://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel",
                ),
                sha1="9116a64c0fbe4459d18f4bb6b56d647b63920377",
                note="NVIDIA Box mirror is tried first because the BVLC host is often unavailable.",
            ),
            COMMON_CAFFE_MEAN,
        ),
        source_note="BVLC bundled model metadata plus a verified NVIDIA Box mirror.",
    ),
    "googlenet": ModelSpec(
        key="googlenet",
        display_name="BVLC GoogLeNet",
        output_subdir="googlenet",
        assets=(
            AssetSpec(
                filename="deploy.prototxt",
                urls=(
                    "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt",
                ),
            ),
            AssetSpec(
                filename="bvlc_googlenet.caffemodel",
                urls=(
                    "https://www.deepdetect.com/downloads/platform/pretrained/caffe/googlenet/bvlc_googlenet.caffemodel",
                    "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel",
                    "https://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel",
                ),
                sha1="405fc5acd08a3bb12de8ee5e23a96bec22f08204",
                note="DeepDetect mirror is tried first because the BVLC host is often unavailable.",
            ),
            COMMON_CAFFE_MEAN,
        ),
        source_note="BVLC bundled model metadata plus a verified mirror for weights.",
    ),
}


def build_parser(default_models: Sequence[str]) -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=MODEL_CHOICES + ("all",),
        help=(
            "Model to download. Can be passed more than once. "
            "Default: {0}.".format(",".join(default_models))
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Compatibility option for a single selected model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files again even when they already exist.",
    )
    parser.add_argument(
        "--asset-url",
        action="append",
        default=[],
        metavar="MODEL:FILENAME=URL",
        help=(
            "Prepend a direct mirror for one asset, for example "
            "alexnet:bvlc_alexnet.caffemodel=https://mirror/file.caffemodel."
        ),
    )
    parser.add_argument(
        "--caffemodel-url",
        default=None,
        help="Compatibility alias for GoogLeNet bvlc_googlenet.caffemodel.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print known model keys and exit.",
    )
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
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        with path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def parse_asset_url(value: str) -> Tuple[str, str, str]:
    """Parse MODEL:FILENAME=URL overrides."""
    try:
        model_and_filename, url = value.split("=", 1)
        model, filename = model_and_filename.split(":", 1)
    except ValueError:
        raise ValueError(
            "Expected --asset-url in the form MODEL:FILENAME=URL, got {0}".format(
                value
            )
        )
    if model not in MODEL_SPECS:
        raise ValueError("Unknown model in --asset-url: {0}".format(model))
    if not filename or not url:
        raise ValueError("MODEL, FILENAME, and URL must be non-empty.")
    return model, filename, url


def selected_models(args: argparse.Namespace, default_models: Sequence[str]) -> List[str]:
    """Resolve selected model keys."""
    raw_models = args.model or list(default_models)
    if "all" in raw_models:
        return list(MODEL_CHOICES)
    ordered = []
    for model in raw_models:
        if model not in ordered:
            ordered.append(model)
    return ordered


def asset_urls(
    model_key: str,
    asset: AssetSpec,
    overrides: Dict[Tuple[str, str], List[str]],
) -> Tuple[str, ...]:
    """Return URLs for an asset with mirrors prepended."""
    return tuple(overrides.get((model_key, asset.filename), [])) + asset.urls


def download_asset(
    model_key: str,
    asset: AssetSpec,
    output_dir: Path,
    overrides: Dict[Tuple[str, str], List[str]],
    force: bool = False,
    timeout: int = 120,
) -> bool:
    """Download one asset and verify it when a SHA1 is known."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / asset.filename
    urls = asset_urls(model_key, asset, overrides)

    if path.is_file() and not force:
        print("exists={0}".format(path))
    elif not urls:
        print("manual_required={0}".format(path))
        for manual_url in asset.manual_urls:
            print("manual_url={0}".format(manual_url))
        if asset.note:
            print("note={0}".format(asset.note))
        return False
    else:
        last_error = None
        for url in urls:
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

    expected_sha1 = asset.sha1
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
    return True


def collect_overrides(args: argparse.Namespace) -> Dict[Tuple[str, str], List[str]]:
    """Collect URL overrides from compatibility and generic flags."""
    overrides: Dict[Tuple[str, str], List[str]] = {}
    if args.caffemodel_url:
        overrides[("googlenet", "bvlc_googlenet.caffemodel")] = [
            args.caffemodel_url
        ]

    for value in args.asset_url:
        model, filename, url = parse_asset_url(value)
        overrides.setdefault((model, filename), []).append(url)
    return overrides


def print_known_models() -> None:
    """Print known model keys and output subdirectories."""
    for key in MODEL_CHOICES:
        spec = MODEL_SPECS[key]
        print("{0}\t{1}\t{2}".format(key, spec.display_name, spec.output_subdir))


def output_dir_for_model(args: argparse.Namespace, model_key: str, total: int) -> Path:
    """Return the output directory for a selected model."""
    if args.output_dir:
        if total != 1:
            raise ValueError("--output-dir can only be used with one selected model.")
        return Path(args.output_dir)
    return Path(args.output_root) / MODEL_SPECS[model_key].output_subdir


def main(default_models: Sequence[str] = ("googlenet",)) -> int:
    """Download selected Caffe ImageNet assets."""
    args = build_parser(default_models).parse_args()
    if args.list_models:
        print_known_models()
        return 0

    models = selected_models(args, default_models)
    overrides = collect_overrides(args)
    missing_manual_assets = []

    for model_key in models:
        model = MODEL_SPECS[model_key]
        output_dir = output_dir_for_model(args, model_key, total=len(models))
        print("model={0}".format(model.display_name))
        for asset in model.assets:
            downloaded = download_asset(
                model_key,
                asset,
                output_dir,
                overrides,
                force=bool(args.force),
                timeout=int(args.timeout),
            )
            if not downloaded:
                missing_manual_assets.append((model_key, asset.filename))
        print("{0}_assets_dir={1}".format(model_key, output_dir))

    if missing_manual_assets:
        print("missing_manual_assets={0}".format(missing_manual_assets))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
