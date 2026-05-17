"""Run the MNIST M2 + CW reproduction pipeline end to end."""

from __future__ import print_function

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import List


SCRIPTS_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "_project_root.py").is_file()
)
sys.path.insert(0, str(SCRIPTS_ROOT))
from _project_root import configure_project_paths

PROJECT_ROOT = configure_project_paths(__file__)
M2_CW_DIR = PROJECT_ROOT / "results" / "mnist" / "m2_cw"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-cw-l2", action="store_true")
    parser.add_argument("--skip-cw-linf", action="store_true")
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--kappas", default="0.0,0.5,1.0,2.0,4.0")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=9000)
    parser.add_argument(
        "--conda-env",
        default="",
        help="Optional conda environment name. When set, commands run through `conda run -n`.",
    )
    parser.add_argument(
        "--log-file",
        default=str(M2_CW_DIR / "run.log"),
        help="Path where the orchestrator writes a timestamped execution log.",
    )
    return parser


def _python_command(conda_env: str) -> List[str]:
    """Return the Python command prefix."""
    if conda_env:
        if os.environ.get("CONDA_DEFAULT_ENV") == conda_env:
            return [sys.executable]
        conda_executable = shutil.which("conda")
        if conda_executable is None:
            raise RuntimeError(
                "Could not find `conda` on PATH. Activate `{0}` first or run "
                "from an Anaconda/Miniconda prompt.".format(conda_env)
            )
        return [conda_executable, "run", "-n", conda_env, "python"]
    return [sys.executable]


def _timestamp() -> str:
    """Return a human-readable local timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_log_line(handle: object, message: str) -> None:
    """Write one line to terminal and log file."""
    line = "[{0}] {1}".format(_timestamp(), message)
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def run_step(command: List[str], log_handle: object, step_name: str) -> None:
    """Run one subprocess and stream output to terminal and log file."""
    started = time.time()
    _write_log_line(log_handle, "START {0}".format(step_name))
    _write_log_line(log_handle, "running: {0}".format(" ".join(command)))
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    if process.stdout is not None:
        for raw_line in process.stdout:
            message = raw_line.rstrip()
            print(message, flush=True)
            log_handle.write(message + "\n")
            log_handle.flush()

    return_code = process.wait()
    elapsed = time.time() - started
    if return_code != 0:
        _write_log_line(
            log_handle,
            "FAILED {0} exit_code={1} elapsed_seconds={2:.1f}".format(
                step_name,
                return_code,
                elapsed,
            ),
        )
        raise subprocess.CalledProcessError(return_code, command)
    _write_log_line(log_handle, "DONE {0} elapsed_seconds={1:.1f}".format(step_name, elapsed))


def main() -> int:
    """Run the requested pipeline stages."""
    args = build_parser().parse_args()
    py = _python_command(args.conda_env)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_handle:
        _write_log_line(log_handle, "MNIST M2 + CW pipeline started")
        _write_log_line(log_handle, "log_file={0}".format(log_path))
        _write_log_line(
            log_handle,
            "kappas={0} samples={1} start_index={2}".format(
                args.kappas,
                args.samples,
                args.start_index,
            ),
        )

        if not args.skip_training:
            run_step(
                py + ["scripts/mnist_m2_cw/train_mnist_m2.py", "--load-model"],
                log_handle,
                "train_or_restore_m2",
            )

        if not args.skip_cw_l2:
            run_step(
                py
                + [
                    "scripts/mnist_m2_cw/generate_mnist_cw_l2.py",
                    "--load-model",
                    "--kappas",
                    args.kappas,
                    "--samples",
                    str(args.samples),
                    "--start-index",
                    str(args.start_index),
                ],
                log_handle,
                "generate_cw_l2",
            )

        if not args.skip_cw_linf:
            run_step(
                py
                + [
                    "scripts/mnist_m2_cw/generate_mnist_cw_linf.py",
                    "--load-model",
                    "--samples",
                    str(args.samples),
                    "--start-index",
                    str(args.start_index),
                ],
                log_handle,
                "generate_cw_linf",
            )

        if not args.skip_detector:
            run_step(
                py
                + [
                    "scripts/mnist_m2_cw/evaluate_mnist_m2_cw_detector.py",
                    "--attack",
                    "all",
                    "--kappas",
                    args.kappas,
                    "--samples",
                    str(args.samples),
                    "--start-index",
                    str(args.start_index),
                ],
                log_handle,
                "evaluate_detector",
            )

        if not args.skip_comparison:
            run_step(
                py + ["analysis/generate_mnist_m2_cw_article_comparison.py"],
                log_handle,
                "generate_article_comparison",
            )

        _write_log_line(log_handle, "MNIST M2 + CW pipeline finished")

    print("")
    print("MNIST M2 + CW pipeline artifacts:")
    print("- run_log: {0}".format(log_path))
    print("- checkpoint: {0}".format(M2_CW_DIR / "clean_baseline" / "checkpoints"))
    print("- cw_l2: {0}".format(M2_CW_DIR / "cw_l2"))
    print("- cw_linf: {0}".format(M2_CW_DIR / "cw_linf"))
    print("- detector: {0}".format(M2_CW_DIR / "detector"))
    print("- article_comparison: {0}".format(M2_CW_DIR / "article_comparison"))
    print("- limitation: CW Linf is recorded as not_executed unless a compatible implementation is added.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
