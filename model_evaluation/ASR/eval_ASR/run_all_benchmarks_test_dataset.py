"""Run all Whisper benchmarks for the `test_dataset_v1` dataset."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

DATASET_DIR = Path("../generation_dataset/test_dataset_v1")
MODELS = ["small", "medium", "large-v3-turbo"]
OUTPUT_DIR = Path("whisper_benchmark_results_test_dataset")


def run_command(command: list[str]) -> None:
    """Execute a benchmark command and stop on failure.

    Args:
        command (list[str]): Command line to execute.

    Returns:
        None.

    Raises:
        subprocess.CalledProcessError: If the benchmark process fails.
    """
    print("=" * 60)
    print("RUN:", " ".join(command))
    print("=" * 60)
    subprocess.run(command, check=True)


def main() -> None:
    """Run stock and prompted Whisper benchmarks for each configured model.

    Args:
        None.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the dataset directory is missing.
        subprocess.CalledProcessError: If any benchmark process fails.
    """
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    for model in MODELS:
        run_command([
            "python",
            "benchmark_test_dataset.py",
            "--dataset_dir",
            str(DATASET_DIR),
            "--model",
            model,
            "--out_dir",
            str(OUTPUT_DIR),
        ])
        run_command([
            "python",
            "benchmark_test_dataset.py",
            "--dataset_dir",
            str(DATASET_DIR),
            "--model",
            model,
            "--prompt",
            "--out_dir",
            str(OUTPUT_DIR),
        ])
    print(f"Completed in {((time.time() - started_at) / 60):.1f} minutes")
    print(f"Results directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
