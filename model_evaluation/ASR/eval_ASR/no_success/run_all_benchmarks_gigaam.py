"""Run GigaAM benchmarks for all configured datasets."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

RUNS = [
    {
        "name": "eval_dataset_gtts",
        "command": [
            "python",
            "benchmark_gigaam.py",
            "--dataset",
            "../generation_dataset/eval_dataset_gtts/references.json",
            "--out_dir",
            "gigaam_benchmark_results_all/eval_dataset_gtts",
        ],
    },
    {
        "name": "eval_dataset_meetings_gtts",
        "command": [
            "python",
            "benchmark_gigaam.py",
            "--dataset",
            "../generation_dataset/eval_dataset_meetings_gtts/references.json",
            "--out_dir",
            "gigaam_benchmark_results_all/eval_dataset_meetings_gtts",
        ],
    },
    {
        "name": "test_dataset_v1",
        "command": [
            "python",
            "benchmark_gigaam_test_dataset.py",
            "--dataset_dir",
            "../generation_dataset/test_dataset_v1",
            "--out_dir",
            "gigaam_benchmark_results_all/test_dataset_v1",
        ],
    },
]


def run_command(command: list[str]) -> None:
    """Execute a benchmark command and stop on failure.

    Args:
        command (list[str]): Command line to execute.

    Returns:
        None.

    Raises:
        subprocess.CalledProcessError: If the subprocess fails.
    """
    print("=" * 60)
    print("RUN:", " ".join(command))
    print("=" * 60)
    subprocess.run(command, check=True)


def main() -> None:
    """Run all configured GigaAM benchmark commands.

    Args:
        None.

    Returns:
        None.

    Raises:
        FileNotFoundError: If any required dataset path is missing.
        subprocess.CalledProcessError: If a benchmark subprocess fails.
    """
    started_at = time.time()
    for run in RUNS:
        dataset_flag = "--dataset" if "--dataset" in run["command"] else "--dataset_dir"
        dataset_path = Path(run["command"][run["command"].index(dataset_flag) + 1])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset for {run['name']} not found: {dataset_path.resolve()}")
        output_path = Path(run["command"][run["command"].index("--out_dir") + 1])
        output_path.mkdir(parents=True, exist_ok=True)
        run_command(run["command"])
    print(f"Completed in {((time.time() - started_at) / 60):.1f} minutes")
    print(f"Results root: {Path('gigaam_benchmark_results_all').resolve()}")


if __name__ == "__main__":
    main()
