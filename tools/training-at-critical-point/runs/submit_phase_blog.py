#!/usr/bin/env python3
"""Submit one SLURM job that runs all critical-point blog sweeps in-process."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch command only")
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    worker_script = current_dir / "run_phase_blog.sh"
    analysis_script = (current_dir.parent / "train_phase_blog.py").resolve()
    logs_dir = current_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    assert worker_script.exists(), worker_script
    assert analysis_script.exists(), analysis_script

    cmd = [
        "sbatch",
        f"--output={logs_dir}/phase_blog_%j.out",
        f"--error={logs_dir}/phase_blog_%j.err",
        str(worker_script),
        str(analysis_script),
    ]
    print("Submitting bundled GPU job (all sweeps in one allocation):")
    print(" ", " ".join(cmd))
    if args.dry_run:
        print("(dry-run — not submitted)")
        return

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    print(result.stdout.strip())
    # sbatch prints: Submitted batch job 12345
    for line in result.stdout.strip().splitlines():
        if "Submitted batch job" in line:
            job_id = line.split()[-1]
            print(f"JOB_ID={job_id}")


if __name__ == "__main__":
    main()
