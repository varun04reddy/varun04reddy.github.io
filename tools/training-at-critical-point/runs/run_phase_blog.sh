#!/bin/bash
#SBATCH --job-name=crit_pt_blog
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=0-48:00:00
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

ANALYSIS_SCRIPT="${1:-/n/home00/varunreddy/varun04reddy.github.io/tools/training-at-critical-point/train_phase_blog.py}"
VENV_DIR="/n/home00/varunreddy/dynamics/venv"
PYTHON_EXEC="${VENV_DIR}/bin/python"

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "Error: Analysis script not found at $ANALYSIS_SCRIPT"
    exit 1
fi
if [ ! -x "$PYTHON_EXEC" ]; then
    echo "Error: Python executable not found at $PYTHON_EXEC"
    exit 1
fi

cd /n/home00/varunreddy/varun04reddy.github.io
echo "Host: $(hostname)"
echo "CUDA: $($PYTHON_EXEC -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' 2>/dev/null || echo n/a)"
$PYTHON_EXEC "$ANALYSIS_SCRIPT" --all "${@:2}"
exit $?
