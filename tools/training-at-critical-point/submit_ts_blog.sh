#!/bin/bash
#SBATCH --job-name=ts_blog
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --time=0-02:00:00
#SBATCH --partition=kempner
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --output=/n/home00/varunreddy/varun04reddy.github.io/tools/training-at-critical-point/runs/logs/ts_blog_%j.out
#SBATCH --error=/n/home00/varunreddy/varun04reddy.github.io/tools/training-at-critical-point/runs/logs/ts_blog_%j.err

cd /n/home00/varunreddy/varun04reddy.github.io/tools/training-at-critical-point
/n/home00/varunreddy/dynamics/venv/bin/python train_phase_blog.py --all
