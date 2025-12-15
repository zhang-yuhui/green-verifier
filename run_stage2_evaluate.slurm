#!/bin/bash
#SBATCH --job-name=stage2_evaluate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2               
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err

set -euo pipefail

module purge
module load gcc/13.3.0
module load python/3.11.9
module load cuda/12.4.0

/project2/swabhas_1765/ray-alex/green-verifier/venv/bin/python model/stage2_evaluate.py
