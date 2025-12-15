#!/bin/bash
#SBATCH --job-name=stage2_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --account=swabhas_1765
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err

set -euo pipefail

module purge
module load gcc/13.3.0
module load python/3.11.9
module load cuda/12.4.0

/project2/swabhas_1765/ray-alex/green-verifier/venv/bin/python stage2_training.py
