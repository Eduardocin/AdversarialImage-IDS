#!/bin/bash
#SBATCH --job-name=adv-mnist
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --time=02:00:00

set -eo pipefail

if [ -f ~/.zshrc ]; then
  source ~/.zshrc
elif [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi

conda activate adversarialimage-ids-legacy

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

hostname
date
nvidia-smi || true

python scripts/mnist/check_environment.py
python scripts/download_mnist_test.py

date
