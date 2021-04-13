#!/bin/bash
#
#SBATCH --job-name=l5kit
#SBATCH --output=l5kit.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mail-type=END
module load python3/3.6.6 cuda/10.0
source "$ROOT/venv/bin/activate"
jupyter lab --no-browser --port=10001