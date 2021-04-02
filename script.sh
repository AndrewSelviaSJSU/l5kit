#!/bin/bash
#
#SBATCH --job-name=aselvia
#SBATCH --output=aselvia.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mail-user=andrew.selvia@sjsu.edu
#SBATCH --mail-type=END
module load python3/3.6.6 cuda/10.0
source cmpe258/l5kit/venv/bin/activate
jupyter lab --no-browser --port=10001