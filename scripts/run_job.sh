#!/usr/bin/bash
#SBATCH --job-name arc_no_expl_sweep
#SBATCH --account OPEN-29-45
#SBATCH --partition qgpu
#SBATCH --gpus 1

ml purge
ml load Python/3.11.5-GCCcore-13.2.0
. ./venv/bin/activate

srun python3 czech-llm-finetuning/HPO_test.py

