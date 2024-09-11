#!/bin/bash
ml load Python/3.11.5-GCCcore-13.2.0
. ./venv/bin/activate

export HF_HOME="/mnt/proj2/open-29-45/poludmik/textvision/cache"
export HF_DATASETS_CACHE="/mnt/proj2/open-29-45/poludmik/textvision/cache"
# squeue -u poludmik -l