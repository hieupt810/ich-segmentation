#!/bin/bash
#SBATCH --job-name=train_ct_mae
#SBATCH --partition=amperenodes-medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --output=train_mae.out
#SBATCH --error=train_mae.err

set -euo pipefail

HF_TOKEN="${1:?Please provide the Hugging Face token as the first argument.}"
export HF_TOKEN="$HF_TOKEN"

## Print some info about the job
echo "Job ID: $SLURM_JOB_ID | Running on partition: $SLURM_JOB_PARTITION"
echo "Job started at: $(date)"
echo "-------------------------------"

## Reset and load necessary modules
module reset
module load \
    GCC/14.2.0 \
    Anaconda3 \
    CUDA/11.8.0 \
    cuda11.8/toolkit \
    cuDNN/8.9.2.26-CUDA-11.8.0

## Activate the conda environment
conda activate research_env

## Run the Python script
python main.py --train

## Print job completion info
echo "-------------------------------"
echo "Job completed at: $(date)"