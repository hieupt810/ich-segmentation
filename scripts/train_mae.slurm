#!/bin/bash
#SBATCH --job-name=train_mae_pipeline
#SBATCH --partition=amperenodes-medium
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=train_mae.out
#SBATCH --error=train_mae.err

set -euo pipefail

HF_TOKEN="${1:?Please provide the Hugging Face token as the first argument.}"
export HF_TOKEN="$HF_TOKEN"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

## Print some info about the job
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
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
python main.py train-mae

## Print job completion info
echo "-------------------------------"
echo "Job completed at: $(date)"