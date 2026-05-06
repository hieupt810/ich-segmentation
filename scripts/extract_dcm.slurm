#!/bin/bash
#SBATCH --job-name=extract_rsna_intracranial_hemorrhage
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=40:00:00
#SBATCH --output=extract_dcm.out
#SBATCH --error=extract_dcm.err

## Print some info about the job
echo "Job ID: $SLURM_JOB_ID | Running on partition: $SLURM_JOB_PARTITION"
echo "Job started at: $(date)"
echo "-------------------------------"

## Reset and load necessary modules
module reset
module load \
    GCC/14.2.0 \
    Anaconda3

## Activate the conda environment
conda activate research_env

## Run the Python script
python tools/extract_dcm.py data/rsna-intracranial-hemorrhage-detection/stage_2_train \
    --output data/rsna \
    --workers 8

## Print job completion info
du -h data/rsna-intracranial-hemorrhage-detection
echo "-------------------------------"
echo "Job completed at: $(date)"