#!/bin/bash
#SBATCH --job-name=download_rsna_intracranial_hemorrhage
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=download_rsna.out
#SBATCH --error=download_rsna.err

## Print some info about the job
echo "Job ID: $SLURM_JOB_ID | Running on partition: $SLURM_JOB_PARTITION"
echo "Job started at: $(date)"

## Reset and load necessary modules
module reset
module load \
    GCC/14.2.0 \
    Anaconda3

## Activate the conda environment
conda activate research_env

## Run the Python script
python tools/download_rsna.py

## Print job completion info
echo "Job completed at: $(date)"