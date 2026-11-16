#!/bin/bash

#SBATCH --account=YOUR_ALLOCATION       # don't change this
#SBATCH --job-name=lab2_autoencoder
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5          # GPU-shared allows max 5 cpus per GPU
#SBATCH --time 001:10:00
#SBATCH -o embed.out          
#SBATCH -e embed.err          

#SBATCH --partition=GPU-shared     # don't change this unless you need 8 GPUs
#SBATCH --gpus=1                   # don't increase this unless you need more than 1 GPU

echo "Job ID: $SLURM_JOB_ID"
echo "Running in directory: $(pwd)"
echo "Time: $(date)"

module load anaconda3
conda activate env_214

python get_embeddings_hyparam.py $1 $2