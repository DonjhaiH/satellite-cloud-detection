#!/bin/bash
 
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=lab2_plot
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time 00:15:00
#SBATCH -o plot.out
#SBATCH -e plot.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
 
echo "Job ID: $SLURM_JOB_ID"
echo "Running in directory: $(pwd)"
echo "Time: $(date)"
 
module load anaconda3
conda activate env_214
 
# Usage: sbatch run_generate_plot.sh configs/best.yaml checkpoints/best_emb16-epoch=067.ckpt
python generate_plot.py $1 $2