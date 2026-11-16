#!/bin/bash
set -euo pipefail

#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=part3_cloud
#SBATCH --cpus-per-task=5
#SBATCH --time=00:30:00
#SBATCH -o part3_%j.out
#SBATCH -e part3_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

# load our conda env on PSC Bridges-2
module load anaconda3
conda activate stat214

echo "========================================"
echo "Part 3 Modeling Job"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $(which python)"
echo "========================================"

# make sure we are in the right directory
cd ~/lab2-214-temp/lab2/code

# read command-line args; defaults if not provided
# usage: sbatch part3_job.sh <FEATURE_SET> <SEED> <TOP_K>
FEATURE_SET=${1:-A}
SEED=${2:-42}
TOP_K=${3:-0}

echo "Feature set: $FEATURE_SET"
echo "Seed: $SEED"
echo "Top-k rerun: $TOP_K"

# feature sets B and C need autoencoder embeddings
# generate them here so the modeling script can find the CSVs
if [ "$FEATURE_SET" = "B" ]; then
    echo "Generating AE8 embeddings for the 3 labeled images..."
    python part3_gen_embeddings.py --emb_size 8
    # quick check that all 3 CSV files were actually created
    for img in O012791 O013257 O013490; do
        test -f "../data/${img}_ae8.csv"
    done
    echo "AE8 CSV files verified."
fi
if [ "$FEATURE_SET" = "C" ]; then
    echo "Generating AE16 embeddings for the 3 labeled images..."
    python part3_gen_embeddings.py --emb_size 16
    for img in O012791 O013257 O013490; do
        test -f "../data/${img}_ae16.csv"
    done
    echo "AE16 CSV files verified."
fi

# run the main modeling pipeline
python part3_modeling.py \
    --feature_set "$FEATURE_SET" \
    --seed "$SEED" \
    --top_k "$TOP_K" \
    --output_dir "part3_results_${FEATURE_SET}_seed${SEED}"

echo "Job finished at $(date)"
