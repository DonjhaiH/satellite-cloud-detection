#!/bin/bash
set -euo pipefail

# posthoc job - runs all 9 diagnostic analyses for part 3
# usage: sbatch part3_posthoc_job.sh C 15 5 full 5 1 9
# args: FEATURE_SET SEED TOP_K MODE N_UNLABELED START_STEP END_STEP
# start/end step lets us resume partial runs without re-doing everything

#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=part3_posthoc
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH -o part3_posthoc_%j.out
#SBATCH -e part3_posthoc_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1

module load anaconda3
conda activate stat214

echo "========================================"
echo "Part 3 Post-hoc Job"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $(which python)"
echo "========================================"

cd ~/lab2-214-temp/lab2/code

# parse args with sane defaults
FEATURE_SET=${1:-C}
SEED=${2:-15}
TOP_K=${3:-5}
MODE=${4:-quick}       # quick skips the slow analyses (perturbation, hyperparam, seed, ensemble)
N_UNLABELED=${5:-5}    # how many unlabeled images to predict on
START_STEP=${6:-1}     # resume from this step
END_STEP=${7:-9}       # stop after this step

echo "Feature set: $FEATURE_SET"
echo "Seed: $SEED"
echo "Top-k: $TOP_K"
echo "Mode: $MODE"
echo "n_unlabeled: $N_UNLABELED"
echo "Step range: $START_STEP..$END_STEP"

# generate AE embeddings if using feature set B or C
# the posthoc script needs these for building the feature matrix
if [ "$FEATURE_SET" = "B" ]; then
    echo "Ensuring AE8 embeddings exist..."
    python part3_gen_embeddings.py --emb_size 8
fi
if [ "$FEATURE_SET" = "C" ]; then
    echo "Ensuring AE16 embeddings exist..."
    python part3_gen_embeddings.py --emb_size 16
fi

# output dir includes feature set, seed, and mode so we can have multiple runs
OUT_DIR="part3_posthoc_${FEATURE_SET}_seed${SEED}_${MODE}"

python part3_posthoc.py \
    --feature_set "$FEATURE_SET" \
    --seed "$SEED" \
    --top_k "$TOP_K" \
    --mode "$MODE" \
    --n_unlabeled "$N_UNLABELED" \
    --start_step "$START_STEP" \
    --end_step "$END_STEP" \
    --output_dir "$OUT_DIR"

echo "Finished at $(date)"
echo "Outputs: $OUT_DIR/"
