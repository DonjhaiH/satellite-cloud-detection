#!/bin/bash

# --- CONFIGURATION ---
# You can change these variables here once to update the whole pipeline
AE_CONFIG="configs/ae_default.yaml"
PLOT_CONFIG="configs/best.yaml"
PLOT_CKPT="checkpoints/best_emb16-epoch=067.ckpt"
FEATURE_SET="C"
SEED=15
TOP_K=5
MODE="full"

echo "Submitting ML Pipeline for Feature Set: $FEATURE_SET, Seed: $SEED"


# 2. AUTOENCODER
AE_ID=$(sbatch --parsable part2_run_autoencoder.sh $AE_CONFIG)
echo "Step 1: Autoencoder submitted (Job $AE_ID)"

# 3. GENERATE PLOTS
# Depends on Autoencoder finishing (afterok)
PLOT_ID=$(sbatch --parsable --dependency=afterok:$AE_ID part2_run_generate_plot.sh $PLOT_CONFIG $PLOT_CKPT)
echo "Step 2: Plotting submitted (Job $PLOT_ID) - waiting for $AE_ID"

# 4. MODELING
# Depends on Plotting finishing
MODEL_ID=$(sbatch --parsable --dependency=afterok:$PLOT_ID part3_run_models.sh $FEATURE_SET $SEED $TOP_K)
echo "Step 3: Modeling submitted (Job $MODEL_ID) - waiting for $PLOT_ID"

# 5. POSTHOC ANALYSIS
# Depends on Modeling finishing
POST_ID=$(sbatch --parsable --dependency=afterok:$MODEL_ID part3_run_posthoc.sh $FEATURE_SET $SEED $TOP_K $MODE 5 1 9)
echo "Step 4: Post-hoc submitted (Job $POST_ID) - waiting for $MODEL_ID"

echo "-------------------------------------------"
echo "All jobs queued successfully."
echo "Use 'squeue -u $USER' to monitor progress."