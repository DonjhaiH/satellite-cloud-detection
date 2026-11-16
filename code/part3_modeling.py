"""
part3_modeling.py
Runs all the classifiers for part 3. Image-level 3-fold CV.

  python part3_modeling.py --feature_set C --seed 42 --top_k 5
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- file paths and column names ---
LABELED_FILES = {
    "O012791": "../data/image_data/O012791.npz",
    "O013257": "../data/image_data/O013257.npz",
    "O013490": "../data/image_data/O013490.npz",
}

COLUMN_NAMES = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN", "label"]
BASE_FEATURES = ["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]


def load_labeled_data():
    """Read the 3 labeled npz files into dataframes."""
    dfs = {}
    for name, path in LABELED_FILES.items():
        # each npz has one array with shape (n_pixels, 11)
        arr = np.load(path)["arr_0"]
        df = pd.DataFrame(arr, columns=COLUMN_NAMES)
        # tag which image this pixel came from (needed for CV grouping)
        df["image"] = name
        dfs[name] = df
    return dfs


def build_feature_matrix(dfs, feature_set="A"):
    """Build X, y, group arrays. A=base only, B=base+AE8, C=base+AE16."""
    all_dfs = []
    for name, df in dfs.items():
        # label column is 1 (cloud) or -1 (clear); 0 means unlabeled
        labeled = df[df["label"].isin([1, -1])].copy()

        if feature_set in ("B", "C"):
            # for feature sets B and C we need the autoencoder embeddings
            # these are generated separately by part3_gen_embeddings.py
            ae_size = 8 if feature_set == "B" else 16
            ae_path = f"../data/{name}_ae{ae_size}.csv"
            if os.path.exists(ae_path):
                ae_df = pd.read_csv(ae_path)
                # merge on pixel coords so each row gets its AE dims
                labeled = labeled.merge(ae_df, on=["y", "x"], how="left")
                print(f"  Merged {ae_path} ({len(ae_df.columns)-2} AE dims)")
            else:
                print(f"  [ERROR] AE embeddings not found at {ae_path}")
                print(f"  Run: python part3_gen_embeddings.py --emb_size {ae_size}")
                raise FileNotFoundError(f"Missing {ae_path}. Generate embeddings first.")

        all_dfs.append(labeled)

    # stack all three images into one big dataframe
    combined = pd.concat(all_dfs, ignore_index=True)

    # build the feature column list depending on which set we want
    feature_cols = list(BASE_FEATURES)
    if feature_set == "B":
        # grab just the first 8 AE columns
        ae_cols = [c for c in combined.columns if c.startswith("ae")][:8]
        feature_cols += ae_cols
    elif feature_set == "C":
        # all 16 AE columns
        ae_cols = [c for c in combined.columns if c.startswith("ae")][:16]
        feature_cols += ae_cols

    # in case some columns are missing for whatever reason
    feature_cols = [c for c in feature_cols if c in combined.columns]

    X = combined[feature_cols].values
    # convert labels: original is 1/-1, sklearn needs 0/1
    y = (combined["label"].values > 0).astype(int)
    # group labels for image-level CV splits
    groups = combined["image"].values

    print(f"  Feature set {feature_set}: {len(feature_cols)} features, "
          f"{len(y)} samples (cloud={y.sum()}, clear={len(y)-y.sum()})")

    return X, y, groups, feature_cols


def get_models(seed=42):
    """Return dict of all 5 classifiers with default hyperparams.
    We deliberately chose textbook defaults here — see Section 3.7
    in the report for the sensitivity check."""
    models = {
        "LR_L2": LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, solver="lbfgs", random_state=seed
        ),
        "LR_L1": LogisticRegression(
            penalty="l1", C=1.0, max_iter=1000, solver="saga", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            random_state=seed, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=seed
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed
        ),
    }
    return models


def compute_metrics(y_true, y_prob):
    """Compute the four metrics we care about, all at threshold=0.5.
    Returns a dict so the caller can pick what it needs."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),   # precision-recall AUC
        "F1": f1_score(y_true, y_pred),
        "Bal_Acc": balanced_accuracy_score(y_true, y_pred),  # accounts for imbalance
        "y_pred": y_pred,
    }


def select_topk_feature_idx(model, top_k, n_features):
    """Return sorted indices of the top-k most important features.
    Uses |coef| for linear models, impurity importance for trees.
    Returns None if the model does not expose importances (e.g. SVM)."""
    if top_k <= 0 or top_k >= n_features:
        return None

    # figure out how to get importance from this particular model type
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])  # LR coefficients
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_  # RF / GBT
    else:
        return None  # SVM has no built-in importance

    # sort descending by importance, take first k, then re-sort by index
    # (re-sorting by index so the column order stays consistent)
    idx_desc = np.argsort(imp)[::-1]
    return np.sort(idx_desc[:top_k])


def image_level_cv(X, y, groups, models, feature_cols, output_dir, seed=42, top_k=0):
    """Image-level leave-one-out CV (3 folds since we have 3 images).
    For each fold we train on 2 images and test on the held-out one.
    Returns the per-fold results df, an aggregated summary, and a dict
    of all predictions (we keep those around for the plotting functions)."""
    image_names = sorted(set(groups))
    results = []
    all_preds = {}  # {(model, fold, run_type): {...}} for later use

    for fold_idx, test_image in enumerate(image_names):
        print(f"\n--- Fold {fold_idx+1}: test on {test_image} ---")
        # boolean masks for train/test split
        train_mask = groups != test_image
        test_mask = groups == test_image

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # standardize so LR and SVM are on a level playing field
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)  # transform only, no refit

        for model_name, model_template in models.items():
            # clone so each fold gets a fresh model instance
            from sklearn.base import clone
            model = clone(model_template)

            # SVM with RBF kernel is O(n^2) so we subsample to 30k
            if model_name == "SVM_RBF" and len(X_train_s) > 30000:
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(X_train_s), 30000, replace=False)
                model.fit(X_train_s[idx], y_train[idx])
            else:
                model.fit(X_train_s, y_train)

            # get predicted probabilities (or decision scores for SVM)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_s)[:, 1]
            else:
                y_prob = model.decision_function(X_test_s)

            metrics = compute_metrics(y_test, y_prob)

            results.append({
                "model": model_name,
                "run_type": "full",
                "fold": fold_idx,
                "test_image": test_image,
                "n_features": len(feature_cols),
                "selected_features": "ALL",
                "ROC_AUC": metrics["ROC_AUC"],
                "PR_AUC": metrics["PR_AUC"],
                "F1": metrics["F1"],
                "Bal_Acc": metrics["Bal_Acc"],
            })

            key = (model_name, fold_idx, "full")
            all_preds[key] = {
                "y_test": y_test, "y_prob": y_prob, "y_pred": metrics["y_pred"],
                "test_image": test_image,
                "X_test": X_test, "scaler": scaler, "model": model,
                "selected_feature_cols": list(feature_cols),
            }

            print(
                f"  {model_name:20s} [full]  ROC={metrics['ROC_AUC']:.4f}  "
                f"PR={metrics['PR_AUC']:.4f}  F1={metrics['F1']:.4f}  BA={metrics['Bal_Acc']:.4f}"
            )

            # --- optional top-k rerun ---
            # rank features using the TRAIN fold model, then refit on just those
            # (this avoids leaking test-fold info into feature selection)
            if top_k > 0:
                top_idx = select_topk_feature_idx(model, top_k, X_train_s.shape[1])
                if top_idx is None:
                    print(f"  {model_name:20s} [top{top_k}] skipped (no importance available)")
                    continue

                selected_cols = [feature_cols[i] for i in top_idx]
                model_topk = clone(model_template)
                model_topk.fit(X_train_s[:, top_idx], y_train)

                if hasattr(model_topk, "predict_proba"):
                    y_prob_topk = model_topk.predict_proba(X_test_s[:, top_idx])[:, 1]
                else:
                    y_prob_topk = model_topk.decision_function(X_test_s[:, top_idx])

                metrics_topk = compute_metrics(y_test, y_prob_topk)
                results.append({
                    "model": model_name,
                    "run_type": f"top{top_k}",
                    "fold": fold_idx,
                    "test_image": test_image,
                    "n_features": len(selected_cols),
                    "selected_features": "|".join(selected_cols),
                    "ROC_AUC": metrics_topk["ROC_AUC"],
                    "PR_AUC": metrics_topk["PR_AUC"],
                    "F1": metrics_topk["F1"],
                    "Bal_Acc": metrics_topk["Bal_Acc"],
                })

                key_topk = (model_name, fold_idx, f"top{top_k}")
                all_preds[key_topk] = {
                    "y_test": y_test,
                    "y_prob": y_prob_topk,
                    "y_pred": metrics_topk["y_pred"],
                    "test_image": test_image,
                    "X_test": X_test,
                    "scaler": scaler,
                    "model": model_topk,
                    "selected_feature_cols": selected_cols,
                }
                print(
                    f"  {model_name:20s} [top{top_k}] ROC={metrics_topk['ROC_AUC']:.4f}  "
                    f"PR={metrics_topk['PR_AUC']:.4f}  F1={metrics_topk['F1']:.4f}  "
                    f"BA={metrics_topk['Bal_Acc']:.4f}"
                )

    results_df = pd.DataFrame(results)

    # aggregate across folds: mean and std for each metric
    summary = results_df.groupby(["model", "run_type"]).agg(
        ROC_AUC_mean=("ROC_AUC", "mean"),
        ROC_AUC_std=("ROC_AUC", "std"),
        PR_AUC_mean=("PR_AUC", "mean"),
        PR_AUC_std=("PR_AUC", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        Bal_Acc_mean=("Bal_Acc", "mean"),
        Bal_Acc_std=("Bal_Acc", "std"),
    ).round(4).reset_index()

    return results_df, summary, all_preds


def plot_model_comparison(summary, output_dir):
    """Simple horizontal bar chart comparing all model/run combos by ROC-AUC."""
    fig, ax = plt.subplots(figsize=(8, 4))
    models = [f"{r.model} [{r.run_type}]" for r in summary.itertuples(index=False)]
    means = summary["ROC_AUC_mean"].values
    stds = summary["ROC_AUC_std"].values
    ax.barh(models, means, xerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xlabel("ROC-AUC (mean ± std across folds)")
    ax.set_title("Model Comparison — Image-Level 3-Fold CV")
    ax.set_xlim(0.5, 1.0)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved model_comparison.png")


def plot_roc_curves(all_preds, best_model, best_run_type, output_dir):
    """Plot per-fold ROC curves for the best model only."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for key, data in all_preds.items():
        model_name, fold_idx, run_type = key
        # skip everything that isn't the champion
        if model_name != best_model or run_type != best_run_type:
            continue
        fpr, tpr, _ = roc_curve(data["y_test"], data["y_prob"])
        auc = roc_auc_score(data["y_test"], data["y_prob"])
        ax.plot(fpr, tpr, label=f"Fold {fold_idx+1} ({data['test_image']}) AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)  # diagonal reference
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {best_model} [{best_run_type}]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved roc_curves.png")


def plot_feature_importance(all_preds, best_model, best_run_type, output_dir):
    """Bar chart of feature importances from the best model.
    We just grab the last fold's fitted model for this — it is not
    a proper average but good enough for a quick visual."""
    candidate_folds = [k[1] for k in all_preds if k[0] == best_model and k[2] == best_run_type]
    if not candidate_folds:
        print("  Skipping feature importance (no predictions found for best run)")
        return
    last_fold = max(candidate_folds)
    selected_info = all_preds[(best_model, last_fold, best_run_type)]
    model = selected_info["model"]
    feature_cols = selected_info["selected_feature_cols"]

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        title = "Feature Importance (impurity-based)"
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        title = "Feature Importance (|coefficient|)"
    else:
        print("  Skipping feature importance (model type not supported)")
        return

    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([feature_cols[i] for i in idx], imp[idx], color="coral", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title(f"{title} — {best_model} [{best_run_type}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved feature_importance.png")


# ---- entry point ----

def main():
    parser = argparse.ArgumentParser(description="Part 3 Cloud Classification")
    parser.add_argument("--feature_set", type=str, default="A", choices=["A", "B", "C"],
                        help="Feature set to use: A=base, B=base+AE8, C=base+AE16")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="If >0, run an additional fold-wise top-k feature rerun (selected from train fold only)",
    )
    parser.add_argument("--output_dir", type=str, default="part3_results",
                        help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Part 3: Cloud Classification Pipeline")
    print(f"  Feature set: {args.feature_set}")
    print(f"  Seed: {args.seed}")
    print(f"  Top-k rerun: {args.top_k if args.top_k > 0 else 'disabled'}")
    print(f"  Output: {args.output_dir}/")
    print("=" * 60)

    # step 1 — load the raw pixel data from npz files
    print("\n[1/4] Loading labeled data...")
    dfs = load_labeled_data()

    # step 2 — assemble into a single feature matrix with group labels
    print("\n[2/4] Building feature matrix...")
    X, y, groups, feature_cols = build_feature_matrix(dfs, args.feature_set)

    # step 3 — run the main cross-validation loop
    print("\n[3/4] Running image-level 3-fold CV...")
    models = get_models(args.seed)
    results_df, summary, all_preds = image_level_cv(
        X, y, groups, models, feature_cols, args.output_dir, args.seed, top_k=args.top_k
    )

    # save everything to CSV so the report can pick it up later
    results_df.to_csv(os.path.join(args.output_dir, "cv_results.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "model_summary.csv"), index=False)
    if args.top_k > 0:
        # pivot table for easy full-vs-topk comparison
        cmp_df = summary.pivot(index="model", columns="run_type", values="ROC_AUC_mean").reset_index()
        cmp_df.to_csv(os.path.join(args.output_dir, "full_vs_topk_comparison.csv"), index=False)
    print("\n\n===== MODEL COMPARISON (mean ± std) =====")
    print(summary.to_string(index=False))

    # step 4 — plots for the report
    print("\n\n[4/4] Generating plots...")
    # figure out which config won so we can highlight it in the plots
    best_row = summary.loc[summary["ROC_AUC_mean"].idxmax()]
    best_model = best_row["model"]
    best_run_type = best_row["run_type"]
    print(f"  Best run by ROC-AUC: {best_model} [{best_run_type}]")

    plot_model_comparison(summary, args.output_dir)
    plot_roc_curves(all_preds, best_model, best_run_type, args.output_dir)
    plot_feature_importance(all_preds, best_model, best_run_type, args.output_dir)

    print("\n" + "=" * 60)
    print("Done! Results saved to:", args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
