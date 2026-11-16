"""
Threshold Optimization
Finds the optimal classification threshold per model per fold
(instead of 0.5 by default), maximizing F1 or balanced accuracy.
Threshold is tuned on a held-out validation image.

Usage:
    python part3_threshold_opt.py [--feature_set A|B|C] [--metric f1|balanced_acc]
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from part3_modeling import load_labeled_data, build_feature_matrix, get_models, compute_metrics

warnings.filterwarnings("ignore")


def find_optimal_threshold(y_true, y_prob, metric="f1"):
    """
    Auxiliary function to try thresholds on validation probabilities
    and return the one maximising the chosen metric.
    ONly used on validation fold.
    """
    best_thr, best_score = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= thr).astype(int)
        if metric == "f1":
        	score = f1_score(y_true, y_pred, zero_division=0)
        else:
        	score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr, best_score


def run_threshold_cv(X, y, groups, models, metric, seed, output_dir):
    """
    Running find_optimal_threshold on images,
    3-fold CV comparing fixed 0.5 vs optimized threshold.
    Uses one of the two remaining training images as a validation set
    for threshold tuning.
    """
    image_names = sorted(set(groups))
    results, threshold_records = [], []

    for fold_idx, test_image in enumerate(image_names):
        train_mask = groups != test_image
        test_mask = groups == test_image

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        train_groups = groups[train_mask]

        scaler = StandardScaler() # to standardize features
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # split (inner) training for threshold tuning
        val_image = sorted(set(train_groups))[0]
        inner_train = train_groups != val_image
        inner_val = train_groups == val_image

        for model_name, model_template in models.items():
            # fit on inner train, tune threshold on inner val
            m_inner = clone(model_template)
            if model_name == "SVM_RBF" and inner_train.sum() > 30000:
                rng = np.random.RandomState(seed)
                idx = rng.choice(inner_train.sum(), 30000, replace=False)
                m_inner.fit(X_train_s[inner_train][idx], y_train[inner_train][idx])
            else:
                m_inner.fit(X_train_s[inner_train], y_train[inner_train])

            y_val_prob = m_inner.predict_proba(X_train_s[inner_val])[:, 1]
            opt_thr, val_score = find_optimal_threshold(y_train[inner_val], y_val_prob, metric)

            threshold_records.append({
                "model": model_name, "fold": fold_idx,
                "test_image": test_image, "val_image": val_image,
                "opt_threshold": round(opt_thr, 3),
                "val_score": round(val_score, 4),
                "tuning_metric": metric,
            })

            # refit on full training fold, evaluate on test with both thresholds
            m_full = clone(model_template)
            if model_name == "SVM_RBF" and len(X_train_s) > 30000:
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(X_train_s), 30000, replace=False)
                m_full.fit(X_train_s[idx], y_train[idx])
            else:
                m_full.fit(X_train_s, y_train)

            y_test_prob = m_full.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, y_test_prob)

            for thr_label, thr in [("fixed_0.5", 0.5), ("optimized", opt_thr)]:
                y_pred = (y_test_prob >= thr).astype(int)
                results.append({
                    "model": model_name, "threshold_type": thr_label,
                    "fold": fold_idx, "test_image": test_image,
                    "threshold_used": round(thr, 3),
                    "ROC_AUC": round(auc, 4),
                    "F1":      round(f1_score(y_test, y_pred, zero_division=0), 4),
                    "Bal_Acc": round(balanced_accuracy_score(y_test, y_pred), 4),
                })

            delta = results[-1]["F1"] - results[-2]["F1"]
            print(f"{model_name:20s} fold={fold_idx}  opt_thr={opt_thr:.2f}"
                  f"F1: {results[-2]['F1']:.4f} -> {results[-1]['F1']:.4f} (delta {delta:+.4f})")

    results_df = pd.DataFrame(results)
    threshold_df = pd.DataFrame(threshold_records)

    summary = (
        results_df.groupby(["model", "threshold_type"])
        .agg(F1_mean=("F1","mean"),         F1_std=("F1","std"),
             Bal_Acc_mean=("Bal_Acc","mean"), Bal_Acc_std=("Bal_Acc","std"),
             ROC_AUC_mean=("ROC_AUC","mean"))
        .round(4).reset_index()
    )
    return results_df, summary, threshold_df


# ── plots ─────────────────────────────────────────────────────
def plot_threshold_comparison(summary, metric, output_dir):
    models = summary["model"].unique()
    x, width = np.arange(len(models)), 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (mean_col, std_col, title) in zip(axes, [
        ("F1_mean","F1_std","F1 Score"),
        ("Bal_Acc_mean", "Bal_Acc_std", "Balanced Accuracy"),
    ]):
        for offset, variant, color, label in [
            (-width/2, "fixed_0.5", "steelblue", "Fixed 0.5"),
            (+width/2, "optimized", "coral",     f"Optimized ({metric})"),
        ]:
            sub = summary[summary["threshold_type"] == variant]
            ax.bar(x + offset, sub[mean_col].values, width,
                   yerr=sub[std_col].values, capsize=4,
                   label=label, color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{title}: Fixed vs Optimized Threshold")
        ax.set_ylim(0, 1.05)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "threshold_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_optimal_thresholds(threshold_df, output_dir):
    """Strip plot of where the optimal threshold lands per model/fold."""
    models = threshold_df["model"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (model, color) in enumerate(zip(models, colors)):
        sub = threshold_df[threshold_df["model"] == model]
        ax.scatter([i] * len(sub), sub["opt_threshold"], color=color, s=80, zorder=3)
        ax.hlines(sub["opt_threshold"].mean(), i - 0.3, i + 0.3, color=color, linewidth=2)

    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4, label="0.5 baseline")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Optimal Threshold")
    ax.set_title("Optimal Thresholds per Model and Fold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "optimal_thresholds.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_set", default="A", choices=["A", "B", "C"])
    parser.add_argument("--metric", default="f1", choices=["f1", "balanced_acc"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="part3_threshold_results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Threshold Optimization")
    print(f"  Feature set : {args.feature_set}  |  Metric: {args.metric}")
    print(f"  Output      : {args.output_dir}/")
    print("=" * 60)

    dfs    = load_labeled_data()
    X, y, groups, feature_cols = build_feature_matrix(dfs, args.feature_set)
    models = get_models(args.seed)

    print(f"\nRunning CV with threshold optimization (metric={args.metric})...")
    results_df, summary, threshold_df = run_threshold_cv(
        X, y, groups, models, args.metric, args.seed, args.output_dir
    )

    results_df.to_csv(os.path.join(args.output_dir, "cv_results.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir,    "summary.csv"),    index=False)
    threshold_df.to_csv(os.path.join(args.output_dir, "thresholds.csv"), index=False)

    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))

    plot_threshold_comparison(summary, args.metric, args.output_dir)
    plot_optimal_thresholds(threshold_df, args.output_dir)

    print(f"\nDone. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
