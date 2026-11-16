"""
Feature Engineering (with AE features, but also manual)

Feature sets:
    A = 8 raw base features
    B = A + AE8 embeddings
    C = A + AE16 embeddings
    D = A + 3 interaction features
    E = D + AE8 embeddings
    F = D + AE16 embeddings

Interaction (manual) features:
    NDAI_x_SD      = NDAI * SD
    NDAI_x_log1pSD = NDAI * log1p(SD)
    log1p_SD       = log1p(SD)

Usage:
    python part2_feature_engineering.py [--sets A B C D E F] [--seed 42]
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from part3_modeling import load_labeled_data, BASE_FEATURES

warnings.filterwarnings("ignore")

INTERACTION_FEATURES = ["NDAI_x_SD", "NDAI_x_log1pSD", "log1p_SD"]


# ── interaction feature engineering ──────────────────────────
def engineer_features(df):
    """
    Adds the 3 interaction features to a DataFrame.
    Rationale:
      NDAI_x_SD - because clouds tend to have high NDAI and high SD together
      NDAI_x_log1pSD - same idea as before but but log SD is a little more stable
      log1p_SD
    """
    out = df.copy()
    log1p_sd = np.log1p(out["SD"].clip(lower=0))
    out["NDAI_x_SD"]      = out["NDAI"] * out["SD"]
    out["NDAI_x_log1pSD"] = out["NDAI"] * log1p_sd
    out["log1p_SD"]       = log1p_sd
    return out


# feature matrix builder for all 6 sets based on original modeling
def build_matrix(dfs, feature_set="A"):
    """
    Builds X, y, groups for any of the 6 feature sets.
    Mirrors part3_modeling.build_feature_matrix but supports D, E, F.
    """
    all_dfs = []
    for name, df in dfs.items():
        labeled = df[df["label"].isin([1, -1])].copy()

        if feature_set in ("B", "E"):
            ae_path = f"../data/{name}_ae8.csv"
            if os.path.exists(ae_path):
                labeled = labeled.merge(pd.read_csv(ae_path), on=["y", "x"], how="left")
            else:
                print(f"AE8 embeddings not found: {ae_path}")

        elif feature_set in ("C", "F"):
            ae_path = f"../data/{name}_ae16.csv"
            if os.path.exists(ae_path):
                labeled = labeled.merge(pd.read_csv(ae_path), on=["y", "x"], how="left")
            else:
                print(f" AE16 embeddings not found: {ae_path}")

        if feature_set in ("D", "E", "F"):
            labeled = engineer_features(labeled)

        all_dfs.append(labeled)

    combined = pd.concat(all_dfs, ignore_index=True)

    feature_cols = list(BASE_FEATURES)
    if feature_set in ("D", "E", "F"):
        feature_cols += INTERACTION_FEATURES
    if feature_set in ("B", "E"):
        feature_cols += [c for c in combined.columns if c.startswith("ae")][:8]
    elif feature_set in ("C", "F"):
        feature_cols += [c for c in combined.columns if c.startswith("ae")][:16]

    feature_cols = [c for c in feature_cols if c in combined.columns]

    X = combined[feature_cols].values
    y = (combined["label"].values > 0).astype(int)
    groups = combined["image"].values
    return X, y, groups, feature_cols


def compute_mutual_inf(dfs, n_repeats=5, seed=42):
    """
    Compute MI for base + interaction features,
    averaged over multiple runs to reduce estimation noise.
    """
    parts = []
    for df in dfs.values():
        labeled = df[df["label"].isin([1, -1])].copy()
        labeled = engineer_features(labeled)
        parts.append(labeled)
    combined = pd.concat(parts, ignore_index=True)

    feat_cols = BASE_FEATURES + INTERACTION_FEATURES
    X = combined[feat_cols].values
    y = (combined["label"].values > 0).astype(int)

    scores = np.array([
        mutual_info_classif(X, y, discrete_features=False, random_state=seed + i)
        for i in range(n_repeats)
    ])
    types = ["base"] * len(BASE_FEATURES) + ["interaction"] * len(INTERACTION_FEATURES)

    return pd.DataFrame({
        "feature": feat_cols,
        "MI_mean": scores.mean(axis=0),
        "MI_std":  scores.std(axis=0),
        "type":    types,
    }).sort_values("MI_mean", ascending=False).reset_index(drop=True)


def run_cv(X, y, groups, label, seed=42):
    """Image-level 3-fold CV with GradientBoosting."""
    image_names = sorted(set(groups))
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=seed,
    )
    rows = []
    for fold_idx, test_image in enumerate(image_names):
        train_mask = groups != test_image
        test_mask  = groups == test_image

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[train_mask])
        X_te_s = scaler.transform(X[test_mask])
        y_tr, y_te = y[train_mask], y[test_mask]

        m = clone(model)
        m.fit(X_tr_s, y_tr)
        y_prob = m.predict_proba(X_te_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        rows.append({
            "feature_set": label,
            "fold":        fold_idx,
            "test_image":  test_image,
            "ROC_AUC":     round(roc_auc_score(y_te, y_prob), 4),
            "PR_AUC":      round(average_precision_score(y_te, y_prob), 4),
            "F1":          round(f1_score(y_te, y_pred, zero_division=0), 4),
            "Bal_Acc":     round(balanced_accuracy_score(y_te, y_pred), 4),
        })
        print(f"  [{label}] fold={fold_idx} ({test_image})  "
              f"ROC={rows[-1]['ROC_AUC']:.4f}  "
              f"PR={rows[-1]['PR_AUC']:.4f}  "
              f"F1={rows[-1]['F1']:.4f}")
    return pd.DataFrame(rows)


def plot_mi(mi_df, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#e63946" if t == "interaction" else "steelblue"
              for t in mi_df["type"][::-1]]
    ax.barh(mi_df["feature"][::-1], mi_df["MI_mean"][::-1],
            xerr=mi_df["MI_std"][::-1], capsize=4, color=colors, alpha=0.85)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="steelblue", alpha=0.85, label="Base"),
        Patch(color="#e63946",   alpha=0.85, label="Interaction"),
    ], fontsize=9)
    ax.set_xlabel("Mutual Information (mean ± std, 5 runs)")
    ax.set_title("Feature MI: Base vs Interaction Features")
    plt.tight_layout()
    path = os.path.join(output_dir, "mutual_information.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_top3_distributions(dfs, top3, output_dir):
    """Class-split histograms for the top-3 MI features."""
    parts = []
    for df in dfs.values():
        labeled = df[df["label"].isin([1, -1])].copy()
        labeled = engineer_features(labeled)
        parts.append(labeled)
    combined = pd.concat(parts, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, feat in zip(axes, top3):
        cloud   = combined[combined["label"] ==  1][feat].values
        nocloud = combined[combined["label"] == -1][feat].values
        lo = np.percentile(np.concatenate([cloud, nocloud]), 1)
        hi = np.percentile(np.concatenate([cloud, nocloud]), 99)
        bins = np.linspace(lo, hi, 60)
        ax.hist(nocloud, bins=bins, density=True, alpha=0.55,
                color="#457b9d", label="No-Cloud")
        ax.hist(cloud,   bins=bins, density=True, alpha=0.55,
                color="#e63946", label="Cloud")
        ax.set_title(feat)
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.suptitle("Top-3 Features by Class", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "top3_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cv_comparison(summary, output_dir):
    """Grouped bar chart comparing all feature sets across metrics."""
    metrics = [
        ("ROC_AUC_mean", "ROC_AUC_std", "ROC-AUC"),
        ("PR_AUC_mean",  "PR_AUC_std",  "PR-AUC"),
        ("F1_mean",      "F1_std",       "F1"),
        ("Bal_Acc_mean", "Bal_Acc_std",  "Balanced Acc"),
    ]
    feature_sets = summary["feature_set"].unique()
    x      = np.arange(len(metrics))
    width  = 0.8 / len(feature_sets)
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e63946", "#457b9d"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (fs, color) in enumerate(zip(feature_sets, colors)):
        row    = summary[summary["feature_set"] == fs].iloc[0]
        vals   = [row[m] for m, _, _ in metrics]
        stds   = [row[s] for _, s, _ in metrics]
        offset = (i - len(feature_sets) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=stds, capsize=3,
               label=fs, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([t for _, _, t in metrics])
    ax.set_ylabel("Score (mean ± std across folds)")
    ax.set_title("GBT Performance: Feature Sets A–F")
    ax.set_ylim(0.5, 1.05)
    ax.legend(title="Feature Set", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "cv_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_roc_auc_progression(summary, output_dir):
    """
    Line plot showing ROC-AUC across feature sets A-F,
    making the contribution of each addition clearly visible.
    """
    order  = ["A", "B", "C", "D", "E", "F"]
    labels = {
        "A": "A\n(base)",
        "B": "B\n(+AE8)",
        "C": "C\n(+AE16)",
        "D": "D\n(+interact)",
        "E": "E\n(+interact\n+AE8)",
        "F": "F\n(+interact\n+AE16)",
    }
    present = [s for s in order if s in summary["feature_set"].values]
    means   = [summary[summary["feature_set"] == s]["ROC_AUC_mean"].values[0] for s in present]
    stds    = [summary[summary["feature_set"] == s]["ROC_AUC_std"].values[0]  for s in present]
    x       = np.arange(len(present))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, means, marker="o", color="steelblue", linewidth=2)
    ax.fill_between(x,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.15, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in present], fontsize=9)
    ax.set_ylabel("ROC-AUC (mean ± std)")
    ax.set_title("ROC-AUC Progression Across Feature Sets")
    ax.set_ylim(0.5, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_auc_progression.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sets", nargs="+", default=["A", "B", "C", "D", "E", "F"],
        choices=["A", "B", "C", "D", "E", "F"],
        help="Feature sets to run (default: all six)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="part2_feature_engineering")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Part 2 — Feature Engineering")
    print(f"  Sets to run : {args.sets}")
    print(f"  Output      : {args.output_dir}/")
    print("=" * 60)
    print("\nFeature set legend:")
    print("  A = 8 raw base features")
    print("  B = A + AE8 embeddings")
    print("  C = A + AE16 embeddings")
    print("  D = A + 3 interaction features")
    print("  E = D + AE8 embeddings")
    print("  F = D + AE16 embeddings")
    print("\nInteraction features:")
    print("  NDAI_x_SD      = NDAI * SD")
    print("  NDAI_x_log1pSD = NDAI * log1p(SD)")
    print("  log1p_SD       = log1p(SD)")

    dfs = load_labeled_data()

    # 1. mutual information
    print("\n[1/3] Computing mutual information (base + interaction features)...")
    mi_df = compute_mutual_inf(dfs, n_repeats=5, seed=args.seed)
    mi_df.to_csv(os.path.join(args.output_dir, "mutual_information.csv"), index=False)
    print(mi_df.to_string(index=False))
    top3 = mi_df["feature"].iloc[:3].tolist()
    print(f"\n  Top-3 features by MI: {top3}")
    plot_mi(mi_df, args.output_dir)
    plot_top3_distributions(dfs, top3, args.output_dir)

    # 2. CV for each requested feature set
    print("\n[2/3] Running CV across feature sets...")
    all_results = []
    for fs in args.sets:
        print(f"\n  --- Feature set {fs} ---")
        X, y, groups, feat_cols = build_matrix(dfs, feature_set=fs)
        print(f"  {len(feat_cols)} features: {feat_cols}")
        res = run_cv(X, y, groups, label=fs, seed=args.seed)
        all_results.append(res)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(os.path.join(args.output_dir, "cv_results.csv"), index=False)

    summary = (
        all_results_df.groupby("feature_set")
        .agg(
            ROC_AUC_mean=("ROC_AUC", "mean"), ROC_AUC_std=("ROC_AUC", "std"),
            PR_AUC_mean=("PR_AUC",   "mean"), PR_AUC_std=("PR_AUC",   "std"),
            F1_mean=("F1",           "mean"), F1_std=("F1",           "std"),
            Bal_Acc_mean=("Bal_Acc", "mean"), Bal_Acc_std=("Bal_Acc", "std"),
        )
        .round(4).reset_index()
    )
    summary.to_csv(os.path.join(args.output_dir, "cv_summary.csv"), index=False)

    print("\n===== CV SUMMARY =====")
    print(summary.to_string(index=False))

    # 3. plots
    print("\n[3/3] Generating plots...")
    plot_cv_comparison(summary, args.output_dir)
    plot_roc_auc_progression(summary, args.output_dir)

    print(f"\nDone. Results saved to {args.output_dir}/")
    print("\nKey comparisons to report:")
    print("  A vs D       — do interaction features help on their own?")
    print("  A vs B vs C  — does AE help without interaction features?")
    print("  B vs E       — does adding interaction features help on top of AE8?")
    print("  C vs F       — does adding interaction features help on top of AE16?")


if __name__ == "__main__":
    main()
