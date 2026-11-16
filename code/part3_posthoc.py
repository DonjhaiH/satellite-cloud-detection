"""
part3_posthoc.py - all the post-hoc diagnostics for part 3.
Convergence, calibration, perturbation, seed stability, unlabeled checks, etc.

  python part3_posthoc.py --feature_set C --seed 15 --mode full
"""

import argparse, os, glob, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, roc_curve, confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# paths + column defs
LABELED_FILES = {
    "O012791": "../data/image_data/O012791.npz",
    "O013257": "../data/image_data/O013257.npz",
    "O013490": "../data/image_data/O013490.npz",
}
COL = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN", "label"]
BASE_FEAT = ["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
UNLABELED_DIR = "../data/image_data"
EMB_CONFIG = {
    8: {
        "checkpoint": "checkpoints/best_emb8-epoch=072.ckpt",
        "n_input_channels": 8,
        "patch_size": 9,
        "embedding_size": 8,
    },
    16: {
        "checkpoint": "checkpoints/best_emb16-epoch=067.ckpt",
        "n_input_channels": 8,
        "patch_size": 9,
        "embedding_size": 16,
    },
}

# data helpers -- same loading logic as part3_modeling.py
def load_labeled():
    """Load the 3 labeled images into a dict of dataframes."""
    dfs = {}
    for name, path in LABELED_FILES.items():
        arr = np.load(path)["arr_0"]
        df = pd.DataFrame(arr, columns=COL)
        df["image"] = name  # need this for grouping in CV
        dfs[name] = df
    return dfs

def build_features(dfs, feature_set="C"):
    """Assemble X, y, groups, pixel coords from the labeled dataframes.
    Merges AE embeddings for feature sets B and C."""
    parts = []
    for name, df in dfs.items():
        # only keep labeled pixels (1 = cloud, -1 = clear)
        lab = df[df["label"].isin([1, -1])].copy()
        if feature_set in ("B", "C"):
            ae_sz = 8 if feature_set == "B" else 16
            ae_path = f"../data/{name}_ae{ae_sz}.csv"
            if os.path.exists(ae_path):
                lab = lab.merge(pd.read_csv(ae_path), on=["y", "x"], how="left")
        parts.append(lab)
    combined = pd.concat(parts, ignore_index=True)

    # figure out which columns go into X
    feat_cols = list(BASE_FEAT)
    if feature_set == "B":
        feat_cols += [c for c in combined.columns if c.startswith("ae")][:8]
    elif feature_set == "C":
        feat_cols += [c for c in combined.columns if c.startswith("ae")][:16]
    feat_cols = [c for c in feat_cols if c in combined.columns]  # safety

    X = combined[feat_cols].values
    y = (combined["label"].values > 0).astype(int)  # 0/1 for sklearn
    groups = combined["image"].values
    coords = combined[["y", "x"]].values  # keep coords for spatial error maps
    return X, y, groups, coords, feat_cols, combined

def _topk_idx(model, k, p):
    """Get indices of top-k features by importance.
    Falls back to first k indices if the model has no importance attr."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        # SVM does not have importances, just take the first k
        return np.arange(min(k, p))
    return np.sort(np.argsort(imp)[::-1][:k])

def extract_patches_single_image(image_data, patch_size=9):
    """9x9 patches from a single image, normalized per-image."""
    ys = image_data[:, 0].astype(int)
    xs = image_data[:, 1].astype(int)
    features = image_data[:, 2:10]  # 8 base channels

    n_channels = features.shape[1]
    pad = patch_size // 2

    miny, maxy = ys.min(), ys.max()
    minx, maxx = xs.min(), xs.max()
    height = int(maxy - miny + 1)
    width = int(maxx - minx + 1)

    grid = np.zeros((n_channels, height, width))
    y_rel = ys - miny
    x_rel = xs - minx
    for c in range(n_channels):
        grid[c, y_rel, x_rel] = features[:, c]

    # per-image normalization
    for c in range(n_channels):
        mu = grid[c].mean()
        std = grid[c].std()
        if std > 0:
            grid[c] = (grid[c] - mu) / std

    grid_padded = np.pad(
        grid,
        ((0, 0), (pad, pad), (pad, pad)),
        mode="reflect",
    )

    patches = []
    for y, x in zip(ys, xs):
        yi = int(y - miny + pad)
        xi = int(x - minx + pad)
        patch = grid_padded[:, yi - pad : yi + pad + 1, xi - pad : xi + pad + 1]
        patches.append(patch.astype(np.float32))

    return patches, ys.astype(float), xs.astype(float)

def generate_ae_embeddings_for_array(image_arr, emb_size):
    """Run the trained autoencoder on a raw image array and return
    a dataframe with (y, x, ae0, ae1, ...) columns.
    Used for unlabeled images that do not have pre-computed embeddings."""
    from autoencoder import Autoencoder

    cfg = EMB_CONFIG[emb_size]
    ckpt_path = cfg["checkpoint"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for AE{emb_size}: {ckpt_path}")

    model = Autoencoder(
        patch_size=cfg["patch_size"],
        n_input_channels=cfg["n_input_channels"],
        embedding_size=cfg["embedding_size"],
    )
    map_location = None if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    patches, ys, xs = extract_patches_single_image(image_arr, cfg["patch_size"])
    all_emb = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.tensor(np.array(patches[start : start + batch_size]))
            emb = model.embed(batch).detach().cpu().numpy()
            all_emb.append(emb)
    all_emb = np.concatenate(all_emb, axis=0)

    ae_cols = [f"ae{j}" for j in range(all_emb.shape[1])]
    ae_df = pd.DataFrame(all_emb, columns=ae_cols)
    ae_df.insert(0, "y", ys)
    ae_df.insert(1, "x", xs)
    return ae_df

# model factories -- convenience wrappers so we do not repeat params everywhere
def gbt(seed=15, **kw):
    defaults = dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=0.8, random_state=seed)
    defaults.update(kw)
    return GradientBoostingClassifier(**defaults)

def all_models(seed=15):
    return {
        "LR_L2": LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1),
        "GBT": gbt(seed),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed),
    }

# ---- step 1: CV predictions + convergence ----
def analysis_cv_predictions(X, y, groups, coords, feat_cols, out, seed, top_k=5):
    """Run image-level CV with our champion config (GBT top-k).
    Also records convergence curves (log-loss at each boosting iteration)
    which we use in the report diagnostic figure."""
    print("\n[1/9] Pixel-level CV predictions + convergence...")
    images = sorted(set(groups))
    rows, conv = [], []

    for fi, test_img in enumerate(images):
        tr = groups != test_img
        te = groups == test_img
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        cte = coords[te]

        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

        # two-step process: fit on all features to rank, then refit on top-k
        m_full = gbt(seed)
        m_full.fit(Xtr_s, ytr)
        idx = _topk_idx(m_full, top_k, Xtr_s.shape[1])
        sel = [feat_cols[i] for i in idx]

        m = gbt(seed)
        m.fit(Xtr_s[:, idx], ytr)

        # track convergence: log-loss at each boosting round for train and test
        # only record every 5th iteration to keep the CSV manageable
        for stage_set, Xs, ys, label in [
            ("train", Xtr_s[:, idx], ytr, "train"),
            ("test", Xte_s[:, idx], yte, "test"),
        ]:
            for i, yp in enumerate(m.staged_predict_proba(Xs)):
                if i % 5 == 0 or i == 199:
                    p = np.clip(yp[:, 1], 1e-10, 1 - 1e-10)
                    dev = -np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p))
                    conv.append(dict(fold=fi, test_image=test_img,
                                     n_iter=i + 1, split=label, logloss=dev))

        yp = m.predict_proba(Xte_s[:, idx])[:, 1]
        for j in range(len(yte)):
            rows.append(dict(fold=fi, image=test_img,
                             y_coord=cte[j, 0], x_coord=cte[j, 1],
                             true_label=yte[j], pred_prob=yp[j],
                             pred_label=int(yp[j] >= 0.5)))
        auc = roc_auc_score(yte, yp)
        print(f"  Fold {fi} (test={test_img}) top-{top_k} feats={sel}  ROC-AUC={auc:.4f}")

    pred_df = pd.DataFrame(rows)
    conv_df = pd.DataFrame(conv)
    pred_df.to_csv(os.path.join(out, "fold_predictions.csv"), index=False)
    conv_df.to_csv(os.path.join(out, "convergence.csv"), index=False)
    return pred_df, conv_df

# ---- step 2: confusion matrices ----
def analysis_confusion(pred_df, out):
    """Compute confusion matrices at several thresholds per fold."""
    print("\n[2/9] Confusion matrices...")
    rows = []
    for fi in pred_df["fold"].unique():
        sub = pred_df[pred_df["fold"] == fi]
        yt, yp = sub["true_label"].values, sub["pred_prob"].values
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            cm = confusion_matrix(yt, (yp >= thr).astype(int), labels=[0, 1])
            rows.append(dict(fold=fi, image=sub["image"].iloc[0], threshold=thr,
                             TN=cm[0, 0], FP=cm[0, 1], FN=cm[1, 0], TP=cm[1, 1]))
    cm_df = pd.DataFrame(rows)
    cm_df.to_csv(os.path.join(out, "confusion_matrices.csv"), index=False)
    return cm_df

# ---- step 3: threshold sweep ----
def analysis_threshold(pred_df, out):
    """Sweep the classification threshold from 0.05 to 0.95 and record
    precision, recall, F1, balanced accuracy at each point."""
    print("\n[3/9] Threshold sensitivity sweep...")
    rows = []
    thresholds = np.arange(0.05, 0.96, 0.05)
    for thr in thresholds:
        yt = pred_df["true_label"].values
        yp = (pred_df["pred_prob"].values >= thr).astype(int)
        # manual confusion counts (faster than calling sklearn each time)
        tp = ((yp == 1) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        ba = (rec + spec) / 2
        rows.append(dict(threshold=round(thr, 2), precision=prec, recall=rec,
                         specificity=spec, F1=f1, balanced_acc=ba))
    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(os.path.join(out, "threshold_sweep.csv"), index=False)
    return thr_df

# ---- step 4: perturbation ----
def analysis_perturbation(X, y, groups, feat_cols, out, seed, top_k=5):
    """For each selected feature, add Gaussian noise at several levels
    and measure how much AUC drops. Tells us which features the model
    relies on most."""
    print("\n[4/9] Feature perturbation analysis...")
    images = sorted(set(groups))
    noise_levels = [0.5, 1.0, 2.0]  # std dev of added noise
    rows = []

    for fi, test_img in enumerate(images):
        tr, te = groups != test_img, groups == test_img
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr])
        Xte_s = sc.transform(X[te])
        ytr, yte = y[tr], y[te]

        m_full = gbt(seed); m_full.fit(Xtr_s, ytr)
        idx = _topk_idx(m_full, top_k, Xtr_s.shape[1])
        m = gbt(seed); m.fit(Xtr_s[:, idx], ytr)

        base_prob = m.predict_proba(Xte_s[:, idx])[:, 1]
        base_auc = roc_auc_score(yte, base_prob)

        rng = np.random.RandomState(seed)
        sel_names = [feat_cols[i] for i in idx]
        for feat_i, feat_name in enumerate(sel_names):
            for sigma in noise_levels:
                Xp = Xte_s[:, idx].copy()
                Xp[:, feat_i] += rng.normal(0, sigma, size=len(Xp))
                prob_p = m.predict_proba(Xp)[:, 1]
                auc_p = roc_auc_score(yte, prob_p)
                rows.append(dict(fold=fi, feature=feat_name, noise_std=sigma,
                                 auc_base=base_auc, auc_perturbed=auc_p,
                                 auc_drop=base_auc - auc_p))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "feature_perturbation.csv"), index=False)
    return df

# ---- step 5: calibration ----
def analysis_calibration(pred_df, out):
    """Compute calibration curves (reliability diagrams) and Brier scores.
    These tell us whether the predicted probabilities are trustworthy."""
    print("\n[5/9] Calibration analysis...")
    rows = []
    for fi in pred_df["fold"].unique():
        sub = pred_df[pred_df["fold"] == fi]
        yt, yp = sub["true_label"].values, sub["pred_prob"].values
        # bin predictions into 10 equal-width bins and compare to actual fraction
        frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=10, strategy="uniform")
        brier = brier_score_loss(yt, yp)  # lower is better
        for b in range(len(frac_pos)):
            rows.append(dict(fold=fi, image=sub["image"].iloc[0],
                             bin_mean_pred=mean_pred[b], bin_frac_positive=frac_pos[b],
                             brier_score=brier))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "calibration.csv"), index=False)
    return df

# ---- step 6: hyperparam sweep ----
def analysis_hyperparam(X, y, groups, feat_cols, out, seed, top_k=5):
    """Targeted sensitivity check around our default GBT hyperparameters.
    We vary n_estimators, max_depth, and learning_rate to confirm
    that our defaults are near-optimal."""
    print("\n[6/9] Hyperparameter sensitivity...")
    # each dict is one config to try; our default is the third one
    configs = [
        dict(n_estimators=100, max_depth=3, learning_rate=0.1),
        dict(n_estimators=200, max_depth=3, learning_rate=0.1),
        dict(n_estimators=200, max_depth=5, learning_rate=0.1),   # <-- our default
        dict(n_estimators=200, max_depth=7, learning_rate=0.1),
        dict(n_estimators=200, max_depth=5, learning_rate=0.05),
        dict(n_estimators=300, max_depth=5, learning_rate=0.05),
        dict(n_estimators=200, max_depth=5, learning_rate=0.2),
        dict(n_estimators=400, max_depth=5, learning_rate=0.05),
    ]
    images = sorted(set(groups))
    rows = []

    for ci, cfg in enumerate(configs):
        label = f"ne={cfg['n_estimators']}_md={cfg['max_depth']}_lr={cfg['learning_rate']}"
        fold_aucs = []
        for fi, test_img in enumerate(images):
            tr, te = groups != test_img, groups == test_img
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(X[tr])
            Xte_s = sc.transform(X[te])
            ytr, yte = y[tr], y[te]

            # full → topk
            m_full = gbt(seed, **cfg); m_full.fit(Xtr_s, ytr)
            idx = _topk_idx(m_full, top_k, Xtr_s.shape[1])
            m = gbt(seed, **cfg); m.fit(Xtr_s[:, idx], ytr)
            auc = roc_auc_score(yte, m.predict_proba(Xte_s[:, idx])[:, 1])
            fold_aucs.append(auc)
            rows.append(dict(config=label, fold=fi, ROC_AUC=auc, **cfg))

        mean_auc = np.mean(fold_aucs)
        print(f"  [{ci+1}/{len(configs)}] {label:40s}  mean ROC-AUC={mean_auc:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "hyperparam_sensitivity.csv"), index=False)
    return df

# ---- step 7: seed sensitivity ----
def analysis_seed(X, y, groups, feat_cols, out, top_k=5):
    """Rerun the pipeline with different random seeds to check stability."""
    print("\n[7/9] Seed sensitivity...")
    seeds = [15, 42, 123, 456, 789]  # 5 seeds should be enough
    images = sorted(set(groups))
    rows = []

    for s in seeds:
        fold_aucs = []
        for fi, test_img in enumerate(images):
            tr, te = groups != test_img, groups == test_img
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(X[tr])
            Xte_s = sc.transform(X[te])
            ytr, yte = y[tr], y[te]

            m_full = gbt(s); m_full.fit(Xtr_s, ytr)
            idx = _topk_idx(m_full, top_k, Xtr_s.shape[1])
            m = gbt(s); m.fit(Xtr_s[:, idx], ytr)
            auc = roc_auc_score(yte, m.predict_proba(Xte_s[:, idx])[:, 1])
            fold_aucs.append(auc)
            rows.append(dict(seed=s, fold=fi, test_image=test_img, ROC_AUC=auc))

        print(f"  seed={s:>4d}  mean ROC-AUC={np.mean(fold_aucs):.4f} "
              f"(std={np.std(fold_aucs):.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "seed_sensitivity.csv"), index=False)
    return df

# ---- step 8: ensemble ----
def analysis_ensemble(X, y, groups, feat_cols, out, seed, top_k=5, use_svm=True):
    """Compare individual models vs. simple probability-averaging ensembles.
    If the models make correlated errors, ensembling will not help much."""
    print("\n[8/9] Model ensemble analysis...")
    images = sorted(set(groups))
    model_factories = {
        "GBT": lambda: gbt(seed),
        "RF": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                              random_state=seed, n_jobs=-1),
    }
    if use_svm:
        model_factories["SVM"] = lambda: SVC(kernel="rbf", probability=True, random_state=seed)
    rows = []
    svm_max_train = 10000  # faster local runtime; enough for sanity-check ensemble
    fast_max_train = 40000  # when use_svm=False, downsample all models for speed

    for fi, test_img in enumerate(images):
        print(f"  Fold {fi+1}/{len(images)} (test={test_img})...")
        tr, te = groups != test_img, groups == test_img
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr])
        Xte_s = sc.transform(X[te])
        ytr, yte = y[tr], y[te]

        # Fast mode: downsample train set for all ensemble members.
        if (not use_svm) and len(Xtr_s) > fast_max_train:
            idx_fast = np.random.RandomState(seed + fi).choice(len(Xtr_s), fast_max_train, replace=False)
            Xtr_fit = Xtr_s[idx_fast]
            ytr_fit = ytr[idx_fast]
        else:
            Xtr_fit = Xtr_s
            ytr_fit = ytr

        # SVM subsampling
        svm_idx = np.arange(len(Xtr_s))
        if len(Xtr_s) > svm_max_train:
            svm_idx = np.random.RandomState(seed).choice(len(Xtr_s), svm_max_train, replace=False)

        probs = {}
        for mname, factory in model_factories.items():
            m = factory()
            if mname == "SVM":
                m.fit(Xtr_s[svm_idx], ytr[svm_idx])
            else:
                m.fit(Xtr_fit, ytr_fit)
            probs[mname] = m.predict_proba(Xte_s)[:, 1]
            auc = roc_auc_score(yte, probs[mname])
            rows.append(dict(fold=fi, model=mname, ROC_AUC=auc, type="individual"))

        # simple probability averaging across all models
        avg_prob = np.mean(list(probs.values()), axis=0)
        auc_ens = roc_auc_score(yte, avg_prob)
        rows.append(dict(fold=fi, model="Ensemble(avg)", ROC_AUC=auc_ens, type="ensemble"))

        # also try GBT+RF only (no SVM) since those two are fastest
        avg_2 = (probs["GBT"] + probs["RF"]) / 2
        auc_2 = roc_auc_score(yte, avg_2)
        rows.append(dict(fold=fi, model="Ensemble(GBT+RF)", ROC_AUC=auc_2, type="ensemble"))

    df = pd.DataFrame(rows)
    summary = df.groupby(["model", "type"])["ROC_AUC"].agg(["mean", "std"]).round(4)
    print(summary)
    df.to_csv(os.path.join(out, "ensemble_results.csv"), index=False)
    return df

# ---- step 9: unlabeled image predictions ----
def analysis_unlabeled(dfs, feature_set, top_k, out, seed, n_images=5):
    """Train on ALL labeled data (no hold-out) and predict on randomly
    chosen unlabeled images. This is a sanity check -- we cannot compute
    AUC without ground truth, but we can look at spatial coherence."""
    print(f"\n[9/9] Unlabeled predictions ({feature_set} + GBT + top{top_k}, {n_images} images)...")
    os.makedirs(os.path.join(out, "unlabeled"), exist_ok=True)
    os.makedirs(os.path.join(out, "unlabeled_viz"), exist_ok=True)

    # use all 3 labeled images for training (no hold-out this time)
    Xtr, ytr, _, _, feature_cols, _ = build_features(dfs, feature_set)
    sc_full = StandardScaler()
    Xtr_s = sc_full.fit_transform(Xtr)

    # rank features from the full model, then retrain on top-k only
    # (same procedure as in CV, but now using all labeled data)
    m_rank = gbt(seed)
    m_rank.fit(Xtr_s, ytr)
    top_idx = _topk_idx(m_rank, top_k, Xtr_s.shape[1])
    selected_cols = [feature_cols[i] for i in top_idx]

    sc_sel = StandardScaler()
    Xtr_sel_s = sc_sel.fit_transform(Xtr[:, top_idx])
    m_best = gbt(seed)
    m_best.fit(Xtr_sel_s, ytr)
    print(f"  Selected top-{top_k} columns: {selected_cols}")

    # find all npz files that are NOT one of the 3 labeled images
    all_npz = sorted(glob.glob(os.path.join(UNLABELED_DIR, "*.npz")))
    labeled_names = set(LABELED_FILES.keys())
    unlabeled = [p for p in all_npz
                 if os.path.splitext(os.path.basename(p))[0] not in labeled_names]
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(unlabeled), min(n_images, len(unlabeled)), replace=False)
    summary_rows = []

    for ci in chosen:
        path = unlabeled[ci]
        img_name = os.path.splitext(os.path.basename(path))[0]
        arr = np.load(path)["arr_0"]
        # 10 cols if no label, 11 if there is one
        cols_unl = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
        if arr.shape[1] > 10:
            cols_unl.append("label")
        df_u = pd.DataFrame(arr, columns=cols_unl[:arr.shape[1]])

        # need AE embeddings for B/C feature sets
        if feature_set in ("B", "C"):
            emb_size = 8 if feature_set == "B" else 16
            ae_df = generate_ae_embeddings_for_array(arr, emb_size)
            df_u = df_u.merge(ae_df, on=["y", "x"], how="left")

        missing = [c for c in selected_cols if c not in df_u.columns]
        if missing:
            raise ValueError(f"Missing selected columns in {img_name}: {missing}")

        Xu = df_u[selected_cols].values
        Xu_s = sc_sel.transform(Xu)
        prob = m_best.predict_proba(Xu_s)[:, 1]

        out_df = pd.DataFrame({
            "y_coord": df_u["y"].values,
            "x_coord": df_u["x"].values,
            "pred_prob": prob,
            "pred_label": (prob >= 0.5).astype(int),
        })
        out_path = os.path.join(out, "unlabeled", f"{img_name}.csv")
        out_df.to_csv(out_path, index=False)
        cloud_pct = out_df["pred_label"].mean() * 100
        print(f"  {img_name}: {len(out_df)} pixels, {cloud_pct:.1f}% predicted cloud")

        # save a quick visualization
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)

        sc1 = axes[0].scatter(
            out_df["x_coord"], out_df["y_coord"],
            c=out_df["pred_prob"], s=1.0, cmap="viridis", vmin=0, vmax=1, linewidths=0
        )
        axes[0].set_title(f"{img_name}: Predicted P(cloud)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].invert_yaxis()
        cbar = fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label("Probability")

        axes[1].scatter(
            out_df["x_coord"], out_df["y_coord"],
            c=out_df["pred_label"], s=1.0, cmap="coolwarm", vmin=0, vmax=1, linewidths=0
        )
        axes[1].set_title(f"{img_name}: Binary cloud mask (>=0.5)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].invert_yaxis()

        plt.tight_layout()
        fig.savefig(os.path.join(out, "unlabeled_viz", f"{img_name}.png"))
        plt.close(fig)

        summary_rows.append({
            "image": img_name,
            "n_pixels": int(len(out_df)),
            "cloud_pct": float(out_df["pred_label"].mean() * 100),
            "mean_pred_prob": float(out_df["pred_prob"].mean()),
            "std_pred_prob": float(out_df["pred_prob"].std()),
            "uncertain_40_60_pct": float(((out_df["pred_prob"] >= 0.4) & (out_df["pred_prob"] <= 0.6)).mean() * 100),
            "high_conf_cloud_pct": float((out_df["pred_prob"] >= 0.9).mean() * 100),
            "high_conf_clear_pct": float((out_df["pred_prob"] <= 0.1).mean() * 100),
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("cloud_pct")
        summary_df.to_csv(os.path.join(out, "unlabeled_summary.csv"), index=False)
        print(f"  Saved unlabeled summary to {os.path.join(out, 'unlabeled_summary.csv')}")

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", default="C", choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument(
        "--mode",
        default="quick",
        choices=["quick", "full"],
        help="quick: core diagnostics only; full: run all 9 analyses",
    )
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--n_unlabeled", type=int, default=5)
    ap.add_argument(
        "--start_step",
        type=int,
        default=1,
        help="Start step index (1-9). Useful for resume runs.",
    )
    ap.add_argument(
        "--end_step",
        type=int,
        default=9,
        help="End step index (1-9). Useful for partial runs.",
    )
    ap.add_argument(
        "--skip_unlabeled",
        action="store_true",
        help="Skip unlabeled-image inference step",
    )
    ap.add_argument(
        "--ensemble_fast",
        action="store_true",
        help="Step 8 fast mode: skip SVM and run only GBT/RF ensembles.",
    )
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = f"part3_posthoc_{args.feature_set}_seed{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)

    if not (1 <= args.start_step <= 9 and 1 <= args.end_step <= 9 and args.start_step <= args.end_step):
        raise ValueError("start_step/end_step must satisfy 1 <= start_step <= end_step <= 9")

    print("=" * 60)
    print("Part 3 — Post-Hoc Analysis Pipeline")
    print(f"  Feature set : {args.feature_set}")
    print(f"  Seed        : {args.seed}")
    print(f"  Top-k       : {args.top_k}")
    print(f"  Mode        : {args.mode}")
    print(f"  Ensemble    : {'fast (no SVM)' if args.ensemble_fast else 'full (with SVM)'}")
    print(f"  Step range  : {args.start_step}..{args.end_step}")
    print(f"  Output      : {args.output_dir}/")
    print("=" * 60)

    t0 = time.time()

    dfs = load_labeled()
    X, y, groups, coords, feat_cols, combined = build_features(dfs, args.feature_set)

    # if we are starting from a later step, load existing predictions from CSV
    pred_df = None
    pred_path = os.path.join(args.output_dir, "fold_predictions.csv")

    def step_enabled(step_no):
        return args.start_step <= step_no <= args.end_step

    if step_enabled(1):
        pred_df, _ = analysis_cv_predictions(
            X, y, groups, coords, feat_cols, args.output_dir, args.seed, args.top_k
        )
    elif any(step_enabled(s) for s in (2, 3, 5)):
        if os.path.exists(pred_path):
            pred_df = pd.read_csv(pred_path)
            print("\nUsing existing fold_predictions.csv for steps 2/3/5.")
        else:
            raise FileNotFoundError(
                "fold_predictions.csv not found in output_dir. "
                "Run step 1 first or set start_step=1."
            )

    if step_enabled(2):
        analysis_confusion(pred_df, args.output_dir)
    if step_enabled(3):
        analysis_threshold(pred_df, args.output_dir)
    if step_enabled(4):
        if args.mode == "full":
            analysis_perturbation(X, y, groups, feat_cols, args.output_dir, args.seed, args.top_k)
        else:
            print("\n[quick mode] Step 4 skipped (perturbation). Use --mode full to run it.")
    if step_enabled(5):
        analysis_calibration(pred_df, args.output_dir)
    if step_enabled(6):
        if args.mode == "full":
            analysis_hyperparam(X, y, groups, feat_cols, args.output_dir, args.seed, args.top_k)
        else:
            print("\n[quick mode] Step 6 skipped (hyperparam). Use --mode full to run it.")
    if step_enabled(7):
        if args.mode == "full":
            analysis_seed(X, y, groups, feat_cols, args.output_dir, args.top_k)
        else:
            print("\n[quick mode] Step 7 skipped (seed sensitivity). Use --mode full to run it.")
    if step_enabled(8):
        if args.mode == "full":
            analysis_ensemble(
                X, y, groups, feat_cols, args.output_dir, args.seed, args.top_k,
                use_svm=(not args.ensemble_fast)
            )
        else:
            print("\n[quick mode] Step 8 skipped (ensemble). Use --mode full to run it.")
    if step_enabled(9):
        if not args.skip_unlabeled:
            # fewer images in quick mode
            n_u = min(args.n_unlabeled, 2) if args.mode == "quick" else args.n_unlabeled
            analysis_unlabeled(dfs, args.feature_set, args.top_k, args.output_dir, args.seed, n_u)
        else:
            print("\nSkipping unlabeled-image inference (--skip_unlabeled).")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All analyses complete in {elapsed:.0f}s.  Results → {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
