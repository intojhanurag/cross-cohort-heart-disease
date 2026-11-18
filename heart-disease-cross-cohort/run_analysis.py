"""Cross-cohort heart disease detection: full reproducible analysis.

Develops L2 logistic regression and random forest models on the UCI Cleveland
cohort and evaluates them, without refitting, on the Hungary, Switzerland and
Long Beach VA cohorts.

Run:  python run_analysis.py            (full run, 2,000 bootstrap resamples)
      python run_analysis.py --quick    (200 resamples, for a fast live demo)

Everything written to outputs/ is regenerated from data/raw/ on every run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Configuration (fixed before evaluation; nothing here is tuned on the data)
# --------------------------------------------------------------------------
SEED = 20260904
N_SPLITS, N_REPEATS = 5, 10
N_BOOTSTRAP = 2000
THRESHOLD = 0.50

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
COHORTS = {  # label -> file
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "Long Beach VA": "processed.va.data",
}
DEV = "Cleveland"
EXTERNAL = ["Hungary", "Switzerland", "Long Beach VA"]

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang"]
DROPPED = ["slope", "ca", "thal"]          # largely missing outside Cleveland
FEATURE_SETS = {
    "Portable": {"num": CONTINUOUS, "cat": CATEGORICAL},                       # 10
    "Full": {"num": CONTINUOUS, "cat": CATEGORICAL + ["slope", "ca", "thal"]},     # 13
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_cohorts() -> pd.DataFrame:
    frames = []
    for name, fname in COHORTS.items():
        df = pd.read_csv(RAW / fname, names=COLUMNS, na_values="?")
        df.insert(0, "cohort", name)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["y"] = (data["num"] > 0).astype(int)   # 0 = <50% narrowing, 1-4 = disease
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_pipeline(model: str, feature_set: str) -> Pipeline:
    fs = FEATURE_SETS[feature_set]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), fs["num"]),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   drop="if_binary",
                                                   sparse_output=False))]), fs["cat"]),
    ])
    if model == "logistic":
        # L2 penalty, C = 1. liblinear also shrinks the intercept slightly.
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=5000)
    else:
        clf = RandomForestClassifier(n_estimators=500, max_depth=5,
                                     min_samples_leaf=5, max_features="sqrt",
                                     random_state=SEED, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


MODELS = {  # display name -> (model, feature set)
    "Full logistic": ("logistic", "Full"),
    "Portable logistic": ("logistic", "Portable"),
    "Full RF": ("rf", "Full"),
    "Portable RF": ("rf", "Portable"),
}


def features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    fs = FEATURE_SETS[feature_set]
    return df[fs["num"] + fs["cat"]]


def cross_validated_probabilities(df: pd.DataFrame, model: str, feature_set: str) -> np.ndarray:
    """Repeated stratified 5-fold CV; the 10 out-of-fold probabilities per
    record are averaged. All preprocessing is fitted inside each training fold."""
    X, y = features(df, feature_set), df["y"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    total = np.zeros(len(df))
    for train, test in rskf.split(X, y):
        pipe = make_pipeline(model, feature_set).fit(X.iloc[train], y[train])
        total[test] += pipe.predict_proba(X.iloc[test])[:, 1]
    return total / N_REPEATS


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def auc(y: np.ndarray, p: np.ndarray) -> float:
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """Logistic recalibration  y ~ a + b * logit(p).

    Returns (a, b, citl): calibration intercept a (ideal 0) and slope b
    (ideal 1) from the joint fit, plus calibration-in-the-large, i.e. the
    intercept when the slope is fixed at 1 (ideal 0)."""
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))

    def nll(beta, slope_free):
        z = beta[0] + (beta[1] if slope_free else 1.0) * lp
        return np.sum(np.logaddexp(0, z) - y * z)

    intercept, slope = minimize(nll, [0.0, 1.0], args=(True,), method="BFGS").x
    citl = minimize(nll, [0.0], args=(False,), method="BFGS").x[0]
    return float(intercept), float(slope), float(citl)


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= THRESHOLD
    tp, fn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 1))
    tn, fp = np.sum(~pred & (y == 0)), np.sum(pred & (y == 0))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    intercept, slope, citl = calibration(y, p)
    return {
        "n": len(y), "disease": int(y.sum()),
        "auc": auc(y, p), "average_precision": average_precision_score(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
        "accuracy": (tp + tn) / len(y), "balanced_accuracy": np.nanmean([sens, spec]),
        "cal_intercept": intercept, "cal_slope": slope, "cal_in_the_large": citl,
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    """Participant-level percentile 95% CIs."""
    stats = {k: [] for k in ("auc", "brier", "sensitivity", "specificity")}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():        # resample without both classes
            continue
        pred = pb >= THRESHOLD
        stats["auc"].append(auc(yb, pb))
        stats["brier"].append(np.mean((pb - yb) ** 2))
        stats["sensitivity"].append(pred[yb == 1].mean())
        stats["specificity"].append((~pred[yb == 0]).mean())
    out = {}
    for k, v in stats.items():
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
    return out


def evaluate(y, p, n_boot, rng) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {**point_metrics(y, p), **bootstrap_ci(y, p, n_boot, rng)}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
PALETTE = {"Cleveland": "#4c72b0", "Hungary": "#dd8452",
           "Switzerland": "#55a868", "Long Beach VA": "#c44e52"}


def fig_missingness(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    miss = (data.groupby("cohort", sort=False)[COLUMNS[:-1]]
            .agg(lambda s: s.isna().mean() * 100))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(miss.to_numpy(), cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(miss.shape[1]), miss.columns)
    ax.set_yticks(range(miss.shape[0]), miss.index)
    for i in range(miss.shape[0]):
        for j in range(miss.shape[1]):
            v = miss.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 60 else "black", fontsize=9)
    fig.colorbar(im, label="Missing (%)")
    ax.set_xlabel("Predictor")
    ax.set_title("Feature missingness differs substantially across cohorts")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return miss


def fig_roc(preds: pd.DataFrame, path: Path, model: str = "Portable logistic"):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for cohort in COHORTS:
        d = preds[preds.cohort == cohort]
        fpr, tpr, _ = roc_curve(d.y, d[model])
        ax.plot(fpr, tpr, color=PALETTE[cohort], lw=1.8,
                label=f"{cohort} (AUC {auc(d.y.to_numpy(), d[model].to_numpy()):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate",
           title=f"Cross-cohort discrimination: {model.lower()} model")
    ax.grid(alpha=.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_calibration(preds: pd.DataFrame, path: Path, model: str = "Portable logistic"):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for cohort in COHORTS:
        d = preds[preds.cohort == cohort]
        bins = pd.qcut(d[model], 5, duplicates="drop")
        g = d.groupby(bins, observed=True).agg(pred=(model, "mean"), obs=("y", "mean"))
        ax.plot(g.pred, g.obs, "o-", color=PALETTE[cohort], lw=1.8, ms=4, label=cohort)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Ideal")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed disease proportion",
           title="Calibration shifts across clinical cohorts")
    ax.grid(alpha=.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_coefficients(coefs: pd.Series, path: Path):
    c = coefs.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(c.index, c.values, color=np.where(c.values > 0, "#b5394b", "#2f6f9f"))
    ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="Standardized log-odds coefficient",
           title="Most influential terms in the portable logistic model")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="200 bootstrap resamples")
    args = ap.parse_args()
    n_boot = 200 if args.quick else N_BOOTSTRAP
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 200, "display.max_columns", 30)

    data = load_cohorts()
    dev = data[data.cohort == DEV].reset_index(drop=True)
    ext = data[data.cohort != DEV].reset_index(drop=True)

    # ---- Table 2: cohort composition -------------------------------------
    cohort_tbl = data.groupby("cohort", sort=False).agg(
        records=("y", "size"), disease=("y", "sum"),
        prevalence_pct=("y", lambda s: 100 * s.mean()),
        median_age=("age", "median"),
        female_n=("sex", lambda s: int((s == 0).sum())),
        female_pct=("sex", lambda s: 100 * (s == 0).mean())).round(1)
    print("\n=== Table 2. Cohort composition ===\n", cohort_tbl)

    missing_tbl = fig_missingness(data, OUT / "figures" / "fig1_missingness.png").round(1)
    print("\n=== Figure 1 data. Missing values per predictor (%) ===\n", missing_tbl)

    # ---- Internal validation (repeated CV) and frozen refit --------------
    preds = pd.concat([dev, ext], ignore_index=True)[["cohort", "sex", "y"]].copy()
    internal_rows, external_rows, fitted = [], [], {}
    for name, (model, fs) in MODELS.items():
        print(f"\n[{time.time() - t0:5.1f}s] {name}: {N_REPEATS}x repeated {N_SPLITS}-fold CV on Cleveland ...")
        p_cv = cross_validated_probabilities(dev, model, fs)
        internal_rows.append({"model": name, **evaluate(dev.y, p_cv, n_boot, rng)})

        pipe = make_pipeline(model, fs).fit(features(dev, fs), dev.y)   # refit once, then frozen
        fitted[name] = pipe
        joblib.dump(pipe, OUT / "models" / f"{name.lower().replace(' ', '_')}.joblib")
        p_ext = pipe.predict_proba(features(ext, fs))[:, 1]
        preds[name] = np.concatenate([p_cv, p_ext])     # Cleveland rows = out-of-fold

        for cohort in EXTERNAL + ["External pooled"]:
            m = np.ones(len(ext), bool) if cohort == "External pooled" else (ext.cohort == cohort).to_numpy()
            external_rows.append({"model": name, "cohort": cohort,
                                  **evaluate(ext.y[m], p_ext[m], n_boot, rng)})

    internal = pd.DataFrame(internal_rows).set_index("model")
    external = pd.DataFrame(external_rows).set_index(["model", "cohort"])
    show = ["auc", "auc_lo", "auc_hi", "brier", "sensitivity", "specificity", "cal_intercept", "cal_slope"]
    print("\n=== Table 3. Repeated cross-validation performance in Cleveland ===\n", internal[show].round(3))
    print("\n=== Table 4. External performance (frozen models, no refitting) ===\n",
          external[["n"] + show].round(3))

    # ---- Table 5: exploratory sex-stratified performance -----------------
    sex_rows = []
    for label, mask in [("Cleveland", preds.cohort == DEV), ("External pooled", preds.cohort != DEV)]:
        for sex_name, code in [("Female", 0), ("Male", 1)]:
            d = preds[mask & (preds.sex == code)]
            sex_rows.append({"evaluation": label, "subgroup": sex_name,
                             **evaluate(d.y, d["Portable logistic"], n_boot, rng)})
    sex_tbl = pd.DataFrame(sex_rows).set_index(["evaluation", "subgroup"])
    print("\n=== Table 5. Sex-stratified performance, portable logistic (exploratory) ===\n",
          sex_tbl[["n", "disease", "auc", "auc_lo", "auc_hi", "sensitivity", "specificity"]].round(3))

    # ---- Coefficients of the portable logistic model ---------------------
    pl = fitted["Portable logistic"]
    names = [n.split("__", 1)[1] for n in pl.named_steps["pre"].get_feature_names_out()]
    coefs = pd.Series(pl.named_steps["clf"].coef_[0], index=names, name="coefficient")
    print("\n=== Figure 4 data. Portable logistic coefficients (standardized log-odds) ===\n",
          coefs.sort_values(ascending=False).round(3).to_string())

    # ---- Figures and files ------------------------------------------------
    fig_roc(preds, OUT / "figures" / "fig2_roc_portable_logistic.png")
    fig_calibration(preds, OUT / "figures" / "fig3_calibration_portable_logistic.png")
    fig_coefficients(coefs, OUT / "figures" / "fig4_coefficients_portable_logistic.png")

    cohort_tbl.to_csv(OUT / "table2_cohorts.csv")
    missing_tbl.to_csv(OUT / "fig1_missingness.csv")
    internal.to_csv(OUT / "table3_internal_cv.csv")
    external.to_csv(OUT / "table4_external.csv")
    sex_tbl.to_csv(OUT / "table5_sex_subgroups.csv")
    coefs.to_csv(OUT / "fig4_coefficients.csv")
    preds.to_csv(OUT / "predictions.csv", index_label="record")
    with pd.ExcelWriter(OUT / "results.xlsx") as xl:
        cohort_tbl.to_excel(xl, sheet_name="Table2_cohorts")
        missing_tbl.to_excel(xl, sheet_name="Fig1_missingness")
        internal.to_excel(xl, sheet_name="Table3_internal_cv")
        external.to_excel(xl, sheet_name="Table4_external")
        sex_tbl.to_excel(xl, sheet_name="Table5_sex")
        coefs.to_frame().to_excel(xl, sheet_name="Fig4_coefficients")

    manifest = {
        "seed": SEED, "cv": f"{N_REPEATS}x repeated stratified {N_SPLITS}-fold",
        "bootstrap_resamples": n_boot, "threshold": THRESHOLD,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "data_sha256": {f: sha256(RAW / f) for f in COHORTS.values()},
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    from build_report import build_report
    build_report()
    print(f"\nDone in {time.time() - t0:.1f}s. Tables, figures, models and report.html are in {OUT}")


if __name__ == "__main__":
    main()
