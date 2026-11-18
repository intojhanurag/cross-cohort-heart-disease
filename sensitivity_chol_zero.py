"""Sensitivity check: cholesterol is recorded as 0 for all 123 Switzerland
records and 49 of 200 Long Beach VA records. The main analysis keeps these
values as they are in the UCI files. This script re-scores the frozen portable
logistic model with those zeros treated as missing (median-imputed).

Run:  python sensitivity_chol_zero.py
"""
import numpy as np
import pandas as pd

import run_analysis as ra

data = ra.load_cohorts()
dev = data[data.cohort == ra.DEV]
ext = data[data.cohort != ra.DEV].reset_index(drop=True)
pipe = ra.make_pipeline("logistic", "Portable").fit(ra.features(dev, "Portable"), dev.y)

print("Records with cholesterol = 0:\n",
      data.groupby("cohort", sort=False)["chol"].agg(zero=lambda s: int((s == 0).sum()), records="size"), "\n")

rows = []
variants = {"zeros kept (main analysis)": ext,
            "zeros treated as missing": ext.assign(chol=ext.chol.replace(0, np.nan))}
for label, frame in variants.items():
    p = pipe.predict_proba(ra.features(frame, "Portable"))[:, 1]
    for cohort in ra.EXTERNAL + ["External pooled"]:
        m = np.ones(len(frame), bool) if cohort == "External pooled" else (frame.cohort == cohort).to_numpy()
        r = ra.point_metrics(frame.y.to_numpy()[m], p[m])
        rows.append({"cholesterol": label, "cohort": cohort, **{k: r[k] for k in
                     ("auc", "brier", "sensitivity", "specificity", "cal_intercept", "cal_slope")}})
out = pd.DataFrame(rows).round(3)
ra.OUT.mkdir(exist_ok=True)
out.to_csv(ra.OUT / "sensitivity_chol_zero.csv", index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
