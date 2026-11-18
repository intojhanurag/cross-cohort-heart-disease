"""Compares the numbers reported in the paper with the numbers that
run_analysis.py just produced in outputs/.

Run after run_analysis.py:   python verify_paper_numbers.py

Point estimates are deterministic (fixed seed) and should match to the third
decimal. Confidence-interval bounds come from bootstrap resampling, so they
depend on the random-number stream and are only expected to agree closely.
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
POINT_TOL, CI_TOL = 0.0015, 0.03

internal = pd.read_csv(OUT / "table3_internal_cv.csv").set_index("model")
external = pd.read_csv(OUT / "table4_external.csv").set_index(["model", "cohort"])
sex = pd.read_csv(OUT / "table5_sex_subgroups.csv").set_index(["evaluation", "subgroup"])
cohorts = pd.read_csv(OUT / "table2_cohorts.csv").set_index("cohort")

checks = []  # (where, what, paper value, reproduced value, kind)


def add(where, what, paper, ours, kind="point"):
    checks.append((where, what, paper, float(ours), kind))


def add_row(where, row, auc, lo, hi, brier=None, sens=None, spec=None, slope=None, intercept=None):
    add(where, "AUC", auc, row["auc"])
    add(where, "AUC CI low", lo, row["auc_lo"], "ci")
    add(where, "AUC CI high", hi, row["auc_hi"], "ci")
    for name, val, col in [("Brier", brier, "brier"), ("Sensitivity", sens, "sensitivity"),
                           ("Specificity", spec, "specificity"), ("Cal. slope", slope, "cal_slope"),
                           ("Cal. intercept", intercept, "cal_intercept")]:
        if val is not None:
            add(where, name, val, row[col])


# Table 2 --------------------------------------------------------------
for name, n, d, prev, age, f in [("Cleveland", 303, 139, 45.9, 56, 97), ("Hungary", 294, 106, 36.1, 49, 81),
                                 ("Switzerland", 123, 115, 93.5, 56, 10), ("Long Beach VA", 200, 149, 74.5, 60, 6)]:
    r = cohorts.loc[name]
    for what, paper, col in [("records", n, "records"), ("disease", d, "disease"), ("prevalence %", prev, "prevalence_pct"),
                             ("median age", age, "median_age"), ("female n", f, "female_n")]:
        add(f"T2 {name}", what, paper, r[col])

# Table 3 --------------------------------------------------------------
add_row("T3 Full logistic", internal.loc["Full logistic"], .913, .878, .944, .114, .806, .902, .935)
add_row("T3 Portable logistic", internal.loc["Portable logistic"], .875, .834, .911, .143, .755, .841, .972, -.046)
add_row("T3 Full RF", internal.loc["Full RF"], .908, .875, .938, .125, .791, .872, 1.579)
add_row("T3 Portable RF", internal.loc["Portable RF"], .867, .824, .905, .151, .719, .841, 1.507)

# Table 4 and section 4.2 / 4.3 ------------------------------------------
L, R = "Portable logistic", "Portable RF"
add_row("T4 Logistic Hungary", external.loc[(L, "Hungary")], .877, .833, .918, .135, .783, .814)
add_row("T4 Logistic Switzerland", external.loc[(L, "Switzerland")], .766, .624, .906, .267, .609, .625, intercept=2.968)
add_row("T4 Logistic Long Beach VA", external.loc[(L, "Long Beach VA")], .722, .643, .799, .188, .785, .529, slope=.488)
add_row("T4 RF Hungary", external.loc[(R, "Hungary")], .890, .849, .929, .131, .717, .894)
add_row("T4 RF Switzerland", external.loc[(R, "Switzerland")], .767, .544, .981, .198, .626, .625)
add_row("T4 RF Long Beach VA", external.loc[(R, "Long Beach VA")], .749, .672, .822, .186, .779, .549)
add_row("4.2 Logistic pooled external", external.loc[(L, "External pooled")], .815, .782, .847, .179, .730, .749)

# Table 5 --------------------------------------------------------------
for key, n, d, auc, lo, hi, sens, spec in [
        (("Cleveland", "Female"), 97, 25, .891, .804, .962, .720, .931),
        (("Cleveland", "Male"), 206, 114, .846, .790, .897, .763, .772),
        (("External pooled", "Female"), 97, 25, .754, .630, .866, .400, .917),
        (("External pooled", "Male"), 520, 345, .791, .750, .829, .754, .680)]:
    where = f"T5 {key[0]} {key[1]}"
    add(where, "records", n, sex.loc[key, "n"])
    add(where, "disease", d, sex.loc[key, "disease"])
    add_row(where, sex.loc[key], auc, lo, hi, sens=sens, spec=spec)

# Report ---------------------------------------------------------------
rows, n_fail = [], 0
for where, what, paper, ours, kind in checks:
    diff = ours - paper
    ok = abs(diff) <= (CI_TOL if kind == "ci" else POINT_TOL)
    n_fail += not ok
    status = ("ok (bootstrap)" if kind == "ci" else "MATCH") if ok else "DIFFERS"
    rows.append({"item": where, "metric": what, "paper": paper, "reproduced": round(ours, 3),
                 "diff": round(diff, 3), "status": status})
table = pd.DataFrame(rows)
table.to_csv(OUT / "paper_vs_reproduced.csv", index=False)
pd.set_option("display.width", 200, "display.max_rows", 500)
print(table.to_string(index=False))
n_point = sum(k == "point" for *_, k in checks)
print(f"\n{len(checks)} values checked: {n_point} point estimates (tolerance {POINT_TOL}), "
      f"{len(checks) - n_point} bootstrap CI bounds (tolerance {CI_TOL}).")
print("ALL PAPER NUMBERS REPRODUCED" if n_fail == 0 else f"{n_fail} VALUE(S) DIFFER FROM THE PAPER")
raise SystemExit(1 if n_fail else 0)
