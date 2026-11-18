# Cross-Cohort Heart Disease Detection Using Interpretable and Calibration-Aware Machine Learning

Code and results for I3CSET-2026 Paper ID 473.

L2-regularised logistic regression and random forest models are developed on the
UCI **Cleveland** cohort (303 records) and then tested, **without refitting**, on
617 records from **Hungary**, **Switzerland** and **Long Beach VA**. The study
reports discrimination (AUC), calibration (Brier score, intercept, slope) and an
exploratory comparison by sex.

> Research prototype on a 1988 benchmark dataset. Not a diagnostic tool.

## Run it

```bash
pip install -r requirements.txt
```

```bash
python run_analysis.py
```

```bash
python verify_paper_numbers.py
```

```bash
streamlit run app.py
```

| Command | What it does | Time |
|---|---|---|
| `python run_analysis.py` | Rebuilds every table and figure from `data/raw/` and writes `outputs/` | about 75 s on 4 cores |
| `python run_analysis.py --quick` | Same, with 200 instead of 2,000 bootstrap resamples (only the confidence intervals change) | slightly faster |
| `python verify_paper_numbers.py` | Compares 121 values printed in the paper with the run you just made | 1 s |
| `streamlit run app.py` | Dashboard: data, results, coefficients, single-patient scoring | live |
| `python sensitivity_chol_zero.py` | Sensitivity check for zero-coded cholesterol (see Limitations) | 1 s |

`outputs/report.html` is a single self-contained page with all tables and
figures; it opens in any browser without Python.

## Method

| Step | Setting |
|---|---|
| Outcome | `num` > 0 (at least 50% narrowing in a major vessel) vs `num` = 0 |
| Portable model | 10 predictors: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak |
| Full model | the 10 above plus slope, ca, thal (largely missing outside Cleveland) |
| Preprocessing | continuous: median imputation + standardisation; categorical: most-frequent imputation + one-hot (unseen categories ignored). Fitted inside each training fold |
| Logistic regression | L2 penalty, C = 1, liblinear solver |
| Random forest | 500 trees, depth 5, minimum leaf 5, sqrt feature sampling |
| Internal validation | 10 x repeated stratified 5-fold CV on Cleveland; the 10 out-of-fold probabilities per record are averaged |
| External validation | pipeline refitted once on all 303 Cleveland records, then frozen |
| Threshold | 0.50 for sensitivity, specificity, precision, F1, accuracy |
| Calibration | logistic recalibration `y ~ a + b * logit(p)`: intercept `a` (ideal 0), slope `b` (ideal 1). Calibration-in-the-large is also saved |
| Uncertainty | percentile 95% CI from 2,000 participant-level bootstrap resamples |
| Seed | 20260904 |

No oversampling, class weighting or hyperparameter search is used.

## Main results (portable logistic model)

| Evaluation | Records | AUC | Brier | Sensitivity | Specificity | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|---|
| Cleveland (internal CV) | 303 | 0.875 | 0.143 | 0.755 | 0.841 | -0.046 | 0.972 |
| Hungary | 294 | 0.877 | 0.135 | 0.783 | 0.814 | -0.551 | 0.909 |
| Switzerland | 123 | 0.766 | 0.267 | 0.609 | 0.625 | 2.968 | 0.585 |
| Long Beach VA | 200 | 0.722 | 0.188 | 0.785 | 0.529 | 0.664 | 0.488 |
| External pooled | 617 | 0.815 | 0.179 | 0.730 | 0.749 | 0.405 | 0.702 |

All four models, confidence intervals and the sex-stratified table are in
`outputs/` (`table3_internal_cv.csv`, `table4_external.csv`,
`table5_sex_subgroups.csv`, `results.xlsx`).

## Files

```
data/raw/                  four UCI files + SHA256SUMS
run_analysis.py            the whole analysis
build_report.py            outputs/report.html
verify_paper_numbers.py    paper values vs. reproduced values
sensitivity_chol_zero.py   zero-cholesterol sensitivity check
app.py                     Streamlit dashboard
outputs/
  table2..5 *.csv, results.xlsx, predictions.csv (one row per patient)
  figures/fig1..4 *.png
  run_manifest.json        seed, package versions, data hashes
  paper_vs_reproduced.csv  output of verify_paper_numbers.py
  console_log.txt          console output of the committed run
  models/                  fitted pipelines (.joblib, created on each run, not committed)
```

## Reproducibility notes

- Point estimates are deterministic for a given seed. The committed run matches
  all 91 point estimates in the paper to within 0.001.
- Three random-forest values differ from the paper in the third decimal
  (calibration slope 1.508 vs 1.507, Hungary Brier 0.132 vs 0.131, Long Beach VA
  AUC 0.750 vs 0.749).
- Confidence-interval bounds come from bootstrap resampling and depend on the
  random-number stream. They agree with the paper to within 0.01, except the
  random-forest interval for Switzerland (0.522-0.983 here, 0.544-0.981 in the
  paper), which rests on only 8 disease-free patients.
- Random-forest results can shift in the third decimal across scikit-learn versions.
  The versions used are recorded in `outputs/run_manifest.json`.

## Limitations

- **Zero-coded cholesterol.** Cholesterol is recorded as 0 for all 123
  Switzerland records and 49 of 200 Long Beach VA records. The main analysis
  keeps the UCI values unchanged, so Figure 1 counts them as present. Treating
  them as missing leaves the per-site AUCs almost unchanged (Switzerland 0.766,
  Long Beach VA 0.727) but improves Switzerland's Brier score from 0.267 to
  0.155 and its calibration intercept from 2.968 to 2.359, and raises the pooled
  external AUC from 0.815 to 0.853. Run `python sensitivity_chol_zero.py`.
- Data are from about 1988 and from patients referred for angiography.
- 303 development records; Switzerland has only 8 disease-free patients.
- The sex comparison is exploratory: only 25 external women had disease.

## Data

Janosi A, Steinbrunn W, Pfisterer M, Detrano R. Heart Disease. UCI Machine
Learning Repository, 1989. doi:10.24432/C52P4X (CC BY 4.0). See `data/README.md`.
