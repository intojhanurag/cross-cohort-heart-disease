# Data

The four processed files of the UCI Heart Disease repository, unmodified:

| File | Cohort | Records |
|---|---|---|
| `processed.cleveland.data` | Cleveland Clinic Foundation, USA | 303 |
| `processed.hungarian.data` | Hungarian Institute of Cardiology, Budapest | 294 |
| `processed.switzerland.data` | University Hospitals Zurich and Basel, Switzerland | 123 |
| `processed.va.data` | V.A. Medical Center, Long Beach, USA | 200 |

Source: Janosi A, Steinbrunn W, Pfisterer M, Detrano R. Heart Disease. UCI Machine
Learning Repository, 1989. doi:10.24432/C52P4X. Licence: CC BY 4.0.

Each file has 14 comma-separated columns and no header; `?` marks a missing value:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num`

`num` is the angiographic label (0 = less than 50% narrowing, 1-4 = disease).
`raw/SHA256SUMS` holds the file hashes; `run_analysis.py` records the hashes it
actually used in `outputs/run_manifest.json`.
