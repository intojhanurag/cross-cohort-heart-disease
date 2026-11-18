"""Live demo dashboard.   Run:  streamlit run app.py

Shows the tables and figures produced by run_analysis.py and lets you score a
single patient with the frozen portable logistic model.
Research prototype only; not for clinical use.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import run_analysis as ra

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="Cross-Cohort Heart Disease Detection", layout="wide")
st.title("Cross-Cohort Heart Disease Detection")
st.caption("Trained on Cleveland (303 records), tested unchanged on Hungary, Switzerland and "
           "Long Beach VA (617 records). Research prototype, not a diagnostic tool.")

if not (OUT / "table3_internal_cv.csv").exists():
    st.error("No results found. Run `python run_analysis.py` first.")
    st.stop()

@st.cache_resource
def portable_logistic():
    """Same frozen model as in the paper: fitted once on all 303 Cleveland records."""
    data = ra.load_cohorts()
    dev = data[data.cohort == ra.DEV]
    return ra.make_pipeline("logistic", "Portable").fit(ra.features(dev, "Portable"), dev.y)


manifest = json.loads((OUT / "run_manifest.json").read_text())
internal = pd.read_csv(OUT / "table3_internal_cv.csv")
external = pd.read_csv(OUT / "table4_external.csv")
sex = pd.read_csv(OUT / "table5_sex_subgroups.csv")
cohorts = pd.read_csv(OUT / "table2_cohorts.csv")
SHOW = ["n", "auc", "auc_lo", "auc_hi", "brier", "sensitivity", "specificity", "cal_intercept", "cal_slope"]

tab_data, tab_results, tab_explain, tab_patient, tab_repro = st.tabs(
    ["Data", "Results", "Interpretability & subgroups", "Score a patient", "Reproducibility"])

with tab_data:
    st.subheader("Four UCI cohorts, 920 patients")
    st.dataframe(cohorts, hide_index=True, width="stretch")
    st.bar_chart(cohorts.set_index("cohort")["prevalence_pct"], y_label="Patients with disease (%)")
    st.subheader("Predictor missingness by cohort")
    st.image(str(OUT / "figures" / "fig1_missingness.png"))
    st.info("Vessel count (ca), thalassemia (thal) and ST slope are mostly missing outside Cleveland, "
            "so the portable model uses the 10 remaining predictors.")

with tab_results:
    pl_int = internal.set_index("model").loc["Portable logistic"]
    ext = external.set_index(["model", "cohort"])
    c = st.columns(5)
    c[0].metric("Cleveland AUC (internal CV)", f"{pl_int.auc:.3f}")
    for col, name in zip(c[1:4], ["Hungary", "Switzerland", "Long Beach VA"]):
        v = ext.loc[("Portable logistic", name), "auc"]
        col.metric(f"{name} AUC", f"{v:.3f}", f"{v - pl_int.auc:+.3f}")
    c[4].metric("Pooled external AUC", f"{ext.loc[('Portable logistic', 'External pooled'), 'auc']:.3f}")

    st.subheader("Internal validation: 10x repeated stratified 5-fold CV on Cleveland")
    st.dataframe(internal[["model"] + SHOW].round(3), hide_index=True, width="stretch")
    st.subheader("External validation: frozen models, no refitting")
    st.dataframe(external[["model", "cohort"] + SHOW].round(3), hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.image(str(OUT / "figures" / "fig2_roc_portable_logistic.png"), caption="ROC curves, portable logistic")
    b.image(str(OUT / "figures" / "fig3_calibration_portable_logistic.png"), caption="Calibration, portable logistic")

with tab_explain:
    a, b = st.columns(2)
    a.subheader("Standardized coefficients")
    a.image(str(OUT / "figures" / "fig4_coefficients_portable_logistic.png"))
    a.caption("Associations in a historical angiography-referred cohort, not causal effects.")
    b.subheader("Sex-stratified performance (exploratory)")
    b.dataframe(sex[["evaluation", "subgroup", "n", "disease", "auc", "auc_lo", "auc_hi",
                     "sensitivity", "specificity"]].round(3), hide_index=True, width="stretch")
    b.warning("Only 25 external women had disease. This is a warning to test further, not proof of bias.")

with tab_patient:
    st.subheader("Score one patient with the frozen portable logistic model")
    model = portable_logistic()
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)", 20, 90, 56)
    sex_in = c1.selectbox("Sex", ["Male", "Female"])
    cp = c1.selectbox("Chest-pain type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], index=3)
    trestbps = c2.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = c2.number_input("Serum cholesterol (mg/dl)", 100, 600, 240)
    fbs = c2.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
    restecg = c2.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"])
    thalach = c3.number_input("Maximum heart rate (bpm)", 60, 220, 150)
    exang = c3.selectbox("Exercise-induced angina", ["No", "Yes"])
    oldpeak = c3.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)

    row = pd.DataFrame([{
        "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach, "oldpeak": oldpeak,
        "sex": float(sex_in == "Male"),
        "cp": float(["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp) + 1),
        "fbs": float(fbs == "Yes"),
        "restecg": float(["Normal", "ST-T abnormality", "LV hypertrophy"].index(restecg)),
        "exang": float(exang == "Yes"),
    }])
    p = float(model.predict_proba(row)[0, 1])
    st.metric("Predicted probability of angiographic coronary artery disease", f"{p:.1%}",
              "above 0.50 cut-off" if p >= 0.5 else "below 0.50 cut-off", delta_color="off")
    st.progress(p)
    st.warning("This probability is calibrated for Cleveland only. The study shows it under-predicts "
               "risk in high-prevalence cohorts (calibration intercept +2.97 in Switzerland). Not for clinical use.")

with tab_repro:
    st.subheader("Run configuration")
    st.json(manifest)
    cmp = OUT / "paper_vs_reproduced.csv"
    if cmp.exists():
        st.subheader("Paper values vs. this run")
        st.dataframe(pd.read_csv(cmp), hide_index=True, width="stretch")
    st.code("python run_analysis.py          # regenerate everything from data/raw\n"
            "python verify_paper_numbers.py  # compare against the numbers in the paper", language="bash")
