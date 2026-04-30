import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ── PAGE CONFIG ─────────────────────────────
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# ── CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --pink1:#fdb3c2;
    --pink2:#f891a5;
    --pink3:#e56d85;
    --pink4:#bf3853;
    --pink5:#a41f39;
    --text:#140d07;
}

html, body, [data-testid="stAppViewContainer"] {
    background:#fff7fa !important;
    color:var(--text) !important;
    font-family:'Inter',sans-serif !important;
}

#MainMenu, footer, header {visibility:hidden;}

/* Banner */
.banner {
    background: linear-gradient(90deg, var(--pink3), var(--pink5));
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    color: white;
}
.banner h1 {
    font-family:'Syne',sans-serif;
    font-size:2rem;
    margin:0;
}
.banner p {
    font-size:0.95rem;
    margin:0.3rem 0 0 0;
}

/* Inputs */
.stNumberInput input {
    background:white !important;
    border:1px solid var(--pink2) !important;
    border-radius:6px !important;
    color:#140d07 !important;
}

/* Button */
.stButton button {
    background:var(--pink4);
    color:white;
    border-radius:8px;
    font-weight:600;
}
.stButton button:hover {
    background:var(--pink5);
}

/* Result */
.result {
    padding:1.5rem;
    border-radius:10px;
    margin-top:1.5rem;
}
.benign {background:#ffe6ec; border-left:5px solid var(--pink2);}
.malignant {background:#ffd6dd; border-left:5px solid var(--pink5);}

</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl"), joblib.load("scaler.pkl")
    except:
        return None, None

model, scaler = load_model()

# ── BANNER ─────────────────────────────────
st.markdown("""
<div class="banner">
    <h1>Breast Cancer Detection</h1>
    <p>Machine Learning based prediction using tumor measurements</p>
</div>
""", unsafe_allow_html=True)

# ── FEATURES ───────────────────────────────
features = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

# ── INPUT GRID ─────────────────────────────
inputs = {}
cols = st.columns(5)

for i, f in enumerate(features):
    with cols[i % 5]:
        inputs[f] = st.number_input(f, value=0.0)

# ── PREDICT ────────────────────────────────
if st.button("Run Prediction"):

    if model is None:
        st.error("Model not loaded")
    else:
        data = np.array([[inputs[f] for f in features]])
        data = scaler.transform(data)

        pred = model.predict(data)[0]

        # ✅ FIXED CONFIDENCE
        try:
            proba = model.predict_proba(data)[0]
            confidence = max(proba) * 100
        except:
            score = model.decision_function(data)[0]
            confidence = min(99.0, 50 + abs(float(score)) * 10)

        # ── RESULT ────────────────────────
        if pred == 0:
            st.markdown(f"""
            <div class="result benign">
            <h3>✔ Benign</h3>
            Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result malignant">
            <h3>⚠ Malignant</h3>
            Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

        # ── FEATURE IMPORTANCE ───────────
        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
            top = np.argsort(importance)[-10:]

            plt.figure(figsize=(8,4))
            plt.barh([features[i] for i in top], importance[top])
            plt.title("Top Influencing Features")
            st.pyplot(plt)
            plt.clf()

# ── DISCLAIMER ─────────────────────────────
st.markdown("--- ⚠ Educational use only. Not a medical diagnosis.")