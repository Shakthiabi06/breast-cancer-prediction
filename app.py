import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ── PAGE CONFIG ─────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Detection",
    layout="wide"
)

# ── CSS FIXED (YOUR COLORS + VISIBILITY) ────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --pink1: #fdb3c2;
    --pink2: #f891a5;
    --pink3: #e56d85;
    --pink4: #bf3853;
    --pink5: #a41f39;
    --text-main: #140d07;
}

/* FORCE LIGHT MODE */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #fff7fa !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif !important;
}

/* REMOVE STREAMLIT DEFAULT */
#MainMenu, footer, header { visibility: hidden; }

/* FIX INPUT VISIBILITY */
.stNumberInput input {
    background-color: white !important;
    color: #140d07 !important;
    border: 1px solid #f891a5 !important;
    border-radius: 6px !important;
    font-size: 0.95rem !important;
}

label {
    color: #140d07 !important;
    font-size: 0.8rem !important;
}

/* BUTTON */
.stButton button {
    background-color: var(--pink4) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.stButton button:hover {
    background-color: var(--pink5) !important;
}

/* RESULT BOX */
.result-box {
    padding: 1.5rem;
    border-radius: 10px;
    margin-top: 1.5rem;
}

.benign {
    background: #ffe6ec;
    border-left: 5px solid var(--pink2);
}

.malignant {
    background: #ffd6dd;
    border-left: 5px solid var(--pink5);
}

/* TITLE */
.title {
    font-family: 'Syne', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    color: var(--pink5);
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .title {
        font-size: 1.6rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ──────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# ── TITLE ───────────────────────────────────
st.markdown('<div class="title">Breast Cancer Detection</div>', unsafe_allow_html=True)

# ── FEATURES ────────────────────────────────
feature_names = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

# ── INPUT GRID ──────────────────────────────
input_values = {}
cols = st.columns(5)

for i, feature in enumerate(feature_names):
    with cols[i % 5]:
        input_values[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.5f"
        )

# ── PREDICT BUTTON ──────────────────────────
if st.button("Run Prediction"):

    if model is None:
        st.error("Model not loaded")
    else:
        input_array = np.array([[input_values[f] for f in feature_names]])
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        confidence = max(proba) * 100

        # ── RESULT ──────────────────────────
        if prediction == 0:
            st.markdown(f'''
            <div class="result-box benign">
            <h3>✔ Benign</h3>
            Confidence: {confidence:.2f}%
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="result-box malignant">
            <h3>⚠ Malignant</h3>
            Confidence: {confidence:.2f}%
            </div>
            ''', unsafe_allow_html=True)

        # ── SIMPLE FEATURE IMPORTANCE (fallback) ──
        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
            top_idx = np.argsort(importance)[-10:]

            plt.figure(figsize=(8,4))
            plt.barh([feature_names[i] for i in top_idx], importance[top_idx])
            plt.title("Top Influencing Features")
            st.pyplot(plt)
            plt.clf()

# ── DISCLAIMER ──────────────────────────────
st.markdown("""
---
⚠ This tool is for educational purposes only. Not a medical diagnosis.
""")