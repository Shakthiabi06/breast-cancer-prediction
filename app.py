import streamlit as st
import joblib
import numpy as np

# ── PAGE CONFIG ─────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🧬",
    layout="wide"
)

# ── CSS (CLEAN + ACCESSIBLE) ────────────────────────
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

/* Force light mode */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #fff7fa !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Remove streamlit junk */
#MainMenu, footer, header { visibility: hidden; }

/* Layout */
.block-container {
    padding: 2rem 3rem !important;
}

/* Title */
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--pink5);
    margin-bottom: 0.5rem;
}

/* Subtitle */
.sub-text {
    font-size: 1rem;
    color: var(--text-main);
    margin-bottom: 1.5rem;
}

/* Inputs */
.stNumberInput input {
    font-size: 0.95rem !important;
    padding: 8px !important;
}

/* Button */
.stButton button {
    background-color: var(--pink4);
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.6rem;
}

.stButton button:hover {
    background-color: var(--pink5);
}

/* Result Box */
.result-box {
    padding: 1.5rem;
    border-radius: 10px;
    margin-top: 1.5rem;
    font-size: 1rem;
}

.benign {
    background: #ffe6ec;
    border-left: 5px solid var(--pink2);
}

.malignant {
    background: #ffd6dd;
    border-left: 5px solid var(--pink5);
}

/* Responsive */
@media (max-width: 768px) {
    .main-title {
        font-size: 1.8rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ──────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# ── TITLE ───────────────────────────────────────────
st.markdown('<div class="main-title">Breast Cancer Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Enter tumor measurements to predict diagnosis</div>', unsafe_allow_html=True)

# ── INPUTS ──────────────────────────────────────────
feature_names = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

input_values = []

cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(feature, value=0.0)
        input_values.append(val)

# ── PREDICT ─────────────────────────────────────────
if st.button("Predict"):
    if model is None:
        st.error("Model not loaded")
    else:
        data = np.array([input_values])
        data = scaler.transform(data)

        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0]

        confidence = max(prob) * 100

        if pred == 0:
            st.markdown(f'''
            <div class="result-box benign">
            <b>Result:</b> Benign<br><br>
            Confidence: {confidence:.2f}%
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="result-box malignant">
            <b>Result:</b> Malignant<br><br>
            Confidence: {confidence:.2f}%
            </div>
            ''', unsafe_allow_html=True)

# ── DISCLAIMER ──────────────────────────────────────
st.markdown("""
---
This tool is for educational purposes only. Always consult a medical professional.
""")