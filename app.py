import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# ── CSS ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --p1:#fdb3c2;
    --p2:#f891a5;
    --p3:#e56d85;
    --p4:#bf3853;
    --p5:#a41f39;
    --text:#140d07;
}

.block-container {padding-top:1rem !important;}

html, body, [data-testid="stAppViewContainer"] {
    background:#fff7fa !important;
    color:var(--text) !important;
    font-family:'Inter',sans-serif !important;
}

#MainMenu, footer, header {visibility:hidden;}

.banner {
    background: linear-gradient(90deg, var(--p3), var(--p5));
    padding: 1.2rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    color:white;
}

.banner h1 {
    font-family:'Syne';
    font-size:1.8rem;
    margin:0;
}

label {
    font-size:0.75rem !important;
    font-weight:500;
    color:#140d07 !important;
}

.stNumberInput input {
    background:white !important;
    border:1px solid var(--p2) !important;
    border-radius:6px !important;
    color:#140d07 !important;
}

.stButton button {
    background:var(--p4);
    color:white;
    border-radius:8px;
    font-weight:600;
}
.stButton button:hover {
    background:var(--p5);
}

.result {
    padding:1.5rem;
    border-radius:10px;
    margin-top:1rem;
}

.benign {
    background:#f5f50c;
    color:#000;
}

.malignant {
    background:#ffd6dd;
    border-left:5px solid var(--p5);
}

.small {
    font-size:0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl"), joblib.load("scaler.pkl")

model, scaler = load_model()

# ── HEADER ──────────────────────────
st.markdown("""
<div class="banner">
<h1>Breast Cancer Detection</h1>
</div>
""", unsafe_allow_html=True)

# ── FEATURE GROUPS ───────────────────
MEAN = [
"radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
"compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"
]

SE = [
"radius_se","texture_se","perimeter_se","area_se","smoothness_se",
"compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se"
]

WORST = [
"radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
"compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

# ── INPUT TABS ───────────────────────
inputs = {}

tab1, tab2, tab3 = st.tabs(["Mean Features", "Standard Error", "Worst Features"])

def render(features):
    cols = st.columns(5)
    for i, f in enumerate(features):
        with cols[i % 5]:
            inputs[f] = st.number_input(f, value=0.0)

with tab1: render(MEAN)
with tab2: render(SE)
with tab3: render(WORST)

# ── PREDICT ─────────────────────────
if st.button("Run Prediction"):

    all_features = MEAN + SE + WORST
    data = np.array([[inputs[f] for f in all_features]])
    data = scaler.transform(data)

    pred = model.predict(data)[0]

    try:
        confidence = max(model.predict_proba(data)[0]) * 100
    except:
        score = model.decision_function(data)[0]
        confidence = min(99, 50 + abs(score)*10)

    # ── RESULT ──────────────────────
    if pred == 0:
        st.markdown(f"""
        <div class="result benign">
        <h3>✔ Benign</h3>
        <p class="small">
        Non-cancerous tumor. Usually does not spread.
        </p>
        <b>Confidence:</b> {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result malignant">
        <h3>⚠ Malignant</h3>
        <p class="small">
        Possible cancerous tumor. Seek medical advice.
        </p>
        <b>Confidence:</b> {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # ── GRAPH (FIXED AESTHETIC) ─────
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        idx = np.argsort(imp)[-8:]

        plt.figure(figsize=(6,3))
        plt.barh([all_features[i] for i in idx], imp[idx], color="#e56d85")

        plt.xticks(fontsize=7, color="#140d07")
        plt.yticks(fontsize=7, color="#140d07")

        plt.gca().set_facecolor("#fff7fa")
        plt.gcf().patch.set_facecolor("#fff7fa")

        plt.title("Top Features", fontsize=9, color="#140d07")

        st.pyplot(plt)
        plt.clf()

st.markdown("⚠ Educational use only")