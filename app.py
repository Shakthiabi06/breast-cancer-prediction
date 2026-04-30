import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# ── THEME ──────────────────────────────
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

/* Remove top gap */
.block-container {padding-top:1rem !important;}

html, body, [data-testid="stAppViewContainer"] {
    background:#fff7fa !important;
    color:var(--text) !important;
    font-family:'Inter',sans-serif !important;
}

#MainMenu, footer, header {visibility:hidden;}

/* Banner */
.banner {
    background: linear-gradient(90deg, var(--p3), var(--p5));
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    color:white;
}
.banner h1 {
    font-family:'Syne';
    margin:0;
    font-size:1.8rem;
}
.banner p {
    margin:0;
    font-size:0.85rem;
}

/* Inputs */
label {
    color:#140d07 !important;
    font-size:0.75rem !important;
    font-weight:500;
}

.stNumberInput input {
    background:white !important;
    border:1px solid var(--p2) !important;
    border-radius:6px !important;
    color:#140d07 !important;
}

/* Button */
.stButton button {
    background:var(--p4);
    color:white;
    border-radius:8px;
    font-weight:600;
}
.stButton button:hover {
    background:var(--p5);
}

/* Result */
.result {
    padding:1.5rem;
    border-radius:10px;
    margin-top:1.5rem;
}
.benign {
    background:#ffe6ec;
    border-left:5px solid var(--p2);
}
.malignant {
    background:#ffd6dd;
    border-left:5px solid var(--p5);
}
.small {
    font-size:0.8rem;
    color:#5c3b42;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl"), joblib.load("scaler.pkl")
    except:
        return None, None

model, scaler = load_model()

# ── BANNER ─────────────────────────────
st.markdown("""
<div class="banner">
    <h1>Breast Cancer Detection</h1>
    <p>AI-powered classification using 30 tumor features</p>
</div>
""", unsafe_allow_html=True)

# ── FEATURE RANGES (REALISTIC) ─────────
feature_ranges = {
    "radius_mean": (6, 30), "texture_mean": (9, 40), "perimeter_mean": (40, 200),
    "area_mean": (100, 2500), "smoothness_mean": (0.05, 0.2),
    "compactness_mean": (0.01, 0.4), "concavity_mean": (0, 0.5),
    "concave points_mean": (0, 0.2), "symmetry_mean": (0.1, 0.35),
    "fractal_dimension_mean": (0.05, 0.1),

    "radius_se": (0.1, 3), "texture_se": (0.3, 5), "perimeter_se": (0.5, 25),
    "area_se": (5, 600), "smoothness_se": (0.002, 0.03),
    "compactness_se": (0.002, 0.15), "concavity_se": (0, 0.4),
    "concave points_se": (0, 0.05), "symmetry_se": (0.01, 0.08),
    "fractal_dimension_se": (0.001, 0.03),

    "radius_worst": (7, 40), "texture_worst": (12, 50),
    "perimeter_worst": (50, 260), "area_worst": (200, 4300),
    "smoothness_worst": (0.07, 0.25), "compactness_worst": (0.02, 1.1),
    "concavity_worst": (0, 1.3), "concave points_worst": (0, 0.3),
    "symmetry_worst": (0.15, 0.7), "fractal_dimension_worst": (0.05, 0.2)
}

features = list(feature_ranges.keys())

# ── INPUT GRID ─────────────────────────
inputs = {}
cols = st.columns(5)

for i, f in enumerate(features):
    with cols[i % 5]:
        mn, mx = feature_ranges[f]
        inputs[f] = st.number_input(
            f,
            min_value=float(mn),
            max_value=float(mx),
            value=float((mn+mx)/2)
        )

# ── PREDICTION ─────────────────────────
if st.button("Run Prediction"):

    data = np.array([[inputs[f] for f in features]])
    data = scaler.transform(data)

    pred = model.predict(data)[0]

    # confidence fix
    try:
        confidence = max(model.predict_proba(data)[0]) * 100
    except:
        score = model.decision_function(data)[0]
        confidence = min(99, 50 + abs(score)*10)

    # ── RESULT UI ──────────────────────
    if pred == 0:
        st.markdown(f"""
        <div class="result benign">
        <h3>✔ Benign Tumor</h3>
        <p class="small">
        The model predicts this tumor is <b>non-cancerous</b>.
        Benign tumors do not spread and are usually less harmful.
        </p>
        <b>Confidence:</b> {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result malignant">
        <h3>⚠ Malignant Tumor</h3>
        <p class="small">
        The model indicates possible <b>cancerous growth</b>.
        Immediate medical consultation is strongly recommended.
        </p>
        <b>Confidence:</b> {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # ── GRAPH (FIXED COLORS) ───────────
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        idx = np.argsort(imp)[-10:]

        plt.figure(figsize=(7,4))
        plt.barh([features[i] for i in idx], imp[idx])
        plt.title("Top Influencing Features")
        plt.gca().set_facecolor("#fff7fa")
        plt.gcf().patch.set_facecolor("#fff7fa")

        # remove blue
        for bar in plt.gca().patches:
            bar.set_color("#e56d85")

        plt.xticks(color="#140d07")
        plt.yticks(color="#140d07")

        st.pyplot(plt)
        plt.clf()

# ── DISCLAIMER ─────────────────────────
st.markdown("⚠ For educational use only. Not a medical diagnosis.")