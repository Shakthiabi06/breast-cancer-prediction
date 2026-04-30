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
    --p3:#e56d85;
    --p4:#bf3853;
    --p5:#a41f39;
    --text:#140d07;
}

/* remove extra space */
.block-container {padding-top:0.8rem !important;}

html, body, [data-testid="stAppViewContainer"] {
    background:#fff7fa !important;
    color:var(--text) !important;
    font-family:'Inter',sans-serif !important;
}

#MainMenu, footer, header {visibility:hidden;}

/* banner */
.banner {
    background: linear-gradient(90deg, var(--p3), var(--p5));
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin-bottom: 0.8rem;
    color:white;
}
.banner h1 {
    margin:0;
    font-size:1.6rem;
    font-family:'Syne';
}

/* inputs */
label {
    font-size:0.72rem !important;
    color:#140d07 !important;
}

.stNumberInput input {
    background:white !important;
    border:1px solid #f891a5 !important;
    border-radius:5px !important;
    color:#140d07 !important;
    font-size:0.8rem !important;
}

/* button */
.stButton button {
    background:var(--p4);
    color:white;
    border-radius:6px;
    font-weight:600;
}
.stButton button:hover {
    background:var(--p5);
}

/* results */
.result {
    padding:1.2rem;
    border-radius:8px;
    margin-top:1rem;
}

.benign {
    background:#f4f400;
    color:#000;
}

.malignant {
    background:#ffd6dd;
    border-left:4px solid var(--p5);
}

.small {font-size:0.75rem;}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────
@st.cache_resource
def load():
    return joblib.load("model.pkl"), joblib.load("scaler.pkl")

model, scaler = load()

# ── HEADER ──────────────────────────
st.markdown('<div class="banner"><h1>Breast Cancer Detection</h1></div>', unsafe_allow_html=True)

# ── FEATURE GROUPS ───────────────────
MEAN = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"]

SE = ["radius_se","texture_se","perimeter_se","area_se","smoothness_se",
      "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se"]

WORST = ["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
         "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

inputs = {}

# ── FIXED TABS ───────────────────────
tabs = st.tabs(["Mean Features", "Standard Error", "Worst Features"])

def render(features):
    cols = st.columns(5)
    for i, f in enumerate(features):
        with cols[i % 5]:
            inputs[f] = st.number_input(f, value=0.0, key=f)

with tabs[0]:
    render(MEAN)

with tabs[1]:
    render(SE)

with tabs[2]:
    render(WORST)

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
        <p class="small">Non-cancerous tumor</p>
        <b>Confidence:</b> {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result malignant">
        <h3>⚠ Malignant</h3>
        <p class="small">Possible cancerous tumor</p>
        <b>Confidence:</b> {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # ── CLEAN GRAPH ─────────────────
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        idx = np.argsort(imp)[-8:]

        fig, ax = plt.subplots(figsize=(5.5,3))

        ax.barh([all_features[i] for i in idx], imp[idx], color="#e56d85")

        ax.set_facecolor("#fff7fa")
        fig.patch.set_facecolor("#fff7fa")

        ax.tick_params(axis='x', labelsize=7, colors="#140d07")
        ax.tick_params(axis='y', labelsize=7, colors="#140d07")

        ax.set_title("Top Features", fontsize=8, color="#140d07")

        st.pyplot(fig)

# ── FOOTER ─────────────────────────
st.markdown("⚠ Educational use only")