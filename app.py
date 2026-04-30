import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# ── CLEAN CSS (SAFE ONLY) ─────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700&family=Inter:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background:#fff7fa;
    color:#140d07;
    font-family:'Inter',sans-serif;
}

#MainMenu, footer {visibility:hidden;}

.banner {
    background: linear-gradient(90deg, #e56d85, #a41f39);
    padding: 1rem;
    border-radius: 8px;
    color:white;
    margin-bottom: 1rem;
}

.banner h1 {
    margin:0;
    font-family:'Syne';
    font-size:1.6rem;
}

/* FIX INPUT (single clean border) */
div[data-baseweb="input"] {
    border-radius:6px !important;
}

/* BUTTON */
.stButton button {
    background:#bf3853;
    color:white;
    border-radius:6px;
}
.stButton button:hover {
    background:#a41f39;
}

/* RESULT */
.result {
    padding:1.2rem;
    border-radius:8px;
    margin-top:1rem;
}

.benign {
    background:#f5f5dc;  /* REAL BEIGE */
    color:#000;
}

.malignant {
    background:#ffd6dd;
    border-left:4px solid #a41f39;
}

.small {font-size:0.8rem;}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────
@st.cache_resource
def load():
    return joblib.load("model.pkl"), joblib.load("scaler.pkl")

model, scaler = load()

# ── HEADER ───────────────────────────────
st.markdown("""
<div class="banner">
<h1>Breast Cancer Detection</h1>
</div>
""", unsafe_allow_html=True)

# ── FEATURES ─────────────────────────────
MEAN = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"]

SE = ["radius_se","texture_se","perimeter_se","area_se","smoothness_se",
      "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se"]

WORST = ["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
         "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

inputs = {}

# ── FIXED TABS (NO CSS INTERFERENCE) ─────
tab1, tab2, tab3 = st.tabs(["Mean Features", "Standard Error", "Worst Features"])

def render(features):
    cols = st.columns(5)
    for i, f in enumerate(features):
        with cols[i % 5]:
            inputs[f] = st.number_input(f, value=0.0, key=f)

with tab1:
    render(MEAN)

with tab2:
    render(SE)

with tab3:
    render(WORST)

# ── PREDICT ─────────────────────────────
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

    # ── RESULT ─────────────────────────
    if pred == 0:
        st.markdown(f"""
        <div class="result benign">
        <h3>✔ Benign Tumor</h3>
        <p class="small">Non-cancerous growth. Usually not harmful.</p>
        <b>Confidence:</b> {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result malignant">
        <h3>⚠ Malignant Tumor</h3>
        <p class="small">Potential cancerous growth. Seek medical advice.</p>
        <b>Confidence:</b> {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # ── CLEAN GRAPH ───────────────────
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        idx = np.argsort(imp)[-6:]

        fig, ax = plt.subplots(figsize=(5,3))

        ax.barh([all_features[i] for i in idx], imp[idx], color="#bf3853")

        ax.set_facecolor("#fff7fa")
        fig.patch.set_facecolor("#fff7fa")

        ax.tick_params(axis='both', labelsize=7, colors="#140d07")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title("Key Influencing Features", fontsize=9, color="#140d07")

        st.pyplot(fig)

# ── FOOTER ───────────────────────────────
st.markdown("⚠ For educational use only")