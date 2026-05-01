import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ──
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# ── THEME & STYLING ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #140d07;
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit components */
#MainMenu, footer, header {visibility: hidden;}

/* Professional Banner */
.banner {
    background: linear-gradient(90deg, #f06292 0%, #a41f39 100%);
    padding: 3rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2.5rem;
    text-align: left;
}

.banner h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: -1px;
}

/* Tab Styling - Better Visibility */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    height: 45px;
    background-color: white;
    border-radius: 8px;
    color: #a41f39;
    font-weight: 600;
    border: 1px solid #ffebee;
}

.stTabs [aria-selected="true"] {
    background-color: #a41f39 !important;
    color: white !important;
}

/* Fixed Input Colors - No blending */
label[data-testid="stWidgetLabel"] p {
    color: #140d07 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

div[data-baseweb="input"] {
    background-color: #fff !important;
    border: 1.5px solid #ffebee !important;
    border-radius: 10px !important;
}

/* Professional Button */
.stButton button {
    width: 100%;
    background-color: #a41f39;
    color: white;
    border: none;
    padding: 1rem;
    border-radius: 12px;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    transition: 0.3s;
}

.stButton button:hover {
    background-color: #e91e63;
    border: none;
    color: white;
}

/* Results Cards */
.result-card {
    padding: 2rem;
    border-radius: 20px;
    margin-top: 2rem;
    border: 1px solid transparent;
}

.benign {
    background-color: #fce4ec;
    border-color: #f06292;
    color: #880e4f;
}

.malignant {
    background-color: #140d07;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ──
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

if model is None:
    st.error("Model files not found. Ensure 'model.pkl' and 'scaler.pkl' are in the root directory.")
    st.stop()

# ── HEADER ──
st.markdown("""
<div class="banner">
    <h1>Diagnostics</h1>
    <p style="font-weight:600; opacity:0.9;">Breast Cancer Prediction & Feature Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── FEATURES ──
MEAN = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"]
SE = ["radius_se","texture_se","perimeter_se","area_se","smoothness_se",
      "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se"]
WORST = ["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
         "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

inputs = {}

# ── TABS ──
tab1, tab2, tab3 = st.tabs(["PRIMARY METRICS", "STANDARD ERROR", "EXTREME VALUES"])

def render_inputs(features):
    cols = st.columns(5)
    for i, f in enumerate(features):
        with cols[i % 5]:
            label = f.split("_")[0].title()
            inputs[f] = st.number_input(label, value=0.0, key=f)

with tab1: render_inputs(MEAN)
with tab2: render_inputs(SE)
with tab3: render_inputs(WORST)

st.markdown("<br>", unsafe_allow_html=True)

# ── PREDICTION ──
if st.button("RUN CLINICAL ANALYSIS"):
    all_features = MEAN + SE + WORST
    data = np.array([[inputs[f] for f in all_features]])
    data_scaled = scaler.transform(data)
    
    pred = model.predict(data_scaled)[0]
    
    try:
        prob = model.predict_proba(data_scaled)[0]
        confidence = max(prob) * 100
    except:
        score = model.decision_function(data_scaled)[0]
        confidence = min(99.0, 50 + abs(score)*10)

    # ── RESULT DISPLAY ──
    if pred == 0:
        st.markdown(f"""
        <div class="result-card benign">
            <h2 style="margin:0; font-family:'Syne';">BENIGN</h2>
            <p>Non-cancerous growth detected. Analysis suggests a stable cellular structure.</p>
            <p style="font-size:0.9rem;"><b>Confidence Level:</b> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card malignant">
            <h2 style="margin:0; font-family:'Syne'; color:#f06292;">MALIGNANT</h2>
            <p>Malignant cellular patterns detected. Immediate medical consultation is advised.</p>
            <p style="font-size:0.9rem;"><b>Confidence Level:</b> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # ── GRAPH (AESTHETIC MATPLOTLIB) ──
    if hasattr(model, "coef_"):
        st.markdown("<br><h3 style='font-family:Syne;'>Informed Feature Drivers</h3>", unsafe_allow_html=True)
        
        importance = np.abs(model.coef_[0])
        top_idx = np.argsort(importance)[-6:]
        labels = [all_features[i].replace("_", " ").title() for i in top_idx]
        values = importance[top_idx]

        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Aesthetic horizontal bars
        bars = ax.barh(labels, values, color='#a41f39', height=0.6)
        
        # Styling
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#ffebee')
        ax.spines['left'].set_color('#ffebee')
        
        ax.tick_params(axis='y', colors='#140d07', labelsize=10)
        ax.tick_params(axis='x', colors='#a41f39', labelsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)

# ── FOOTER ──
st.markdown("""
<div style="text-align:center; margin-top:50px; color:#bbb; font-size:0.8rem;">
    FOR EDUCATIONAL PURPOSES ONLY • DATA PRIVACY COMPLIANT
</div>
""", unsafe_allow_html=True)