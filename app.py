import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ── CONFIG ──
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# ── THEME & STYLING ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #fdfbfc;
    color: #140d07;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Hide Streamlit elements */
#MainMenu, footer, header {visibility: hidden;}

/* Custom Banner */
.banner {
    background: linear-gradient(135deg, #a41f39 0%, #e56d85 100%);
    padding: 2.5rem;
    border-radius: 15px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(164, 31, 57, 0.2);
}

.banner h1 {
    margin: 0;
    font-weight: 700;
    font-size: 2.2rem;
    letter-spacing: -0.5px;
}

/* Input Card Styling */
div[data-baseweb="input"] {
    border-radius: 8px !important;
    border: 1px solid #eee !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: #f8f9fa;
    border-radius: 8px 8px 0 0;
    padding: 0 20px;
    color: #666;
}

.stTabs [aria-selected="true"] {
    background-color: #a41f39 !important;
    color: white !important;
}

/* Button */
.stButton button {
    width: 100%;
    background-color: #a41f39;
    color: white;
    border: none;
    padding: 0.75rem;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton button:hover {
    background-color: #83182d;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(164, 31, 57, 0.3);
}

/* Results */
.result-card {
    padding: 2rem;
    border-radius: 15px;
    margin-top: 1.5rem;
    text-align: center;
}

.benign {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
}

.malignant {
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
}

.disclaimer {
    font-size: 0.8rem;
    color: #888;
    text-align: center;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ──
@st.cache_resource
def load_assets():
    # Make sure these filenames match your local files exactly
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Model files not found. Ensure 'model.pkl' and 'scaler.pkl' are in your folder.")
    st.stop()

# ── HEADER ──
st.markdown("""
<div class="banner">
    <h1>Breast Cancer Diagnostics</h1>
    <p style="opacity:0.9">Advanced Neural Analysis Interface</p>
</div>
""", unsafe_allow_html=True)

# ── FEATURE LISTS ──
MEAN = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"]

SE = ["radius_se","texture_se","perimeter_se","area_se","smoothness_se",
      "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se"]

WORST = ["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
         "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

inputs = {}

# ── TABS ──
tab1, tab2, tab3 = st.tabs(["📊 Mean", "📉 Standard Error", "🔝 Worst Cases"])

def render_inputs(features):
    cols = st.columns(2) 
    for i, f in enumerate(features):
        with cols[i % 2]:
            label = f.replace("_", " ").title()
            inputs[f] = st.number_input(label, value=0.0, key=f)

with tab1: render_inputs(MEAN)
with tab2: render_inputs(SE)
with tab3: render_inputs(WORST)

st.markdown("<br>", unsafe_allow_html=True)

# ── PREDICTION LOGIC ──
if st.button("Analyze Biopsy Data"):
    all_features = MEAN + SE + WORST
    data_raw = np.array([[inputs[f] for f in all_features]])
    data_scaled = scaler.transform(data_raw)
    
    prediction = model.predict(data_scaled)[0]
    
    # Calculate Confidence
    try:
        prob = model.predict_proba(data_scaled)[0]
        confidence = max(prob) * 100
    except:
        # Fallback for models like SVM that don't always have predict_proba
        score = model.decision_function(data_scaled)[0]
        confidence = min(99.9, 50 + abs(score)*10)

    # ── DISPLAY RESULT ──
    res_class = "benign" if prediction == 0 else "malignant"
    res_label = "Benign (Non-Cancerous)" if prediction == 0 else "Malignant (Potential Growth)"
    res_icon = "🍀" if prediction == 0 else "⚠️"

    st.markdown(f"""
    <div class="result-card {res_class}">
        <h2 style="margin:0;">{res_icon} {res_label}</h2>
        <p style="font-size:1.2rem; margin-top:10px;">Analysis Confidence: <b>{confidence:.2f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    # ── KEY DRIVERS GRAPH (PLOTLY) ──
    if hasattr(model, "coef_"):
        st.markdown("<h4 style='text-align: center; margin-top: 2rem;'>Diagnostic Drivers</h4>", unsafe_allow_html=True)
        
        importance = np.abs(model.coef_[0])
        top_indices = np.argsort(importance)[-8:]
        top_features = [all_features[i].replace("_", " ").title() for i in top_indices]
        top_values = importance[top_indices]

        fig = go.Figure(go.Bar(
            x=top_values,
            y=top_features,
            orientation='h',
            marker=dict(color='#a41f39', line=dict(color='#83182d', width=1))
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=10),
            height=350,
            xaxis=dict(showgrid=True, gridcolor='#eee'),
            font=dict(family="Plus Jakarta Sans", size=12, color="#140d07")
        )

        st.plotly_chart(fig, use_container_width=True)

# ── FOOTER ──
st.markdown("""
<div class="disclaimer">
    ⚠️ <b>Disclaimer:</b> For educational use only. Not for medical diagnosis.
</div>
""", unsafe_allow_html=True)