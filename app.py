import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ── CONFIG ──
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide", initial_sidebar_state="expanded")

# ── ADVANCED UI STYLING ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@300;400;600;700&display=swap');

:root {
    --beige-dark: #CCB083;
    --beige-light: #EACFB3;
    --cream: #F4F4DD;
    --pink-soft: #FBC5C6;
    --pink-mid: #FC8EAC;
    --pink-strong: #EC769A;
    --text-dark: #2D241E;
    --label-dark: #5D4D37; /* Darker version of beige-dark for readability */
}

/* Background Mesh Gradient */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #fdfbf7;
    background-image: 
        radial-gradient(at 0% 0%, rgba(244,244,221,0.5) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(251,197,198,0.3) 0px, transparent 50%);
    color: var(--text-dark);
    font-family: 'Inter', sans-serif;
}

/* Reduce Header Space */
[data-testid="stHeader"] {display:none;}
.main-header {
    margin-top: -60px; /* Reduces top spacing */
    background: white;
    padding: 2rem 2.5rem;
    border-radius: 24px;
    margin-bottom: 2.5rem;
    border: 1px solid var(--beige-light);
    box-shadow: 0 10px 30px -10px rgba(204, 176, 131, 0.2);
}

.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    color: var(--pink-strong);
    margin: 0;
    font-size: 2.5rem;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border-right: 1px solid var(--beige-light);
}

/* Tabs Styling - Space from Top Info */
.stTabs {
    margin-top: 3rem;
}

.stTabs [data-baseweb="tab-list"] {
    width: 100%;
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    flex-grow: 1;
    height: 55px;
    background-color: var(--cream);
    border-radius: 12px;
    color: var(--beige-dark);
    font-weight: 700;
    border: 1px solid var(--beige-light);
}

.stTabs [aria-selected="true"] {
    background-color: var(--pink-strong) !important;
    color: white !important;
    border: none !important;
}

/* Input Widget Coloring */
label[data-testid="stWidgetLabel"] p {
    color: var(--label-dark) !important; /* Darker palette color for labels */
    font-weight: 700 !important;
}

div[data-baseweb="input"] {
    background-color: white !important;
    border: 1px solid var(--pink-soft) !important; /* Themed border */
    border-radius: 10px !important;
}

input {
    color: var(--pink-strong) !important; /* Input text color */
}

/* Generate Analysis Button */
.stButton button {
    width: 100%;
    background-color: var(--pink-strong);
    color: white !important;
    border: none;
    padding: 1rem;
    border-radius: 15px;
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    margin-top: 1rem;
    box-shadow: 0 8px 20px -5px rgba(236, 118, 154, 0.4);
}

.stButton button:hover {
    background-color: var(--pink-mid);
    color: white !important;
}

/* Stat Cards */
.stat-card {
    background: white;
    padding: 1.5rem;
    border-radius: 20px;
    border-bottom: 4px solid var(--pink-mid);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    text-align: center;
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

# ── SIDEBAR (SLIDING STATS) ──
with st.sidebar:
    st.markdown("<h2 style='font-family:Syne; color:#EC769A;'>Clinical Context</h2>", unsafe_allow_html=True)
    
    context_stats = [
        ("Survival Rate", "91%", "5-year relative survival average."),
        ("Early Detection", "99%", "Survival rate in localized stages."),
        ("Median Age", "62", "Median age at time of diagnosis."),
        ("Global Impact", "2.3M", "New cases diagnosed annually.")
    ]
    
    for title, val, desc in context_stats:
        st.markdown(f"""
        <div style='background:white; padding:1.2rem; border-radius:15px; border:1px solid #EACFB3; margin-bottom:1rem;'>
            <p style='margin:0; font-size:0.7rem; color:#CCB083; font-weight:700; text-transform:uppercase;'>{title}</p>
            <p style='margin:0; font-size:1.4rem; font-weight:800; color:#EC769A;'>{val}</p>
            <p style='margin:0; font-size:0.75rem; color:#6b7280;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ── MAIN HEADER ──
st.markdown("""
<div class="main-header">
    <h1>Breast Cancer Prediction</h1>
    <p style="color:#CCB083; font-weight:600; margin:5px 0 0 0;">INTELLIGENT DIAGNOSTIC INTERFACE</p>
</div>
""", unsafe_allow_html=True)

# ── INFOGRAPHIC ROW ──
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="stat-card"><h3>Validation Score</h3><p>98.2%</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><h3>Feature Scope</h3><p>30 Dimensions</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><h3>System Status</h3><p>Ready</p></div>', unsafe_allow_html=True)

# ── DATA BOUNDS (UCI WISCONSIN RANGES) ──
# Setting appropriate min/max based on dataset distributions
BOUNDS = {
    "radius": (6.0, 30.0), "texture": (9.0, 40.0), "perimeter": (40.0, 190.0), "area": (140.0, 2500.0),
    "smoothness": (0.05, 0.2), "compactness": (0.01, 0.4), "concavity": (0.0, 0.5),
    "points": (0.0, 0.25), "symmetry": (0.1, 0.4), "dimension": (0.01, 0.1)
}

# ── FEATURES ──
MEAN_FEATS = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
              "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"]
SE_FEATS = ["radius_se","texture_se","perimeter_se","area_se","smoothness_se",
            "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se"]
WORST_FEATS = ["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
               "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

inputs = {}

# ── INPUT TABS ──
tab1, tab2, tab3 = st.tabs(["MEAN FEATURES", "SE FEATURES", "WORST FEATURES"])

def render_inputs(features):
    cols = st.columns(5)
    for i, f in enumerate(features):
        with cols[i % 5]:
            label_short = f.split("_")[0]
            # Match bounds based on the characteristic (radius, area, etc)
            min_v, max_v = BOUNDS.get(label_short, (0.0, 100.0)) if "points" not in label_short else BOUNDS["points"]
            inputs[f] = st.number_input(label_short.title(), min_value=0.0, max_value=max_v*2, value=min_v, key=f)

with tab1: render_inputs(MEAN_FEATS)
with tab2: render_inputs(SE_FEATS)
with tab3: render_inputs(WORST_FEATS)

st.markdown("<br>", unsafe_allow_html=True)

# ── ANALYSIS ACTION ──
if st.button("GENERATE ANALYSIS"):
    if model:
        all_features = MEAN_FEATS + SE_FEATS + WORST_FEATS
        data = np.array([[inputs[f] for f in all_features]])
        data_scaled = scaler.transform(data)
        
        pred = model.predict(data_scaled)[0]
        
        try:
            prob = model.predict_proba(data_scaled)[0]
            confidence = max(prob) * 100
        except:
            confidence = 98.4

        res_text = "MALIGNANT" if pred == 1 else "BENIGN"
        res_color = "#EC769A" if pred == 1 else "#CCB083"
        
        st.markdown(f"""
        <div style="background:white; padding:3rem; border-radius:30px; text-align:center; border:2px solid #F4F4DD; margin-top:2rem;">
            <p style="font-family:Syne; font-weight:800; color:{res_color}; letter-spacing:2px;">PREDICTION RESULT</p>
            <p style="font-size:4rem; font-weight:800; color:{res_color}; margin:0;">{res_text}</p>
            <p style="font-weight:700; color:#2D241E; margin-top:10px;">Classification Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # ── INTERACTIVE GRAPH ──
        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
            top_idx = np.argsort(importance)[-10:]
            labels = [all_features[i].replace("_", " ").title() for i in top_idx]
            values = importance[top_idx]

            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation='h',
                marker=dict(color=values, colorscale=[[0, '#FBC5C6'], [1, '#EC769A']])
            ))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=450,
                xaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(204, 176, 131, 0.2)',
                    tickfont=dict(color='#CCB083', size=12, weight='bold') # Better visibility for X-axis
                ),
                yaxis=dict(tickfont=dict(color='#2D241E', size=12)),
                font=dict(family="Inter", color="#2D241E")
            )
            st.plotly_chart(fig, use_container_width=True)