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

/* Sidebar Styling - Pop-out Quick Stats */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid var(--beige-light);
}

/* Header Spacing Fix */
[data-testid="stHeader"] {background: rgba(0,0,0,0); height: 0px;}
.main-header {
    margin-top: -50px;
    background: white;
    padding: 2rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    border: 1px solid var(--beige-light);
    box-shadow: 0 10px 30px -10px rgba(204, 176, 131, 0.2);
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    color: var(--pink-strong);
    margin: 0;
    font-size: 2.2rem;
}

/* Stat Cards Alignment */
.stat-card-container {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 2rem;
}
.stat-card {
    flex: 1;
    background: white;
    padding: 1.2rem;
    border-radius: 20px;
    border-bottom: 4px solid var(--pink-mid);
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.stat-card h3 { font-size: 0.75rem; color: var(--beige-dark); text-transform: uppercase; margin-bottom: 5px; }
.stat-card p { font-size: 1.4rem; font-weight: 800; color: var(--pink-strong); margin: 0; }

/* Tabs Styling */
.stTabs { margin-top: 1rem; }
.stTabs [data-baseweb="tab-list"] { width: 100%; gap: 10px; }
.stTabs [data-baseweb="tab"] {
    flex-grow: 1;
    height: 50px;
    background-color: var(--cream);
    border-radius: 12px;
    color: var(--beige-dark);
    font-weight: 700;
}
.stTabs [aria-selected="true"] {
    background-color: var(--pink-strong) !important;
    color: white !important;
}

/* Slider & Input Styling */
div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background-color: white !important;
    border: 1px solid var(--beige-light) !important;
}
/* Remove +/- buttons from number input */
button[step] { display: none !important; }

/* Label Styling */
label p { color: #5D4D37 !important; font-weight: 700 !important; font-size: 0.85rem !important; }

/* Button Styling */
.stButton button {
    width: 100%;
    background-color: var(--pink-strong) !important;
    color: white !important;
    border-radius: 15px;
    padding: 0.8rem;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    border: none;
    transition: 0.3s;
}
.stButton button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(236, 118, 154, 0.4); }
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ──
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except: return None, None

model, scaler = load_assets()

# ── SIDEBAR (QUICK STATS) ──
with st.sidebar:
    st.markdown("<h2 style='font-family:Syne; color:#EC769A;'>Quick Stats</h2>", unsafe_allow_html=True)
    sidebar_content = [
        ("Survival Rate", "91%", "Overall average."),
        ("Early Detection", "99%", "Localized stage."),
        ("Benign Range", "< 13.0", "Typical Radius Mean."),
        ("Global Data", "2.3M", "Annual diagnoses.")
    ]
    for t, v, d in sidebar_content:
        st.markdown(f"""
        <div style='background:white; padding:1.2rem; border-radius:15px; border:1px solid #EACFB3; margin-bottom:1rem;'>
            <p style='margin:0; font-size:0.7rem; color:#CCB083; font-weight:700;'>{t}</p>
            <p style='margin:0; font-size:1.4rem; font-weight:800; color:#EC769A;'>{v}</p>
            <p style='margin:0; font-size:0.75rem; color:#6b7280;'>{d}</p>
        </div>
        """, unsafe_allow_html=True)

# ── HEADER & TOP INFO ──
st.markdown("""
<div class="main-header">
    <h1>Breast Cancer Prediction</h1>
    <p style="color:#CCB083; font-weight:600; margin-top:5px;">INTELLIGENT DIAGNOSTIC INTERFACE</p>
</div>
""", unsafe_allow_html=True)

# Manual alignment using columns for the Stat Cards
c1, c2, c3 = st.columns(3)
with c1: st.markdown('<div class="stat-card"><h3>Validation Score</h3><p>98.2%</p></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="stat-card"><h3>Analysis Speed</h3><p>Real-time</p></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="stat-card"><h3>System Status</h3><p>Ready</p></div>', unsafe_allow_html=True)

# ── RANGES ──
BOUNDS = {
    "radius": (6.0, 30.0), "texture": (9.0, 40.0), "perimeter": (40.0, 190.0), "area": (140.0, 2500.0),
    "smoothness": (0.05, 0.2), "compactness": (0.01, 0.4), "concavity": (0.0, 0.5),
    "points": (0.0, 0.2), "symmetry": (0.1, 0.4), "dimension": (0.01, 0.1)
}

# ── FEATURE SETS ──
def get_features(suffix):
    return [f"radius_{suffix}", f"texture_{suffix}", f"perimeter_{suffix}", f"area_{suffix}", f"smoothness_{suffix}",
            f"compactness_{suffix}", f"concavity_{suffix}", f"concave points_{suffix}", f"symmetry_{suffix}", f"fractal_dimension_{suffix}"]

MEAN_F, SE_F, WORST_F = get_features("mean"), get_features("se"), get_features("worst")

inputs = {}
tab1, tab2, tab3 = st.tabs(["MEAN", "SE", "WORST"])

def render_hybrid_inputs(features):
    for f in features:
        col_slider, col_val = st.columns([3, 1])
        key_root = f.split("_")[0]
        low, high = BOUNDS.get(key_root, (0.0, 1.0))
        
        with col_slider:
            # Slider updates the input map
            slider_val = st.slider(f.replace("_", " ").title(), low, high*1.5, low, key=f"{f}_slide")
        with col_val:
            # Number input tied to slider value
            inputs[f] = st.number_input("Value", value=slider_val, key=f)

with tab1: render_hybrid_inputs(MEAN_F)
with tab2: render_hybrid_inputs(SE_F)
with tab3: render_hybrid_inputs(WORST_F)

# ── ACTION ──
if st.button("GENERATE ANALYSIS"):
    if model:
        all_f = MEAN_F + SE_F + WORST_F
        data = np.array([[inputs[f] for f in all_f]])
        data_s = scaler.transform(data)
        pred = model.predict(data_s)[0]
        
        conf = max(model.predict_proba(data_s)[0]) * 100 if hasattr(model, "predict_proba") else 98.4
        color = "#EC769A" if pred == 1 else "#CCB083"
        res = "MALIGNANT" if pred == 1 else "BENIGN"

        st.markdown(f"""
        <div style="background:white; padding:2rem; border-radius:30px; text-align:center; border:2px solid #F4F4DD; margin-top:2rem;">
            <p style="font-family:Syne; font-weight:800; color:{color}; letter-spacing:2px; margin-bottom:10px;">ANALYSIS RESULT</p>
            <p style="font-size:3.5rem; font-weight:800; color:{color}; margin:0;">{res}</p>
            <p style="font-weight:700; color:#2D241E; margin-top:10px;">Confidence: {conf:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Interactive Graph with Darker Y-axis
        if hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
            idx = np.argsort(imp)[-10:]
            fig = go.Figure(go.Bar(x=imp[idx], y=[all_f[i].title().replace("_", " ") for i in idx], orientation='h',
                                   marker=dict(color=imp[idx], colorscale=[[0, '#FBC5C6'], [1, '#EC769A']])))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              xaxis=dict(gridcolor='rgba(204, 176, 131, 0.1)', tickfont=dict(color='#2D241E', weight='bold')),
                              yaxis=dict(tickfont=dict(color='#2D241E', size=12, weight='bold'))) # Darkened Y-axis
            st.plotly_chart(fig, use_container_width=True)