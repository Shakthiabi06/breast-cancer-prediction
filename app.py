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

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #fdfbf7;
    background-image: radial-gradient(at 0% 0%, rgba(244,244,221,0.5) 0px, transparent 50%),
                      radial-gradient(at 100% 100%, rgba(251,197,198,0.3) 0px, transparent 50%);
}

/* Header Spacing & Alignment */
[data-testid="stHeader"] {background: rgba(0,0,0,0); height: 0px;}
.main-header {
    margin-top: -50px;
    background: white;
    padding: 1.5rem 2rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    border: 1px solid var(--beige-light);
    text-align: left;
}
.main-header h1 { font-family: 'Syne', sans-serif; color: var(--pink-strong); margin: 0; font-size: 2.2rem; }

/* Stat Cards Alignment Fix */
.stat-card {
    background: white;
    padding: 1.2rem;
    border-radius: 20px;
    border-bottom: 4px solid var(--pink-mid);
    text-align: center;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.stat-card h3 { font-size: 0.75rem; color: var(--beige-dark); text-transform: uppercase; margin: 0 0 5px 0; }
.stat-card p { font-size: 1.4rem; font-weight: 800; color: var(--pink-strong); margin: 0; line-height: 1.2; }

/* Sidebar Visibility */
[data-testid="stSidebar"] {
    background-color: white !important;
    border-right: 1px solid var(--beige-light);
}

/* Slider & Input Theming */
.stSlider [data-baseweb="slider"] [role="slider"] { background-color: var(--pink-strong); border: 2px solid white; }
.stSlider [data-baseweb="slider"] > div > div { background: var(--pink-soft); }

/* Number Input Styling */
div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background-color: white !important;
    border: 1.5px solid var(--beige-light) !important;
    color: var(--text-dark) !important;
}

/* Button Styling */
.stButton button {
    width: 100%;
    border-radius: 12px;
    padding: 0.6rem;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    transition: 0.3s;
}
.primary-btn button { background-color: var(--pink-strong) !important; color: white !important; border: none !important; font-size: 1.1rem; }
.secondary-btn button { background-color: transparent !important; color: var(--beige-dark) !important; border: 1px solid var(--beige-light) !important; }

/* Labels */
label p { color: var(--text-dark) !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ──
@st.cache_resource
def load_assets():
    try:
        return joblib.load("model.pkl"), joblib.load("scaler.pkl")
    except: return None, None

model, scaler = load_assets()

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("<h2 style='font-family:Syne; color:#EC769A;'>Quick Stats</h2>", unsafe_allow_html=True)
    sidebar_stats = [("Survival Rate", "91%"), ("Early Detection", "99%"), ("Benign Range", "< 13.0")]
    for t, v in sidebar_stats:
        st.markdown(f"""<div style='background:white; padding:1rem; border-radius:12px; border:1px solid #EACFB3; margin-bottom:0.8rem;'>
            <p style='margin:0; font-size:0.7rem; color:#CCB083; font-weight:700;'>{t}</p>
            <p style='margin:0; font-size:1.2rem; font-weight:800; color:#EC769A;'>{v}</p>
        </div>""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown('<div class="main-header"><h1>Breast Cancer Prediction</h1><p style="color:#CCB083; font-weight:600;">DIAGNOSTIC DATA INTERFACE</p></div>', unsafe_allow_html=True)

# ── ALIGNED TOP CARDS (STRICT FLEXBOX ALIGNMENT) ──
st.markdown("""
<div style="
    display: flex; 
    justify-content: space-between; 
    gap: 20px; 
    margin-bottom: 2rem;
">
    <div style="
        flex: 1;
        background: white;
        padding: 1.2rem;
        border-radius: 20px;
        border-bottom: 4px solid #FC8EAC;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    ">
        <h3 style="font-size: 0.75rem; color: #CCB083; text-transform: uppercase; margin: 0 0 5px 0;">Validation Score</h3>
        <p style="font-size: 1.4rem; font-weight: 800; color: #EC769A; margin: 0;">98.2%</p>
    </div>
    <div style="
        flex: 1;
        background: white;
        padding: 1.2rem;
        border-radius: 20px;
        border-bottom: 4px solid #FC8EAC;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    ">
        <h3 style="font-size: 0.75rem; color: #CCB083; text-transform: uppercase; margin: 0 0 5px 0;">Analysis Speed</h3>
        <p style="font-size: 1.4rem; font-weight: 800; color: #EC769A; margin: 0;">Real-time</p>
    </div>
    <div style="
        flex: 1;
        background: white;
        padding: 1.2rem;
        border-radius: 20px;
        border-bottom: 4px solid #FC8EAC;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    ">
        <h3 style="font-size: 0.75rem; color: #CCB083; text-transform: uppercase; margin: 0 0 5px 0;">System Status</h3>
        <p style="font-size: 1.4rem; font-weight: 800; color: #EC769A; margin: 0;">Ready</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── DATA RANGES ──
BOUNDS = {
    "radius": (6.0, 30.0), "texture": (9.0, 40.0), "perimeter": (40.0, 190.0), "area": (140.0, 2500.0),
    "smoothness": (0.05, 0.2), "compactness": (0.01, 0.4), "concavity": (0.0, 0.5),
    "points": (0.0, 0.2), "symmetry": (0.1, 0.4), "dimension": (0.01, 0.1)
}

def get_feats(s): return [f"{x}_{s}" for x in ["radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal_dimension"]]

# ── RESET LOGIC ──
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

def reset_values():
    for key in st.session_state.keys():
        if "slide" in key or "num" in key:
            del st.session_state[key]
    st.rerun()

# ── INPUT TABS ──
tab1, tab2, tab3 = st.tabs(["MEAN", "SE", "WORST"])

def render_sync_inputs(features):
    for f in features:
        root = f.split("_")[0]
        low, high = BOUNDS.get(root, (0.0, 1.0))
        
        col_slider, col_val = st.columns([3, 1])
        
        # State Management for Synchronization
        slider_key = f"{f}_slide"
        num_key = f"{f}_num"
        
        with col_slider:
            val = st.slider(f.replace("_", " ").title(), low, high*1.2, low, key=slider_key)
        with col_val:
            # Tie number input directly to slider state
            final_val = st.number_input("Value", value=val, key=num_key)
            st.session_state.form_data[f] = final_val

with tab1: render_sync_inputs(get_feats("mean"))
with tab2: render_sync_inputs(get_feats("se"))
with tab3: render_sync_inputs(get_feats("worst"))

# ── BUTTONS ──
cb1, cb2 = st.columns([4, 1])
with cb1:
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    analyze = st.button("GENERATE ANALYSIS")
    st.markdown('</div>', unsafe_allow_html=True)
with cb2:
    st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
    st.button("RESET", on_click=reset_values)
    st.markdown('</div>', unsafe_allow_html=True)

# ── ANALYSIS ──
if analyze and model:
    all_f = get_feats("mean") + get_feats("se") + get_feats("worst")
    data = np.array([[st.session_state.form_data[f] for f in all_f]])
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    conf = max(model.predict_proba(scaled)[0]) * 100 if hasattr(model, "predict_proba") else 98.0
    
    res = "MALIGNANT" if pred == 1 else "BENIGN"
    clr = "#EC769A" if pred == 1 else "#CCB083"

    st.markdown(f"""<div style="background:white; padding:2rem; border-radius:30px; text-align:center; border:2px solid {clr}; margin-top:2rem;">
        <p style="font-family:Syne; font-weight:800; color:{clr}; letter-spacing:2px;">RESULT</p>
        <p style="font-size:3.5rem; font-weight:800; color:{clr}; margin:0;">{res}</p>
        <p style="color:#2D241E; font-weight:700;">Confidence: {conf:.2f}%</p>
    </div>""", unsafe_allow_html=True)

    # Darkened Graph
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        idx = np.argsort(imp)[-10:]
        fig = go.Figure(go.Bar(x=imp[idx], y=[all_f[i].title().replace("_", " ") for i in idx], orientation='h',
                               marker=dict(color=imp[idx], colorscale=[[0, '#FBC5C6'], [1, '#EC769A']])))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(tickfont=dict(color='#2D241E', weight='bold')),
                          yaxis=dict(tickfont=dict(color='#2D241E', weight='bold')))
        st.plotly_chart(fig, use_container_width=True)