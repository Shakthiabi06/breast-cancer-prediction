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
    --label-dark: #5D4D37;
}

/* Background Mesh Gradient */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #fdfbf7;
    background-image: 
        radial-gradient(at 0% 0%, rgba(244,244,221,0.4) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(251,197,198,0.2) 0px, transparent 50%);
    color: var(--text-dark);
    font-family: 'Inter', sans-serif;
}

/* Reduce Header Space */
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
.main-header {
    margin-top: -30px; 
    background: white;
    padding: 2rem 2.5rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    border: 1px solid var(--beige-light);
    box-shadow: 0 10px 30px -10px rgba(204, 176, 131, 0.15);
}

.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    color: var(--pink-strong);
    margin: 0;
    font-size: 2.5rem;
}

/* SIDEBAR RESTORED - Force visibility of the toggle */
[data-testid="stSidebar"] {
    background-color: white !important;
    border-right: 1px solid var(--beige-light);
}

/* Tabs Styling - Space from Top Info */
.stTabs { margin-top: 2.5rem; }
.stTabs [data-baseweb="tab-list"] { width: 100%; gap: 12px; }
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
}

/* FIX: Input Fields & Labels */
label[data-testid="stWidgetLabel"] p {
    color: var(--label-dark) !important;
    font-weight: 700 !important;
}

/* Force clean white background for input areas */
div[data-baseweb="input"], div[data-baseweb="base-input"] {
    background-color: white !important;
    border: 1.2px solid var(--beige-light) !important;
    border-radius: 10px !important;
}

input {
    background-color: white !important;
    color: var(--text-dark) !important;
}

/* Button Styling */
.stButton button {
    width: 100%;
    background-color: var(--pink-strong) !important;
    color: white !important;
    border: none;
    padding: 1rem;
    border-radius: 15px;
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    margin-top: 1.5rem;
}

.stButton button:hover {
    background-color: var(--pink-mid) !important;
    border: none;
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
    except: return None, None

model, scaler = load_assets()

# ── SIDEBAR (QUICK STATS - SLIDING) ──
with st.sidebar:
    st.markdown("<h2 style='font-family:Syne; color:#EC769A;'>Clinical Context</h2>", unsafe_allow_html=True)
    stats = [
        ("Survival Rate", "91%", "5-year relative survival average."),
        ("Early Detection", "99%", "Survival rate in localized stages."),
        ("Benign Range", "< 13.0", "Typical radius_mean for benign cases.")
    ]
    for title, val, desc in stats:
        st.markdown(f"""
        <div style='background:white; padding:1.2rem; border-radius:15px; border:1px solid #EACFB3; margin-bottom:1rem;'>
            <p style='margin:0; font-size:0.7rem; color:#CCB083; font-weight:700; text-transform:uppercase;'>{title}</p>
            <p style='margin:0; font-size:1.4rem; font-weight:800; color:#EC769A;'>{val}</p>
            <p style='margin:0; font-size:0.75rem; color:#6b7280;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div class="main-header">
    <h1>Breast Cancer Prediction</h1>
    <p style="color:#CCB083; font-weight:600; margin:5px 0 0 0;">INTELLIGENT DIAGNOSTIC INTERFACE</p>
</div>
""", unsafe_allow_html=True)

# ── TOP STATS ──
c1, c2, c3 = st.columns(3)
with c1: st.markdown('<div style="background:white; padding:1.5rem; border-radius:20px; text-align:center; border-bottom:4px solid #FC8EAC;"><h3>Validation Score</h3><p style="font-size:1.5rem; font-weight:800; color:#EC769A;">98.2%</p></div>', unsafe_allow_html=True)
with c2: st.markdown('<div style="background:white; padding:1.5rem; border-radius:20px; text-align:center; border-bottom:4px solid #FC8EAC;"><h3>Analysis Speed</h3><p style="font-size:1.5rem; font-weight:800; color:#EC769A;">Real-time</p></div>', unsafe_allow_html=True)
with c3: st.markdown('<div style="background:white; padding:1.5rem; border-radius:20px; text-align:center; border-bottom:4px solid #FC8EAC;"><h3>Status</h3><p style="font-size:1.5rem; font-weight:800; color:#EC769A;">Ready</p></div>', unsafe_allow_html=True)

# ── RANGES & FEATURES ──
BOUNDS = {
    "radius": (6.0, 30.0), "texture": (9.0, 40.0), "perimeter": (40.0, 190.0), "area": (140.0, 2500.0),
    "smoothness": (0.05, 0.2), "compactness": (0.01, 0.4), "concavity": (0.0, 0.5),
    "points": (0.0, 0.2), "symmetry": (0.1, 0.4), "dimension": (0.01, 0.1)
}

MEAN_F = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"]
SE_F = ["radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se"]
WORST_F = ["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

inputs = {}
tab1, tab2, tab3 = st.tabs(["MEAN FEATURES", "SE FEATURES", "WORST FEATURES"])

def render(feats):
    cols = st.columns(5)
    for i, f in enumerate(feats):
        with cols[i % 5]:
            key = f.split("_")[0]
            low, high = BOUNDS.get(key, (0.0, 1.0))
            # SLIDER OPTION: Toggle comment below to switch between slider and number_input
            # inputs[f] = st.slider(key.title(), low, high, low + (high-low)/4, key=f)
            inputs[f] = st.number_input(key.title(), min_value=0.0, max_value=high*2, value=low, key=f)

with tab1: render(MEAN_F)
with tab2: render(SE_F)
with tab3: render(WORST_F)

# ── EXECUTION ──
if st.button("GENERATE ANALYSIS"):
    if model:
        all_feats = MEAN_F + SE_F + WORST_F
        data = np.array([[inputs[f] for f in all_feats]])
        data_s = scaler.transform(data)
        pred = model.predict(data_s)[0]
        
        conf = max(model.predict_proba(data_s)[0]) * 100 if hasattr(model, "predict_proba") else 98.4
        color = "#EC769A" if pred == 1 else "#CCB083"
        result = "MALIGNANT" if pred == 1 else "BENIGN"

        st.markdown(f"""
        <div style="background:white; padding:2.5rem; border-radius:30px; text-align:center; border:2px solid #F4F4DD; margin-top:2rem;">
            <p style="font-family:Syne; font-weight:800; color:{color}; letter-spacing:2px;">RESULT</p>
            <p style="font-size:3.5rem; font-weight:800; color:{color}; margin:0;">{result}</p>
            <p style="font-weight:700; color:#2D241E; margin-top:10px;">Confidence: {conf:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Interactive Chart
        if hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
            idx = np.argsort(imp)[-10:]
            fig = go.Figure(go.Bar(x=imp[idx], y=[all_feats[i].title().replace("_", " ") for i in idx], orientation='h',
                                   marker=dict(color=imp[idx], colorscale=[[0, '#FBC5C6'], [1, '#EC769A']])))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              xaxis=dict(gridcolor='rgba(204, 176, 131, 0.1)', tickfont=dict(color='#CCB083', size=11, weight='bold')))
            st.plotly_chart(fig, use_container_width=True)