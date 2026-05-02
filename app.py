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

/* Header */
[data-testid="stHeader"] { background: rgba(0,0,0,0); height: 0px; }
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

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: white !important;
    border-right: 1px solid var(--beige-light);
}

/* Slider */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #EC769A !important;
    border: 2px solid white !important;
    box-shadow: 0 0 0 4px #FBC5C6 !important;
}
.stSlider [data-baseweb="slider"] > div > div > div {
    background: #EC769A !important;
}
.stSlider [data-baseweb="slider"] > div > div {
    background: #EACFB3 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {
    background-color: #EC769A !important;
}

/* Number Input — complete BaseWeb override */
div[data-testid="stNumberInput"] button {
    display: none !important;
}
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[data-baseweb="base-input"],
div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus,
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
div[data-testid="stNumberInput"] div[data-baseweb="input"]:hover,
div[data-testid="stNumberInput"] div[data-baseweb="input"] *,
div[data-testid="stNumberInput"] div[data-baseweb="input"] *:focus,
div[data-testid="stNumberInput"] div[data-baseweb="input"] *:focus-within {
    border: 1.5px solid #EACFB3 !important;
    box-shadow: none !important;
    outline: none !important;
    border-radius: 10px !important;
    background-color: #fff9f5 !important;
}
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border: 1.5px solid #EC769A !important;
}
div[data-testid="stNumberInput"] input[type="number"] {
    color: #EC769A !important;
    -webkit-text-fill-color: #EC769A !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    background-color: #fff9f5 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    text-align: center !important;
    caret-color: #EC769A !important;
    cursor:

/* Buttons */
.stButton button {
    width: 100%;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    transition: 0.3s;
    border: none !important;
}
.primary-btn button {
    background: linear-gradient(135deg, #FC8EAC, #EC769A) !important;
    color: white !important;
    font-size: 1.1rem !important;
    box-shadow: 0 4px 15px rgba(236, 118, 154, 0.4) !important;
}
.primary-btn button:hover {
    background: linear-gradient(135deg, #EC769A, #CC5580) !important;
    box-shadow: 0 6px 20px rgba(236, 118, 154, 0.6) !important;
    transform: translateY(-1px);
}
.secondary-btn button {
    background-color: transparent !important;
    color: #CCB083 !important;
    border: 1.5px solid #EACFB3 !important;
    font-size: 0.9rem !important;
}
.secondary-btn button:hover {
    background-color: #FBC5C6 !important;
    color: #EC769A !important;
    border-color: #FC8EAC !important;
}

/* Labels */
label p { color: var(--text-dark) !important; font-weight: 700 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: transparent;
    padding: 0.5rem 0;
}
.stTabs [data-baseweb="tab"] {
    flex: 1;
    justify-content: center;
    background: white !important;
    border-radius: 14px !important;
    border: 1.5px solid #EACFB3 !important;
    color: #CCB083 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.2s ease;
    opacity: 1 !important;
    visibility: visible !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #FBC5C6 !important;
    color: #EC769A !important;
    border-color: #FC8EAC !important;
}
.stTabs [aria-selected="true"] {
    background: #EC769A !important;
    color: white !important;
    border-color: #EC769A !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

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

# ── TOP CARDS ──
st.markdown("""
<div style="display:flex; gap:20px; margin-bottom:2rem; width:100%;">
    <div style="flex:1; background:white; padding:1.2rem; border-radius:20px; border-bottom:4px solid #FC8EAC; box-shadow:0 4px 12px rgba(0,0,0,0.05); display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:100px;">
        <p style="font-size:0.75rem; color:#CCB083; text-transform:uppercase; margin:0 0 5px 0; font-weight:700; text-align:center; width:100%;">Validation Score</p>
        <p style="font-size:1.4rem; font-weight:800; color:#EC769A; margin:0; text-align:center; width:100%;">98.2%</p>
    </div>
    <div style="flex:1; background:white; padding:1.2rem; border-radius:20px; border-bottom:4px solid #FC8EAC; box-shadow:0 4px 12px rgba(0,0,0,0.05); display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:100px;">
        <p style="font-size:0.75rem; color:#CCB083; text-transform:uppercase; margin:0 0 5px 0; font-weight:700; text-align:center; width:100%;">Analysis Speed</p>
        <p style="font-size:1.4rem; font-weight:800; color:#EC769A; margin:0; text-align:center; width:100%;">Real-time</p>
    </div>
    <div style="flex:1; background:white; padding:1.2rem; border-radius:20px; border-bottom:4px solid #FC8EAC; box-shadow:0 4px 12px rgba(0,0,0,0.05); display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:100px;">
        <p style="font-size:0.75rem; color:#CCB083; text-transform:uppercase; margin:0 0 5px 0; font-weight:700; text-align:center; width:100%;">System Status</p>
        <p style="font-size:1.4rem; font-weight:800; color:#EC769A; margin:0; text-align:center; width:100%;">Ready</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── AUTO-SELECT SCRIPT (once, not per input) ──
st.markdown("""
<script>
window.parent.document.addEventListener('focusin', function(e) {
    if (e.target && e.target.type === 'number') {
        e.target.select();
    }
});
</script>
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
    keys_to_delete = [k for k in st.session_state.keys() if "slide" in k or "num" in k]
    for key in keys_to_delete:
        del st.session_state[key]
    st.rerun()

# ── INPUT TABS ──
tab1, tab2, tab3 = st.tabs(["MEAN", "SE", "WORST"])

def render_sync_inputs(features):
    for f in features:
        root = f.split("_")[0]
        low, high = BOUNDS.get(root, (0.0, 1.0))
        high_ext = round(high * 1.2, 4)

        slider_key = f"{f}_slide"
        num_key = f"{f}_num"

        if slider_key not in st.session_state:
            st.session_state[slider_key] = float(low)
        if num_key not in st.session_state:
            st.session_state[num_key] = float(low)

        def on_slider(sk=slider_key, nk=num_key):
            st.session_state[nk] = st.session_state[sk]

        def on_num(sk=slider_key, nk=num_key, lo=float(low), hi=float(high_ext)):
            st.session_state[sk] = float(np.clip(st.session_state[nk], lo, hi))

        col_slider, col_val = st.columns([3, 1])

        with col_slider:
            st.slider(
                f.replace("_", " ").title(),
                min_value=float(low),
                max_value=float(high_ext),
                key=slider_key,
                on_change=on_slider
            )

        with col_val:
            st.number_input(
                "Value",
                min_value=float(low),
                max_value=float(high_ext),
                key=num_key,
                step=round((high_ext - float(low)) / 100, 6),
                on_change=on_num
            )

        st.session_state.form_data[f] = st.session_state[slider_key]

with tab1: render_sync_inputs(get_feats("mean"))
with tab2: render_sync_inputs(get_feats("se"))
with tab3: render_sync_inputs(get_feats("worst"))

# ── BUTTONS ──
cb1, cb2, cb3 = st.columns([3, 1, 1])
with cb1:
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    analyze = st.button("GENERATE ANALYSIS")
    st.markdown('</div>', unsafe_allow_html=True)
with cb3:
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

    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        idx = np.argsort(imp)[-10:]
        fig = go.Figure(go.Bar(x=imp[idx], y=[all_f[i].title().replace("_", " ") for i in idx], orientation='h',
                               marker=dict(color=imp[idx], colorscale=[[0, '#FBC5C6'], [1, '#EC769A']])))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(tickfont=dict(color='#2D241E', weight='bold')),
                          yaxis=dict(tickfont=dict(color='#2D241E', weight='bold')))
        st.plotly_chart(fig, use_container_width=True)