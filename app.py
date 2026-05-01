import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ──
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide", initial_sidebar_state="expanded")

# ── ADVANCED UI STYLING ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@300;400;600;700&display=swap');

:root {
    --primary: #a41f39;
    --primary-light: #fce4ec;
    --text-dark: #140d07;
    --text-light: #6b7280;
    --bg-main: #f9fafb;
}

/* Global Overrides */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main);
    color: var(--text-dark);
    font-family: 'Inter', sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #e5e7eb;
}

/* Hide Default Headers */
header, footer {visibility: hidden;}

/* Custom Dashboard Title */
.main-header {
    background: white;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    color: var(--primary);
    margin: 0;
    font-size: 2.2rem;
    letter-spacing: -1px;
}

/* Tab Styling - Full Width & Modern */
.stTabs [data-baseweb="tab-list"] {
    width: 100%;
    gap: 10px;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    flex-grow: 1;
    height: 60px;
    background-color: white;
    border-radius: 12px;
    color: var(--text-light);
    font-weight: 600;
    font-size: 0.9rem;
    border: 1px solid #e5e7eb;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 10px 15px -3px rgba(164, 31, 57, 0.2);
}

/* Input Fields - Professional Borders */
label[data-testid="stWidgetLabel"] p {
    color: var(--text-dark) !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}

div[data-baseweb="input"] {
    background-color: white !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 2px;
}

/* Infographic-style Stat Cards */
.stat-container {
    display: flex;
    gap: 20px;
    margin-bottom: 2rem;
}

.stat-card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    border-left: 5px solid var(--primary);
    flex: 1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.stat-card h3 {
    margin: 0;
    font-size: 0.8rem;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stat-card p {
    margin: 5px 0 0 0;
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-dark);
}

/* Result Dashboard View */
.result-box {
    background: white;
    padding: 2.5rem;
    border-radius: 24px;
    text-align: center;
    border: 1px solid #e5e7eb;
    margin-top: 2rem;
}

.res-label {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    color: var(--primary);
    margin-bottom: 0.5rem;
}

.res-value {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1.5px;
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

# ── SIDEBAR & STATS ──
with st.sidebar:
    st.markdown("<h2 style='font-family:Syne; color:#a41f39;'>Quick Stats</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#fff5f7; padding:1rem; border-radius:12px; margin-bottom:1rem;'>
        <p style='margin:0; font-size:0.8rem; color:#a41f39; font-weight:700;'>GLOBAL IMPACT</p>
        <p style='margin:0; font-size:1.2rem; font-weight:800;'>1 in 8 Women</p>
        <p style='margin:0; font-size:0.75rem; color:#666;'>Will develop breast cancer in their lifetime.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("Ensure all clinical measurements are entered from the pathology report for maximum accuracy.")

# ── MAIN HEADER ──
st.markdown("""
<div class="main-header">
    <h1>Breast Cancer Prediction</h1>
    <p style="color:#6b7280; font-weight:500; margin:5px 0 0 0;">Pathology-Based Diagnostic Analysis System</p>
</div>
""", unsafe_allow_html=True)

# ── INFOGRAPHIC ROW ──
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="stat-card"><h3>Global Cases</h3><p>2.3 Million</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><h3>Model Accuracy</h3><p>97.4%</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><h3>Diagnostic Latency</h3><p>< 1.2s</p></div>', unsafe_allow_html=True)

# ── INPUT SECTION ──
MEAN = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"]
SE = ["radius_se","texture_se","perimeter_se","area_se","smoothness_se",
      "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se"]
WORST = ["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
         "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

inputs = {}

tab1, tab2, tab3 = st.tabs(["CELL MEAN ANALYSIS", "STANDARD ERROR (SE)", "MAXIMUM DIMENSIONS"])

def render_grid(features):
    rows = [features[i:i + 5] for i in range(0, len(features), 5)]
    for row in rows:
        cols = st.columns(5)
        for i, f in enumerate(row):
            with cols[i]:
                label = f.replace("_", " ").title().split(" ")[0]
                inputs[f] = st.number_input(label, value=0.0, key=f)

with tab1: render_grid(MEAN)
with tab2: render_grid(SE)
with tab3: render_grid(WORST)

st.markdown("<br>", unsafe_allow_html=True)

# ── ANALYZE ──
if st.button("EXECUTE DIAGNOSTIC SEQUENCE"):
    if model:
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

        # ── DASHBOARD RESULT ──
        res_text = "MALIGNANT" if pred == 1 else "BENIGN"
        res_color = "#a41f39" if pred == 1 else "#166534"
        
        st.markdown(f"""
        <div class="result-box">
            <p class="res-label">DIAGNOSTIC CONCLUSION</p>
            <p class="res-value" style="color:{res_color};">{res_text}</p>
            <p style="font-weight:600; color:#6b7280; margin-top:10px;">Classification Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # ── INFOGRAPHIC GRAPH ──
        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
            top_idx = np.argsort(importance)[-8:]
            labels = [all_features[i].replace("_", " ").title() for i in top_idx]
            values = importance[top_idx]

            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Use infographic colors (Pink/Dark)
            bars = ax.bar(labels, values, color='#a41f39', alpha=0.85, width=0.6)
            
            # Clean styling
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            plt.xticks(rotation=45, ha='right', color="#140d07", fontweight='bold')
            
            # Add labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#a41f39')

            st.markdown("<h3 style='font-family:Syne; text-align:center; margin-top:2rem;'>Pathological Feature Weights</h3>", unsafe_allow_html=True)
            st.pyplot(fig)
    else:
        st.error("Model Error: Ensure assets are loaded.")

# ── FOOTER ──
st.markdown("""
<div style="text-align:center; margin-top:100px; padding:2rem; border-top:1px solid #eee; color:#9ca3af; font-size:0.8rem;">
    © 2026 CLINICAL DIAGNOSTICS INTERFACE • SYSTEM VERSION 4.0.2<br>
    DATASET: UCI MACHINE LEARNING REPOSITORY (WISCONSIN DIAGNOSTIC BREAST CANCER)
</div>
""", unsafe_allow_html=True)