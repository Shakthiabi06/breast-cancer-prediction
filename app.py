import streamlit as st
import pickle
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg-main: #fff7fa;
    --bg-card: #ffffff;

    --primary: #c48197;     /* main pink */
    --primary-dark: #b66681;
    --primary-soft: #f3d6df;

    --accent: #dfc1cb;      /* borders */

    --text-main: #3a2b2f;
    --text-muted: #7a5a63;

    --success: #66bb6a;
    --danger: #ef5350;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #fff7fa !important;
    color: #3a2b2f !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.block-container {
    padding: 2rem 3rem 6rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Hero ── */
.hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    background: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-radius: 4px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: #1565c0;
}
.hero-tag {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    color: #c48197;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--text-dark) !important;
    line-height: 1.05;
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem 0;
}
.hero-title span { color: var(--primary) !important; }
.hero-sub {
    font-size: 0.72rem;
    color: var(--soft) !important;
    letter-spacing: 0.05em;
    margin: 0;
}
.hero-badges { display: flex; gap: 0.75rem; }
.hero-badge {
    background: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-radius: 4px;
    padding: 0.9rem 1.5rem;
    text-align: center;
    min-width: 80px;
}
.hero-badge .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--primary) !important;
    display: block;
    letter-spacing: 0.05em;
}
.hero-badge .lbl {
    font-size: 0.55rem;
    color: var(--secondary) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-bottom: 1px solid var(--accent) !important;
    gap: 0 !important;
    padding: 0 !important;
    border-radius: 4px 4px 0 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 1rem 2rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--primary-soft) !important;
    color: var(--primary-dark) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-top: none !important;
    border-radius: 0 0 4px 4px !important;
    padding: 2rem !important;
}

/* ── Inputs ── */
.stNumberInput > div > div {
    background: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-radius: 6px !important;
    color: var(--text-dark) !important;
}

.stNumberInput input {
    color: #2a1f22 !important;
    background: #ffffff !important;
    font-weight: 500 !important;
}

.stNumberInput label {
    color: var(--secondary) !important;
    font-size: 0.7rem !important;
}

/* tooltip icon colour */
.stNumberInput label + div svg { fill: #c48197 !important; }

[data-testid="column"] { padding: 0 0.3rem !important; }

.tab-hint {
    font-size: 0.68rem;
    color: var(--text-muted) !important;
    border-left: 2px solid var(--primary) !important;
    background: #fff0f5;
    letter-spacing: 0.08em;
    margin-bottom: 1.5rem;
    padding: 0.6rem 1rem;
}

/* ── Predict button ── */
.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

.stButton > button:hover {
    background: var(--secondary) !important;
}
.stButton > button:active { transform: scale(0.99) !important; }

/* ── Result cards ── */
.result-benign {
    background: #f1fbf3;
    border-left: 4px solid var(--success);
    color: #4a2e35;
}

.result-malignant {
    background: #fdecea;
    border-left: 4px solid var(--danger);
    color: #4a2e35;
}
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
}
.result-desc {
    font-size: 0.7rem;
    color: #6a8aaa;
    margin-top: 0.6rem;
    letter-spacing: 0.03em;
    line-height: 1.7;
}
.result-confidence {
    display: inline-block;
    margin-top: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-dark);
    letter-spacing: 0.1em;
    padding: 0.25rem 0.6rem;
    border: 1px solid #c48197;
    border-radius: 2px;
}

/* ── Chart section header ── */
.chart-head {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1rem 0;
}
.chart-head .bar { width: 3px; height: 16px; background: var(--primary); border-radius: 2px; }
.chart-head .title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--primary-dark);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.chart-head .sub {
    margin-left: auto;
    font-size: 0.6rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
}

/* ── Warning / Disclaimer ── */
.warn-banner {
    background: #1a0e02;
    border: 1px solid #4a2800;
    border-left: 4px solid #e65100;
    border-radius: 4px;
    padding: 0.9rem 1.5rem;
    font-size: 0.72rem;
    color: #bf8040;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}
.disclaimer {
    background: #fff0f5 !important;
    border: 1px solid var(--accent) !important;
    color: var(--text-dark) !important;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.5rem;
    font-size: 0.65rem;
    letter-spacing: 0.04em;
    line-height: 1.8;
}

/* ── Bottom bar ── */
.bottom-bar {
    background: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-radius: 4px;
    padding: 1.2rem 2rem;
    margin-top: 2rem;
}
.bottom-bar-label {
    font-size: 0.65rem;
    color: var(--secondary) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #040810; }
::-webkit-scrollbar-thumb { background: #0d2a4a; border-radius: 2px; }
            
@media (max-width: 768px) {
    .hero {
        flex-direction: column !important;
        gap: 1rem;
        padding: 1.5rem !important;
    }

    .hero-title {
        font-size: 1.6rem !important;
        text-align: center;
    }

    .hero-badges {
        flex-direction: column !important;
        width: 100%;
    }

    .block-container {
        padding: 1rem !important;
    }
}
            
/* ───── FORCE REMOVE STREAMLIT BLUE THEME ───── */

/* All text */
body, p, span, div, label {
    color: var(--text-main) !important;
}

/* Fix remaining blue texts */
.tab-hint,
.chart-head .title,
.chart-head .sub {
    color: var(--secondary) !important;
}

/* Remove blue hover backgrounds */
.stTabs [data-baseweb="tab"]:hover {
    background: #fff0f5 !important;
    color: var(--primary) !important;
}

/* ───── BUTTON FIX (IMPORTANT) ───── */
div[data-testid="stButton"] button {
    background-color: var(--primary) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
}

div[data-testid="stButton"] button:hover {
    background: var(--primary-dark) !important;
}

/* ───── INPUT BOX FIX ───── */
div[data-baseweb="input"] {
    background-color: white !important;
    border-color: var(--accent) !important;
}

/* ───── TOOLTIP ICON FIX ───── */
svg {
    fill: var(--secondary) !important;
}

/* ───── REMOVE DARK PATCHES ───── */
[data-testid="stAppViewContainer"] {
    background-color: #fff7fa !important;
}
</style>
""", unsafe_allow_html=True)




# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model  = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()


# ── Feature definitions: (label, min, max, default, tooltip) ─────────────────
MEAN_FEATURES = {
    "radius_mean":            ("Radius Mean",            6.98,  28.11,  14.13, "Mean distance from center to edge of the tumor nucleus."),
    "texture_mean":           ("Texture Mean",           9.71,  39.28,  19.29, "Variation in gray-scale intensity — measures surface roughness."),
    "perimeter_mean":         ("Perimeter Mean",         43.79, 188.5,  91.97, "Mean length of the tumor nucleus boundary."),
    "area_mean":              ("Area Mean",              143.5, 2501.0, 654.89,"Mean cross-sectional area of the tumor nucleus."),
    "smoothness_mean":        ("Smoothness Mean",        0.053, 0.163,  0.096, "Local variation in radius lengths — smoother = more uniform boundary."),
    "compactness_mean":       ("Compactness Mean",       0.019, 0.345,  0.104, "Perimeter² / area − 1. Higher values mean more irregular shape."),
    "concavity_mean":         ("Concavity Mean",         0.0,   0.427,  0.089, "Measures depth of inward curves (concave portions) in the tumor boundary."),
    "concave points_mean":    ("Concave Points Mean",    0.0,   0.201,  0.049, "Number of concave (indented) points on the nucleus boundary."),
    "symmetry_mean":          ("Symmetry Mean",          0.106, 0.304,  0.181, "How symmetrical the nucleus is. Malignant cells tend to be less symmetric."),
    "fractal_dimension_mean": ("Fractal Dimension Mean", 0.05,  0.097,  0.063, "Coastline approximation — complexity of the boundary. Higher = more irregular."),
}
SE_FEATURES = {
    "radius_se":              ("Radius SE",              0.112, 2.873,  0.405, "Standard error of radius — how much radius varies across nuclei."),
    "texture_se":             ("Texture SE",             0.36,  4.885,  1.217, "Standard error of texture — variability of surface roughness."),
    "perimeter_se":           ("Perimeter SE",           0.757, 21.98,  2.866, "Standard error of perimeter measurement across nuclei."),
    "area_se":                ("Area SE",                6.802, 542.2,  40.34, "Standard error of nucleus area — spread in size across cells."),
    "smoothness_se":          ("Smoothness SE",          0.002, 0.031,  0.007, "Standard error of smoothness — how consistently smooth the boundaries are."),
    "compactness_se":         ("Compactness SE",         0.002, 0.135,  0.025, "Standard error of compactness — variability of shape irregularity."),
    "concavity_se":           ("Concavity SE",           0.0,   0.396,  0.032, "Standard error of concavity depth across nuclei."),
    "concave points_se":      ("Concave Points SE",      0.0,   0.053,  0.012, "Standard error of number of concave points on the boundary."),
    "symmetry_se":            ("Symmetry SE",            0.008, 0.079,  0.021, "Standard error of symmetry — consistency of shape symmetry."),
    "fractal_dimension_se":   ("Fractal Dimension SE",   0.001, 0.03,   0.004, "Standard error of fractal dimension — boundary complexity variability."),
}
WORST_FEATURES = {
    "radius_worst":           ("Radius Worst",           7.93,  36.04,  16.27, "Largest (worst) radius value among all nuclei in the sample."),
    "texture_worst":          ("Texture Worst",          12.02, 49.54,  25.68, "Largest texture value — most extreme surface roughness observed."),
    "perimeter_worst":        ("Perimeter Worst",        50.41, 251.2,  107.26,"Largest perimeter value — most irregular boundary length observed."),
    "area_worst":             ("Area Worst",             185.2, 4254.0, 880.58,"Largest nucleus area observed in the sample."),
    "smoothness_worst":       ("Smoothness Worst",       0.071, 0.223,  0.132, "Worst (most irregular) smoothness value across all nuclei."),
    "compactness_worst":      ("Compactness Worst",      0.027, 1.058,  0.254, "Worst compactness — most extreme shape irregularity observed."),
    "concavity_worst":        ("Concavity Worst",        0.0,   1.252,  0.272, "Deepest inward curves observed — worst concavity in the sample."),
    "concave points_worst":   ("Concave Points Worst",   0.0,   0.291,  0.115, "Highest number of concave boundary points across all nuclei."),
    "symmetry_worst":         ("Symmetry Worst",         0.156, 0.664,  0.29,  "Most asymmetric nucleus observed — worst symmetry in the sample."),
    "fractal_dimension_worst":("Fractal Dimension Worst",0.055, 0.208,  0.084, "Most complex boundary observed — worst fractal dimension in sample."),
}
ALL_FEATURES = {**MEAN_FEATURES, **SE_FEATURES, **WORST_FEATURES}


# ── Feature importance chart ──────────────────────────────────────────────────
def plot_feature_importance(model, feature_names, prediction):
    # coef_ shape differs: LinearSVC → (1, n) or (n,), SVC → (1, n)
    try:
        coef = model.coef_
        if hasattr(coef, "toarray"):
            coef = coef.toarray()
        coef = np.array(coef).flatten()
    except AttributeError:
        # model has no coef_ (e.g. RBF SVM) — skip chart
        return None

    coeffs     = coef
    abs_coeffs = np.abs(coeffs)
    top_idx    = np.argsort(abs_coeffs)[-10:][::-1]   # top 10, highest first

    top_names  = [ALL_FEATURES[feature_names[i]][0] for i in top_idx]
    top_vals   = abs_coeffs[top_idx]
    top_signs  = coeffs[top_idx]   # positive → increases cancer risk

    n = len(top_vals)
    max_val = top_vals.max()

    # ── Gradient colors based on importance rank ──────────────────────────────
    # Benign: green palette  |  Malignant: red palette
    if prediction == 0:
        high_col = np.array([0.267, 0.627, 0.278])   # #44a047
        low_col  = np.array([0.102, 0.263, 0.122])   # #1a431f
    else:
        high_col = np.array([0.937, 0.325, 0.314])   # #ef5350
        low_col  = np.array([0.400, 0.082, 0.082])   # #661515

    colors = []
    for i in range(n):
        t = 1.0 - (i / max(n - 1, 1)) * 0.75   # 1.0 → 0.25 (most to least important)
        c = high_col * t + low_col * (1 - t)
        colors.append(tuple(c))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor("#fff7fa")
    ax.set_facecolor("#ffffff")

    # draw bars in reversed order (most important at top)
    rev_names  = top_names[::-1]
    rev_vals   = top_vals[::-1]
    rev_signs  = top_signs[::-1]
    rev_colors = colors[::-1]

    bars = ax.barh(rev_names, rev_vals, color=rev_colors, height=0.52, edgecolor="none")

    # ── Highlight top feature (last bar = index -1 in reversed list = top feature) ──
    top_bar = bars[-1]
    top_bar.set_linewidth(1.2)
    top_bar.set_edgecolor("#ffffff44")
    # glow effect: draw a slightly wider translucent bar behind it
    ax.barh([rev_names[-1]], [rev_vals[-1]], color=rev_colors[-1],
            height=0.72, alpha=0.25, edgecolor="none", zorder=1)
    bars[-1].set_zorder(3)

    # ── Direction label (+/−) and impact label ────────────────────────────────
    for bar, val, sign, name in zip(bars, rev_vals, rev_signs, rev_names):
        direction = "+ Risk" if sign > 0 else "− Risk"
        dir_color = "#ef9a9a" if sign > 0 else "#90caf9"
        ratio = val / max_val
        impact = "High" if ratio >= 0.66 else ("Moderate" if ratio >= 0.33 else "Low")

        # direction tag
        ax.text(val + max_val * 0.012, bar.get_y() + bar.get_height() / 2,
                direction, va="center", ha="left",
                fontsize=7, color=dir_color, fontfamily="monospace", fontweight="bold")

        # impact tag (further right)
        ax.text(val + max_val * 0.13, bar.get_y() + bar.get_height() / 2,
                impact, va="center", ha="left",
                fontsize=6.5, color="#7a5a63", fontfamily="monospace")

    # ── Top feature bold y-tick label ─────────────────────────────────────────
    tick_labels = ax.get_yticklabels()
    ax.set_yticklabels(rev_names)   # re-set so we can style last one
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == rev_names[-1]:
            lbl.set_fontweight("bold")
            lbl.set_color("#e8f0fe")
        else:
            lbl.set_color("#7a5a63")

    ax.set_xlabel("Coefficient Magnitude", color="#3a2b2f")
    ax.tick_params(axis="both", labelsize=7.5, length=0)
    ax.tick_params(axis="x", colors="#5a3d45")
    for spine in ax.spines.values():
        spine.set_edgecolor("#0d2a4a")
    ax.set_xlim(0, max_val * 1.45)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color="#0a1f38", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout(pad=1.0)
    return fig


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div>
        <h1 class="hero-title">Breast Cancer <span>Detection</span></h1>
        <p class="hero-sub">Wisconsin Dataset · 30-Feature Analysis · Logistic Regression</p>
    </div>
    <div class="hero-badges">
        <div class="hero-badge"><span class="num">30 features</span><span class="lbl">Input Parameters</span></div>
        <div class="hero-badge"><span class="num">569 samples</span><span class="lbl">Training Data</span></div>
        <div class="hero-badge"><span class="num">97% accuracy</span><span class="lbl">Model Score</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.markdown("""
    <div class="warn-banner">
        ⚠ &nbsp; <strong>trained_models.pkl</strong> or <strong>scaler.pkl</strong> not found.
        Predictions unavailable until model files are placed in the app folder.
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
input_values = {}

tab1, tab2, tab3 = st.tabs([
    "◈  Mean Features",
    "◇  Standard Error",
    "◆  Worst Features",
])

def render_inputs(features, hint):
    st.markdown(f'<div class="tab-hint">{hint}</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (key, (label, mn, mx, default, tooltip)) in enumerate(features.items()):
        with cols[i % 5]:
            input_values[key] = st.number_input(
                label,
                min_value=float(mn),
                max_value=float(mx) * 2,
                value=float(default),
                format="%.5f",
                key=key,
                help=tooltip,
            )

with tab1:
    render_inputs(MEAN_FEATURES, "Average values computed from the cell nuclei in the image.")
with tab2:
    render_inputs(SE_FEATURES,   "Standard error — measures variability of each feature across nuclei.")
with tab3:
    render_inputs(WORST_FEATURES,"Largest (worst) values for each feature across all nuclei.")


# ── Bottom bar + predict button ───────────────────────────────────────────────
st.markdown("""
<div class="bottom-bar">
    <div class="bottom-bar-label">Fill all 3 tabs · Then click predict</div>
</div>
""", unsafe_allow_html=True)

col_btn, _ = st.columns([1, 4])
with col_btn:
    predict_clicked = st.button("⬡  Run Prediction", use_container_width=True)


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_clicked:
    feature_order = list(ALL_FEATURES.keys())
    input_array   = np.array([[input_values[f] for f in feature_order]])

    if model is not None and scaler is not None:
        input_scaled = scaler.transform(input_array)
        prediction   = model.predict(input_scaled)[0]

        # Linear SVM (LinearSVC) has no predict_proba — use decision_function instead
        try:
            proba      = model.predict_proba(input_scaled)[0]
            confidence = max(proba) * 100
        except (AttributeError, Exception):
            score      = model.decision_function(input_scaled)[0]
            confidence = min(99.0, 50.0 + abs(float(score)) * 15.0)

        # result + chart side by side
        col_res, col_chart = st.columns([1, 1.4])

        with col_res:
            if prediction == 0:
                st.markdown(f"""
                <div class="result-benign">
                    <div class="result-label" style="color:#4caf50;">✓ &nbsp; Prediction Result</div>
                    <p class="result-value" style="color:#66bb6a;">Benign</p>
                    <p class="result-desc">
                        The entered measurements suggest a <strong style="color:#81c784;">non-cancerous</strong> mass.
                        Benign tumors do not invade nearby tissue or spread elsewhere.
                        Always consult a medical professional for confirmed diagnosis.
                    </p>
                    <span class="result-confidence">Model Confidence · {confidence:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-malignant">
                    <div class="result-label" style="color:#ef5350;">⚠ &nbsp; Prediction Result</div>
                    <p class="result-value" style="color:#ef5350;">Malignant</p>
                    <p class="result-desc">
                        The entered measurements indicate possible <strong style="color:#e57373;">malignancy</strong>.
                        This is a screening aid only — immediate consultation with an oncologist
                        is strongly recommended for imaging and biopsy confirmation.
                    </p>
                    <span class="result-confidence">Model Confidence · {confidence:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

        with col_chart:
            st.markdown("""
            <div class="chart-head">
                <div class="bar"></div>
                <div class="title">Top Features Influencing This Prediction</div>
            </div>
            """, unsafe_allow_html=True)
            fig = plot_feature_importance(model, feature_order, prediction)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown("""
                <div style="font-size:0.62rem; color:#2e5a8a; letter-spacing:0.06em;
                            margin-top:0.4rem; padding-left:0.25rem;">
                    These features had the strongest influence on the model's decision.
                    + Risk increases cancer likelihood · − Risk decreases it.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="font-size:0.68rem; color:#2e5a8a; padding:1rem;
                            border:1px solid #0d2a4a; border-radius:4px; margin-top:0.5rem;">
                    Feature importance chart not available for this model type.
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-malignant" style="border-left-color:#e65100;border-color:#4a2800;background:#1a0e02;">
            <div class="result-label" style="color:#ff8a65;">⚠ &nbsp; Model Not Loaded</div>
            <p class="result-value" style="color:#ff7043;font-size:1.2rem;">No model file found</p>
            <p class="result-desc">
                Place <strong>model.pkl</strong> and <strong>scaler.pkl</strong>
                in the same folder as app.py and restart.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠ &nbsp; <strong>Disclaimer</strong> — This tool is for educational and research purposes only.
        It is not a substitute for professional medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)
