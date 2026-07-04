import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# ADVANCED CUSTOM CSS (SaaS Dashboard Aesthetic)
# -------------------------------------------------------
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Hero Banner */
    .hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #3B82F6 100%);
        border-radius: 24px;
        padding: 48px 32px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero h1 {
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: -0.05em;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #FFFFFF, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero p {
        font-size: 1.2rem;
        color: #94A3B8;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Section Headers */
    h3 {
        font-weight: 600;
        letter-spacing: -0.03em;
        color: #1E293B;
    }

    /* Prediction Banner */
    .prediction-box {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 36px 24px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(16, 185, 129, 0.25);
        margin-top: 1rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .prediction-box p {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 0.2rem;
        opacity: 0.9;
    }
    
    .prediction-box h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.03em;
    }
    
    .prediction-box span {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }

    /* Custom Button Styling */
    div.stButton > button {
        width: 100%;
        height: 52px;
        background: #2563EB;
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    
    div.stButton > button:hover {
        background: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }

    /* Streamlit Metric Cards Polish */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 16px 20px;
        border-radius: 16px;
        text-align: center;
    }
    
    div[data-testid="stMetricLabel"] {
        justify-content: center;
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
    }
    
    div[data-testid="stMetricValue"] {
        justify-content: center;
        color: #0F172A;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🏡 Real Estate Price AI</h1>
    <p>Accurate, real-time property valuation powered by machine learning algorithms</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD DATA & TRAIN MODEL
# -------------------------------------------------------
@st.cache_data
def load_data():
    # Fallback mock dataset generation if CSV is missing
    try:
        return pd.read_csv("houseprice.csv")
    except FileNotFoundError:
        import numpy as np
        np.random.seed(42)
        mock_area = np.random.normal(2000, 750, 150).astype(int)
        mock_area = np.clip(mock_area, 500, 7000)
        mock_price = mock_area * 4500 + np.random.normal(500000, 200000, 150)
        return pd.DataFrame({"area": mock_area, "price": np.clip(mock_price, 1500000, None)})

df = load_data()
feature = df.columns[0]
target = "price"

# Model Training
X = df[[feature]]
y = df[target]
model = LinearRegression()
model.fit(X, y)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8201/8201314.png", width=64)
    st.title("Model Dashboard")
    st.caption("Linear Regression v1.0")
    
    st.markdown("---")
    st.subheader("⚙️ System Specs")
    st.write("This algorithm evaluates property sizing against local historical sales data to project market value.")
    
    with st.expander("🛠️ Technology Stack"):
        st.markdown("""
        * **Frontend:** Streamlit
        * **Computation:** Pandas, NumPy
        * **ML Engine:** Scikit-Learn
        * **Visuals:** Plotly Express
        """)
    st.markdown("---")
    st.caption("Developed with precision UI/UX standards.")

# -------------------------------------------------------
# MAIN DASHBOARD AREA
# -------------------------------------------------------
col_input, col_stats = st.columns([5, 6], gap="large")

# --- LEFT COLUMN: USER INPUTS ---
with col_input:
    st.subheader("🛠️ Property Parameter")
    
    # Custom vs Slider Toggle
    input_method = st.radio(
        "Select Input Style:",
        ["🎯 Interactive Slider", "⌨️ Exact Custom Value"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    min_val = int(df[feature].min())
    max_val = int(df[feature].max())
    mean_val = int(df[feature].mean())
    
    if "Slider" in input_method:
        area = st.slider(
            label=f"Total Area ({feature.title()})",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=50,
            help="Drag to adjust the property footprint."
        )
    else:
        area = st.number_input(
            label=f"Enter Exact Area ({feature.title()})",
            min_value=100,
            max_value=25000,
            value=mean_val,
            step=10,
            help="Type your exact property dimensions directly."
        )

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Generate Valuation ⚡")

# --- RIGHT COLUMN: DATASET STATS ---
with col_stats:
    st.subheader("📊 Market Baseline Data")
    
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("Analyzed Properties", f"{len(df):,}")
        st.metric("Avg. Market Price", f"₹ {df[target].mean():,.0f}")
    with stat_col2:
        st.metric("Size Range", f"{min_val:,} - {max_val:,}")
        st.metric("Price per Sq.Ft (Avg)", f"₹ {(df[target]/df[feature]).mean():,.0f}")

# -------------------------------------------------------
# PREDICTION DISPLAY
# -------------------------------------------------------
if predict_btn or area:
    estimated_price = model.predict([[area]])[0]
    
    # Clamp prediction to zero if extreme negative values occur
    estimated_price = max(0, estimated_price)
    
    st.markdown(f"""
    <div class="prediction-box">
        <p>Estimated Market Valuation</p>
        <h1>₹ {estimated_price:,.0f}</h1>
        <span>Based on a footprint of <b>{area:,}</b> sq. ft.</span>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# VISUALIZATIONS & ML METRICS
# -------------------------------------------------------
st.markdown("---")

col_chart, col_metrics = st.columns([7, 3], gap="large")

with col_chart:
    st.subheader("📈 Market Price Trend & Regression Line")
    
    fig = px.scatter(
        df,
        x=feature,
        y=target,
        trendline="ols",
        trendline_color_override="#EF4444",
        color=target,
        color_continuous_scale="Blues",
        labels={feature: "Area (Sq. Ft.)", target: "Price (₹)"},
        template="plotly_white"
    )
    
    # Inject user's predicted point onto the chart
    if area:
        fig.add_scatter(
            x=[area], 
            y=[estimated_price],
            mode="markers",
            marker=dict(color="#10B981", size=14, line=dict(color="white", width=2)),
            name="Your Property"
        )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_showscale=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_metrics:
    st.subheader("🧮 Model Parameters")
    st.write("Mathematical weights underlying the prediction model:")
    
    st.metric(
        label="Marginal Value (Slope)",
        value=f"₹ {model.coef_[0]:,.2f}",
        delta="Per Sq.Ft added"
    )
    
    st.metric(
        label="Base Value (Intercept)",
        value=f"₹ {model.intercept_:,.2f}",
        help="The baseline calculation value before area is accounted for."
    )
    
    r_sq = model.score(X, y)
    st.metric(
        label="Model Accuracy ($R^2$)",
        value=f"{r_sq * 100:.1f}%",
        help="Proportion of the variance in house prices explained by the size of the house."
    )

# -------------------------------------------------------
# DATASET EXPLORER & FOOTER
# -------------------------------------------------------
with st.expander("📂 Explore Raw Training Dataset"):
    st.dataframe(
        df.style.format({target: "₹ {:,.0f}", feature: "{:,.0f}"}),
        use_container_width=True,
        height=250
    )

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
    Powered by <b>Streamlit</b> & <b>Scikit-Learn</b> • Engineered with Precision UI/UX
</div>
""", unsafe_allow_html=True)
