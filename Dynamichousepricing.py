# -*- coding: utf-8 -*-
"""Enhanced Linear Regression Deployment on Streamlit"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="House Price Predictor AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------
# Custom CSS for UI/UX Polish
# -----------------------------------
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Highlight Prediction Card */
    .prediction-box {
        background: linear-gradient(135deg, #1f77b4, #114b73);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.25);
        margin-bottom: 20px;
    }
    .prediction-box h2 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.9;
    }
    .prediction-box h1 {
        margin: 10px 0 0 0;
        font-size: 2.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Load & Prepare Dataset
# -----------------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("houseprice.csv")
    except FileNotFoundError:
        # Generate realistic fallback data if CSV is missing
        np.random.seed(42)
        areas = np.linspace(800, 6000, 50)
        prices = areas * 4500 + 150000 + np.random.normal(0, 350000, 50)
        return pd.DataFrame({"area": areas.astype(int), "price": prices.round(-3)})

df = load_data()

# Identify feature column automatically
feature_col = [col for col in df.columns if col.lower() != 'price'][0]

# -----------------------------------
# Train Model
# -----------------------------------
X = df[[feature_col]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)
r_sq = model.score(X, y)

# -----------------------------------
# Sidebar Controls
# -----------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/609/609803.png", width=60)
    st.title("Settings & Input")
    st.markdown("Adjust property parameters below to estimate market value.")
    
    min_area = int(df[feature_col].min())
    max_area = int(df[feature_col].max())
    mean_area = int(df[feature_col].mean())
    
    area_input = st.slider(
        "Property Area (Sq. Ft.)",
        min_value=max(100, min_area),
        max_value=max(10000, max_area),
        value=mean_area,
        step=50
    )
    
    predict_btn = st.button("Generate Valuation", type="primary", use_container_width=True)
    
    st.markdown("---")
    with st.expander("📂 View Raw Dataset"):
        st.dataframe(df, use_container_width=True, height=200)

# -----------------------------------
# Main Dashboard
# -----------------------------------
st.title("🏡 AI Property Valuation Engine")
st.markdown("Interactive linear regression model trained on local property market trends.")
st.markdown("---")

col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    # Live Calculation
    prediction = model.predict([[area_input]])[0]
    
    st.markdown(f"""
        <div class="prediction-box">
            <h2>Estimated Market Value</h2>
            <h1>₹ {prediction:,.0f}</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📊 Model Diagnostics")
    m1, m2 = st.columns(2)
    m1.metric("Model Accuracy (R²)", f"{r_sq * 100:.1f}%")
    m2.metric("Base Price (Intercept)", f"₹ {model.intercept_:,.0f}")
    
    m3, m4 = st.columns(2)
    m3.metric("Price per Sq.Ft (Slope)", f"₹ {model.coef_[0]:,.2f}")
    m4.metric("Dataset Sample Size", f"{len(df)} properties")

with col_right:
    st.subheader("📈 Market Trend Visualizer")
    
    # Plotly Interactive Chart
    fig = px.scatter(
        df, 
        x=feature_col, 
        y="price",
        opacity=0.65,
        labels={feature_col: "Area (Sq. Ft.)", "price": "Price (₹)"},
        title="Area vs. Price Regression Analysis"
    )
    
    # Add regression line
    x_range = np.linspace(df[feature_col].min(), df[feature_col].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    fig.add_trace(go.Scatter(
        x=x_range, y=y_range,
        mode="lines",
        name="Regression Trendline",
        line=dict(color="#d62728", width=2)
    ))
    
    # Highlight User Input Marker
    fig.add_trace(go.Scatter(
        x=[area_input], y=[prediction],
        mode="markers",
        name="Your Property",
        marker=dict(color="#2ca02c", size=14, symbol="star", line=dict(width=2, color="white"))
    ))
    
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
