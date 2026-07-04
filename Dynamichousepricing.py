import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Property Valuation Engine",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1350px; }
    
    .hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #2563EB 100%);
        border-radius: 24px; padding: 42px 32px; color: white; text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15); margin-bottom: 2rem;
    }
    .hero h1 { font-size: 3rem; font-weight: 700; margin-bottom: 0.4rem; color: #FFFFFF; }
    .hero p { font-size: 1.15rem; color: #94A3B8; max-width: 650px; margin: 0 auto; }
    
    .prediction-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white; padding: 32px; border-radius: 20px; text-align: center;
        box-shadow: 0 15px 25px -5px rgba(16, 185, 129, 0.3); margin-bottom: 1.5rem;
    }
    .prediction-card p { text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; font-size: 0.9rem; opacity: 0.9; margin: 0; }
    .prediction-card h1 { font-size: 3.4rem; font-weight: 700; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🏡 Multivariate Valuation Engine</h1>
    <p>Predict real estate market prices dynamically using multi-feature machine learning</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# AUTOMATIC DATASET LOADING & PROCESSING
# -------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("houseprice.csv")
    except FileNotFoundError:
        # Fallback realistic multi-variable sample dataset
        np.random.seed(42)
        n = 250
        area = np.random.normal(2200, 600, n).astype(int)
        bedrooms = np.random.randint(1, 6, n)
        bathrooms = np.random.randint(1, 4, n)
        age = np.random.randint(0, 40, n)
        price = (area * 320) + (bedrooms * 25000) + (bathrooms * 18000) - (age * 2200) + np.random.normal(150000, 45000, n)
        df = pd.DataFrame({"area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "age_years": age, "price": price})
    
    # Clean missing values and isolate numerical features
    df = df.dropna()
    return df

df = load_and_preprocess_data()

# Detect target column ('price' or last column)
target_col = "price" if "price" in df.columns.str.lower() else df.columns[-1]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != target_col]

if not feature_cols:
    st.error("No numerical feature columns found in the dataset to train the model.")
    st.stop()

# -------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------
X = df[feature_cols]
y = df[target_col]

model = LinearRegression()
model.fit(X, y)
r_squared = model.score(X, y)

# -------------------------------------------------------
# SIDEBAR: DATASET UPLOADER & OVERVIEW
# -------------------------------------------------------
with st.sidebar:
    st.title("📂 Dataset Controls")
    uploaded_file = st.file_uploader("Upload Custom CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file).dropna()
        st.success("Custom dataset loaded successfully!")
        st.rerun()
        
    st.markdown("---")
    st.subheader("📊 Model Diagnostics")
    st.metric("Trained Features", len(feature_cols))
    st.metric("Dataset Records", f"{len(df):,}")
    st.metric("Model Accuracy (R²)", f"{r_squared * 100:.1f}%")

# -------------------------------------------------------
# MAIN LAYOUT: DYNAMIC INPUTS & PREDICTION
# -------------------------------------------------------
col_inputs, col_results = st.columns([5, 7], gap="large")

with col_inputs:
    st.subheader("🛠️ Property Specifications")
    st.caption("Adjust parameters detected directly from your dataset columns:")
    
    user_inputs = {}
    
    # Dynamically build UI controls for every detected feature in the dataset
    for col in feature_cols:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        col_mean = float(df[col].mean())
        
        # Check if integer or continuous float
        is_integer = np.array_equal(df[col], df[col].astype(int))
        
        if is_integer:
            step_size = max(1, int((col_max - col_min) / 50))
            user_inputs[col] = st.number_input(
                label=f"📌 {col.replace('_', ' ').title()}",
                min_value=int(col_min),
                max_value=int(col_max * 2),
                value=int(col_mean),
                step=step_size
            )
        else:
            step_size = (col_max - col_min) / 100.0
            user_inputs[col] = st.number_input(
                label=f"📌 {col.replace('_', ' ').title()}",
                min_value=float(col_min),
                max_value=float(col_max * 2.0),
                value=float(col_mean),
                step=float(step_size),
                format="%.2f"
            )

with col_results:
    st.subheader("💡 AI Valuation Estimate")
    
    # Create input vector matching training column order
    input_vector = pd.DataFrame([user_inputs])
    prediction = max(0.0, model.predict(input_vector)[0])
    
    st.markdown(f"""
    <div class="prediction-card">
        <p>Predicted Market Value</p>
        <h1>₹ {prediction:,.0f}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Impact Breakdown Table
    st.subheader("⚖️ Feature Weight Analysis")
    impact_df = pd.DataFrame({
        "Feature Attribute": [c.replace("_", " ").title() for c in feature_cols],
        "Input Value": [user_inputs[c] for c in feature_cols],
        "Model Weight (Coefficient)": [f"₹ {coef:,.2f}" for coef in model.coef_]
    })
    st.dataframe(impact_df, use_container_width=True, hide_index=True)

# -------------------------------------------------------
# VISUALIZATION & RAW DATA
# -------------------------------------------------------
st.markdown("---")
st.subheader("📈 Multi-Feature Market Trends")

selected_viz_col = st.selectbox("Select Feature to Plot against Price:", feature_cols)

fig = px.scatter(
    df, x=selected_viz_col, y=target_col,
    trendline="ols", trendline_color_override="#EF4444",
    labels={selected_viz_col: selected_viz_col.replace("_", " ").title(), target_col: "Price (₹)"},
    template="plotly_white", opacity=0.6
)

# Highlight active user prediction on chart
fig.add_scatter(
    x=[user_inputs[selected_viz_col]], y=[prediction],
    mode="markers", name="Your Valuation",
    marker=dict(color="#10B981", size=16, line=dict(color="white", width=2))
)

fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

with st.expander("📂 Inspect Raw Training Dataset"):
    st.dataframe(df, use_container_width=True)
