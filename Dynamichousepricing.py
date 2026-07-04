import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#eef2ff,#f8fafc);
}

/* Hero */
.hero{
    background:linear-gradient(135deg,#2563eb,#4f46e5);
    padding:35px;
    border-radius:20px;
    color:white;
    text-align:center;
    box-shadow:0px 10px 30px rgba(0,0,0,0.2);
    margin-bottom:30px;
}

.hero h1{
    font-size:42px;
}

.hero p{
    font-size:20px;
    opacity:0.9;
}

/* Cards */

.card{
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.08);
}

/* Prediction */

.prediction{
    background:linear-gradient(135deg,#16a34a,#22c55e);
    color:white;
    padding:30px;
    border-radius:20px;
    text-align:center;
    margin-top:25px;
}

.prediction h1{
    font-size:50px;
}

/* Metric */

.metric{
    background:white;
    border-radius:15px;
    padding:20px;
    text-align:center;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
}

/* Button */

div.stButton > button{
    background:linear-gradient(90deg,#2563eb,#4f46e5);
    color:white;
    font-size:18px;
    border:none;
    border-radius:12px;
    padding:12px;
    width:100%;
    transition:0.3s;
}

div.stButton > button:hover{
    transform:scale(1.03);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Hero Section
# -------------------------------------------------

st.markdown("""
<div class='hero'>
<h1>🏡 House Price Prediction</h1>
<p>Predict House Prices using Machine Learning (Linear Regression)</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv("https://github.com/mrgraciz123/dynamic_house_price_prediction/blob/main/houseprice.csv")

X = df.drop("price", axis=1)
y = df["price"]

model = LinearRegression()
model.fit(X,y)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("ℹ About")

st.sidebar.info("""
This app predicts house prices based on area using a trained
Linear Regression model.

**Model**
- Linear Regression
- Scikit-Learn
- Streamlit
""")

st.sidebar.success("Created with ❤️ using Streamlit")

# -------------------------------------------------
# Main Layout
# -------------------------------------------------

left,right = st.columns([1,1])

with left:

    st.markdown("<div class='card'>",unsafe_allow_html=True)

    st.subheader("🏠 Enter House Details")

    area = st.slider(
        "House Area (sq.ft)",
        min_value=100,
        max_value=10000,
        value=3300,
        step=100
    )

    predict = st.button("🚀 Predict Price")

    st.markdown("</div>",unsafe_allow_html=True)

with right:

    st.markdown("<div class='card'>",unsafe_allow_html=True)

    st.subheader("📈 Dataset Overview")

    st.metric("Rows",len(df))
    st.metric("Average Price",f"₹ {df['price'].mean():,.0f}")

    st.progress(min(area/10000,1.0))

    st.caption(f"Selected Area : **{area} sq.ft**")

    st.markdown("</div>",unsafe_allow_html=True)

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if predict:

    prediction = model.predict([[area]])[0]

    st.markdown(f"""
    <div class='prediction'>
        <h3>Estimated House Price</h3>
        <h1>₹ {prediction:,.0f}</h1>
        <p>Generated using Linear Regression</p>
    </div>
    """,unsafe_allow_html=True)

# -------------------------------------------------
# Model Details
# -------------------------------------------------

col1,col2 = st.columns(2)

with col1:
    st.markdown("<div class='metric'>",unsafe_allow_html=True)
    st.subheader("📊 Coefficient")
    st.write(f"**{model.coef_[0]:.2f}**")
    st.markdown("</div>",unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric'>",unsafe_allow_html=True)
    st.subheader("📈 Intercept")
    st.write(f"**{model.intercept_:.2f}**")
    st.markdown("</div>",unsafe_allow_html=True)

# -------------------------------------------------
# Dataset
# -------------------------------------------------

with st.expander("📂 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

st.markdown("---")

st.caption("Made with ❤️ using Streamlit & Scikit-Learn")
