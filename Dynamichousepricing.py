import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="🏡 House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#eef4ff,#ffffff);
}

.main-title{
    text-align:center;
    padding:30px;
    border-radius:20px;
    background:linear-gradient(135deg,#2563eb,#4f46e5);
    color:white;
    margin-bottom:25px;
}

.card{
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0 8px 20px rgba(0,0,0,.08);
}

.metric-card{
    background:#f8fafc;
    padding:18px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 5px 12px rgba(0,0,0,.05);
}

.prediction{
    background:linear-gradient(135deg,#10b981,#22c55e);
    color:white;
    padding:30px;
    border-radius:20px;
    text-align:center;
    margin-top:20px;
}

.prediction h1{
    font-size:50px;
}

div.stButton > button{
    width:100%;
    border-radius:12px;
    background:linear-gradient(90deg,#2563eb,#4f46e5);
    color:white;
    font-size:18px;
    height:55px;
    border:none;
}

div.stButton > button:hover{
    transform:scale(1.02);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown("""
<div class="main-title">
<h1>🏡 House Price Prediction</h1>
<p>Predict House Prices using Machine Learning (Linear Regression)</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Dataset Loader
# --------------------------------------------------

GITHUB_RAW_URL = "https://raw.githubusercontent.com/mrgraciz123/dynamic_house_price_prediction/main/houseprice.csv"

@st.cache_data
def load_data():

    try:
        return pd.read_csv("houseprice.csv")

    except Exception:

        try:
            return pd.read_csv(GITHUB_RAW_URL)

        except Exception as e:
            st.error("Unable to load dataset.")
            st.exception(e)
            st.stop()

df = load_data()

# --------------------------------------------------
# Validate Dataset
# --------------------------------------------------

if "price" not in df.columns:

    st.error("Dataset must contain a 'price' column.")
    st.stop()

# --------------------------------------------------
# Model Training
# --------------------------------------------------

X = df.drop("price", axis=1)
y = df["price"]

model = LinearRegression()
model.fit(X, y)

feature_name = X.columns[0]

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("📌 About")

st.sidebar.success("Linear Regression Model")

st.sidebar.write("""
### Features

- Live Prediction
- Dataset Preview
- Model Details
- Responsive UI
- Streamlit Dashboard
""")

# --------------------------------------------------
# Layout
# --------------------------------------------------

left, right = st.columns([1,1])

with left:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🏠 Property Details")

    area = st.slider(
        "Select House Area (sq.ft)",
        int(df[feature_name].min()),
        int(df[feature_name].max()),
        int(df[feature_name].mean()),
        100
    )

    predict = st.button("Predict Price")

    st.markdown("</div>", unsafe_allow_html=True)

with right:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📊 Dataset Summary")

    c1,c2,c3 = st.columns(3)

    c1.metric("Rows",len(df))
    c2.metric("Columns",len(df.columns))
    c3.metric("Average Price",f"₹ {df['price'].mean():,.0f}")

    st.progress(area/df[feature_name].max())

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict:

    prediction = model.predict([[area]])[0]

    st.markdown(f"""
    <div class='prediction'>
        <h3>Estimated House Price</h3>
        <h1>₹ {prediction:,.0f}</h1>
        <p>Predicted using Linear Regression</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# Model Details
# --------------------------------------------------

st.write("")

m1,m2 = st.columns(2)

with m1:

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)

    st.metric(
        "Coefficient",
        f"{model.coef_[0]:,.2f}"
    )

    st.markdown("</div>", unsafe_allow_html=True)

with m2:

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)

    st.metric(
        "Intercept",
        f"{model.intercept_:,.2f}"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

with st.expander("📂 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption("Made with ❤️ using Streamlit, Pandas & Scikit-Learn")
