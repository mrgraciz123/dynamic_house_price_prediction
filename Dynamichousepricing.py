import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown("""
<style>

.main{
    background:#F5F7FB;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.hero{
    background:linear-gradient(135deg,#2563EB,#4F46E5);
    border-radius:20px;
    padding:45px;
    color:white;
    text-align:center;
    margin-bottom:30px;
}

.hero h1{
    font-size:55px;
    font-weight:700;
}

.hero p{
    font-size:22px;
}

.card{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 10px 25px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

.metric-card{
    background:#ffffff;
    border-radius:15px;
    padding:20px;
    text-align:center;
    box-shadow:0px 5px 18px rgba(0,0,0,.08);
}

.prediction{
    background:linear-gradient(135deg,#16A34A,#22C55E);
    color:white;
    padding:35px;
    border-radius:20px;
    text-align:center;
    margin-top:25px;
}

.prediction h1{
    font-size:52px;
}

div.stButton > button{
    width:100%;
    height:55px;
    background:linear-gradient(90deg,#2563EB,#4F46E5);
    color:white;
    border:none;
    border-radius:12px;
    font-size:18px;
}

div.stButton > button:hover{
    background:#1D4ED8;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HERO
# -------------------------------------------------------

st.markdown("""
<div class="hero">
<h1>🏠 House Price Prediction</h1>
<p>Predict house prices using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("houseprice.csv")

try:
    df = load_data()
except:
    st.error("houseprice.csv not found.")
    st.stop()

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------

X = df.drop("price", axis=1)
y = df["price"]

model = LinearRegression()
model.fit(X,y)

feature = X.columns[0]

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("📌 About")

st.sidebar.success("Linear Regression")

st.sidebar.write("""
Predict house prices based on the area of the house.

Technology Used

- Streamlit
- Pandas
- Scikit-Learn
- Plotly
""")

# -------------------------------------------------------
# TOP CARDS
# -------------------------------------------------------

left,right=st.columns([1,1])

with left:

    st.markdown('<div class="card">',unsafe_allow_html=True)

    st.subheader("🏠 Property Details")

    area=st.slider(
        "Area (Square Feet)",
        int(df[feature].min()),
        int(df[feature].max()),
        int(df[feature].mean()),
        100
    )

    predict=st.button("Predict Price")

    st.markdown("</div>",unsafe_allow_html=True)

with right:

    st.markdown('<div class="card">',unsafe_allow_html=True)

    st.subheader("📊 Dataset Summary")

    c1,c2,c3=st.columns(3)

    c1.metric("Rows",len(df))
    c2.metric("Minimum",f"{df[feature].min():,.0f}")
    c3.metric("Average Price",f"₹ {df['price'].mean():,.0f}")

    st.markdown("</div>",unsafe_allow_html=True)

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if predict:

    value=model.predict([[area]])[0]

    st.markdown(f"""
    <div class="prediction">
    <h2>Estimated House Price</h2>
    <h1>₹ {value:,.0f}</h1>
    <h4>Area : {area:,} sq.ft</h4>
    </div>
    """,unsafe_allow_html=True)

# -------------------------------------------------------
# MODEL DETAILS
# -------------------------------------------------------

st.write("")

m1,m2=st.columns(2)

with m1:

    st.markdown('<div class="metric-card">',unsafe_allow_html=True)

    st.metric(
        "Coefficient",
        f"{model.coef_[0]:.2f}"
    )

    st.markdown("</div>",unsafe_allow_html=True)

with m2:

    st.markdown('<div class="metric-card">',unsafe_allow_html=True)

    st.metric(
        "Intercept",
        f"{model.intercept_:.2f}"
    )

    st.markdown("</div>",unsafe_allow_html=True)

# -------------------------------------------------------
# CHART
# -------------------------------------------------------

st.write("")
st.subheader("📈 House Area vs Price")

fig=px.scatter(
    df,
    x=feature,
    y="price",
    trendline="ols",
    color="price",
    template="plotly_white"
)

fig.update_layout(
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------------------------------------
# DATASET
# -------------------------------------------------------

with st.expander("📂 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("---")

st.caption("Made with ❤️ using Streamlit • Pandas • Scikit-Learn • Plotly")
