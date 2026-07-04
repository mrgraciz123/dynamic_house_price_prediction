# 🏡 AI Property Valuation Engine
**An Interactive Real Estate Price Prediction Dashboard powered by Machine Learning**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)

---

## 📌 Overview

The **AI Property Valuation Engine** is a sleek, modern Streamlit web application that utilizes **Ordinary Least Squares (OLS) Linear Regression** to predict real estate market values. By analyzing historical property datasets, the application identifies key mathematical weights ($\beta$) corresponding to features like square footage, bedrooms, bathrooms, and property age to generate instant, sub-millisecond valuations.

Designed with **SaaS-level UI/UX principles**, this tool allows homeowners, real estate agents, and data analysts to explore market trends interactively and visualize how property specifications directly impact market valuations.

---

## ✨ Key Features

- **⚡ Instant AI Valuations:** Real-time property price predictions calculated dynamically as user parameters change.
- **🛠️ Hybrid & Dynamic Inputs:** Supports both quick interactive sliders and exact numerical inputs, automatically scaling and adapting to your specific CSV dataset distribution.
- **📈 Interactive Visualizations:** Built with **Plotly Express**, rendering responsive scatter plots, dynamic regression trendlines, and a pinpoint marker for the user's active property estimate.
- **🧮 Model Diagnostics & Explainability:** Displays transparent ML metrics, including:
  - **Marginal Value (Slope / Coefficients):** Price increase per added unit/sq.ft.
  - **Base Value (Intercept):** The foundational market baseline price.
  - **Model Accuracy ($R^2$ Score):** The percentage of price variance explained by the model.
- **📂 Automatic Dataset Fallback & Inspection:** Automatically detects numeric columns in custom CSV uploads or generates realistic fallback data for seamless testing and zero-crash deployments.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend UI/UX** | [Streamlit](https://streamlit.io/) | Interactive web framework with custom CSS glassmorphic cards |
| **Data Processing** | [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) | Data manipulation, cleaning, and vector algebra |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) | `LinearRegression` algorithm implementation |
| **Data Visualization** | [Plotly Express](https://plotly.com/python/) | High-performance interactive charting engine |

---

## 🚀 Quickstart & Installation

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/house-price-prediction-streamlit.git](https://github.com/yourusername/house-price-prediction-streamlit.git)
cd house-price-prediction-streamlit
2. Create a Virtual Environment (Recommended)Bash# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
3. Install DependenciesCreate a requirements.txt file (or install directly):Bashpip install -r requirements.txt
(If installing manually: pip install streamlit pandas numpy scikit-learn plotly)4. Prepare Your DatasetEnsure your historical sales dataset (houseprice.csv) is placed in the root directory.Note: If no file is detected, the app will automatically generate realistic synthetic data so you can test immediately.5. Launch the Streamlit AppBashstreamlit run app.py
Open your web browser and navigate to http://localhost:8501.🧠 How the Machine Learning Model WorksThe app employs Multivariate Ordinary Least Squares (OLS) Regression. The algorithm predicts the target property value ($y$) by computing the weighted linear combination of property features ($x_1, x_2, \dots, x_n$):$$\text{Price} = \beta_0 + (\beta_1 \times \text{Area}) + (\beta_2 \times \text{Bedrooms}) + \dots + \epsilon$$Where:$\beta_0$ (Intercept): The baseline price of a property before feature additions.$\beta_i$ (Coefficients): The exact monetary value added or subtracted by each specific feature.$\epsilon$ (Error Term): Market noise and unmeasured variables.📁 Repository StructurePlaintext├── app.py                 # Core Streamlit application & UI layout
├── houseprice.csv         # Sample/historical property sales dataset
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation
🌐 DeploymentEasily deploy this application for free using Streamlit Community Cloud:Push your code and requirements.txt to a GitHub repository.Visit share.streamlit.io and log in with GitHub.Click "New App", select your repository, specify app.py as the main file, and click Deploy.
