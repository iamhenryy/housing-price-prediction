import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
from models_algorithms import K_means, KNN_regressor

# =========================
# 1. Page Config
# =========================
st.set_page_config(page_title="House Price Prediction", page_icon="$", layout="wide")

st.markdown('''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    .stApp {
        background-color: #F7F9FB;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E6E8EB;
    }

    .prediction-card {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 24px;
        border: 1px solid #E6E8EB;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.03);
    }

    /* Typography */
    h1, h2, h3 {
        color: #1D1D1F;
        font-weight: 600 !important;
    }
    
    .stSlider label, .stNumberInput label {
        color: #48484A !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    .stButton>button {
        background: #1D1D1F;
        color: white;
        border: none;
        width: 100%;
        font-weight: 500;
        height: 3.2em;
        border-radius: 12px;
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
        margin-top: 10px;
    }
    
    # .stButton>button:hover {
    #     background: #434345;
    #     box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    #     transform: translateY(-1px);
    # }

    [data-testid="stImage"] img {
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .hr-custom {
        border: 0;
        height: 1px;
        background: #E6E8EB;
        margin: 30px 0;
    }
    </style>
    ''', unsafe_allow_html=True)

# =========================
# 2. Model Loading
# =========================
@st.cache_resource
def load_all():
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("models/knn_models.pkl", "rb") as f:
        knn_models = pickle.load(f)
    return scaler, kmeans, knn_models

scaler, kmeans, knn_models = load_all()
features = list(scaler.keys())

# =========================
# 3. Layout: Title
# =========================
st.title("𖠿 House Price Prediction")
st.markdown('<div class="hr-text"></div>', unsafe_allow_html=True)

col_input, col_display = st.columns([1.2, 1.8], gap="large")

with col_input:
    st.subheader("Property Specs")
    
    house_age = st.slider("House age (years)", 0, 100, 10)
    distance = st.number_input("Distance to MRT (meters)", 0.0, 5000.0, 500.0)
    stores = st.slider("Convenience stores nearby", 0, 15, 3)

    st.markdown("#### ⚲ Location")
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", 24.9, 25.1, 25.0330, format="%.6f")
    lon = c2.number_input("Longitude", 121.4, 121.6, 121.5654, format="%.6f")

    predict_btn = st.button("Calculate Market Value")

# =========================
# 4. Layout: Display Section
# =========================
with col_display:
    if not predict_btn:
        st.image("https://images.unsplash.com/photo-1570129477492-45c003edd2be?q=80&w=2070&auto=format&fit=crop", 
                 caption="Enter details to start analysis", 
                 use_container_width=True)
    else:
        # --- PREDICTION LOGIC ---
        x = np.array([[house_age, np.log1p(distance), stores, lat, lon]])
        x_scaled = x.copy()

        for i, col in enumerate(features):
            mn, mx = scaler[col]
            x_scaled[0, i] = (x_scaled[0, i] - mn) / (mx - mn)

        cluster = kmeans.assign_labels(x_scaled)[0]
        pred_log = knn_models[cluster].predict(x_scaled)[0]
        price = np.expm1(pred_log)

        # --- THE HERO PREDICTION ---
        st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: #888; margin-bottom: 0;">ESTIMATED MARKET VALUE</h3>
                <h1 style="color: #003f88; font-size: 4em; margin-top: 0;">${price:,.2f}</h1>
                <p style="color: #555;">Based on K-Means Cluster {cluster} and K Nearest Neighbor Analysis</p>
            </div>
        """, unsafe_allow_html=True)

        # --- FEATURE IMPACT ---
        st.subheader("⌕ Model Interpretability")
        
        raw_impacts = {
            "House Age": (x_scaled[0, 0] - 0.5) * -1,
            "MRT Access": (x_scaled[0, 1] - 0.5) * -1.5,
            "Local Stores": (x_scaled[0, 2] - 0.5) * 1.2,
            "Position (N/S)": (x_scaled[0, 3] - 0.5) * 0.3,
            "Position (E/W)": (x_scaled[0, 4] - 0.5) * 0.3
        }
        
        impact_df = pd.DataFrame({
            "Feature": list(raw_impacts.keys()),
            "Impact": list(raw_impacts.values())
        }).sort_values(by="Impact")

        impact_df["Color"] = ["#FF4B4B" if val < 0 else "#00FFCC" for val in impact_df["Impact"]]

        fig_impact = px.bar(
            impact_df, x="Impact", y="Feature", orientation='h',
            color="Color", color_discrete_map="identity",
            template="plotly_dark"
        )
        fig_impact.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0), height=300,
            xaxis=dict(showgrid=False, zerolinecolor="#444", showticklabels=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_impact, use_container_width=True)

# =========================
# 5. Full Width Map at the End
# =========================
if predict_btn:
    st.markdown('<div class="hr-text"></div>', unsafe_allow_html=True)
    st.subheader("⌖ Geographic Context")
    st.map(data=[{"lat": lat, "lon": lon}], zoom=15)

st.markdown("<br><br><center><p style='color:#333;'>Group 8 {at} Honors Program VNU-HCMUS © 2025</p></center>", unsafe_allow_html=True)