import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Load and train the model once
# -----------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("Crop_recommendation.csv")
    df.drop_duplicates(inplace=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = load_model()

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Soil Whisperer", layout="wide")

# Inject custom CSS for Soil Whisperer look
st.markdown("""
    <style>
    .hero {
        position: relative;
        background: linear-gradient(rgba(34, 197, 94, 0.8), rgba(34, 197, 94, 0.5)),
                    url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80');
        background-size: cover;
        background-position: center;
        padding: 4rem 2rem;
        text-align: center;
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }
    .card {
        background: white;
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #e0f7e9, #f9fff9);
        border: 1px solid #b2dfdb;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero">
    <h1 style="font-size:3rem; margin-bottom:0.5rem;">🌱 Soil Whisperer</h1>
    <p style="font-size:1.25rem;">Smart crop recommendations tailored to your soil and climate.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Input Form (Card)
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Enter Soil & Climate Data")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 50)
    P = st.number_input("Phosphorus (P)", 0, 200, 30)
    K = st.number_input("Potassium (K)", 0, 200, 40)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
    ph = st.number_input("pH value", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

predict_btn = st.button("🌾 Recommend Crop", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Result (Card)
# -----------------------------
if predict_btn:
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✅ Recommended Crop")
    st.markdown(
        f"""
        <div class="result-card">
            <h2 style="color:#2e7d32; margin:0; font-size:2rem;">{prediction[0]}</h2>
            <p style="color:#555;">This crop is optimal for your current environmental conditions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Info Section (Card)
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("ℹ️ How it Works")
st.write("""
Our AI-powered model analyzes **seven environmental factors**:  

- 🌿 **Soil nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)  
- 🌦️ **Climate**: Temperature, Humidity, Rainfall  
- 🧪 **Soil chemistry**: pH value  

By learning from agricultural datasets, the system provides reliable, region-specific crop recommendations.  
""")
st.markdown('</div>', unsafe_allow_html=True)

