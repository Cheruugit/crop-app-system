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
st.set_page_config(page_title="Crop Recommendation System", layout="wide")

# Custom CSS - polished green theme
st.markdown("""
<style>
body {
    background-color: #f8fdf9;
    font-family: "Segoe UI", Roboto, sans-serif;
}

/* Hero section */
.hero {
    position: relative;
    background: linear-gradient(rgba(22, 101, 52, 0.85), rgba(22, 101, 52, 0.85)),
                url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80');
    background-size: cover;
    background-position: center;
    border-radius: 1rem;
    padding: 4rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 {
    font-size: 3rem;
    margin-bottom: 0.5rem;
    color: #bbf7d0;
}
.hero p {
    font-size: 1.2rem;
    opacity: 0.95;
    color: #e8fbe9;
}

/* Card style */
.card {
    background: white;
    border-radius: 1rem;
    padding: 2rem;
    box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    margin-bottom: 2rem;
}

/* Section titles */
h1, h2, h3, .st-subheader, .stMarkdown h2 {
    color: #166534 !important;
    font-weight: 600;
}

/* Input labels */
label, .stNumberInput label, .stTextInput label {
    color: #166534 !important;
    font-weight: 500;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #bbf7d0, #dcfce7);
    border: 1px solid #22c55e;
    border-radius: 1rem;
    padding: 2rem;
    text-align: center;
}
.result-card h2 {
    color: #166534;
    font-size: 2.2rem;
    margin: 0;
}
.result-card p {
    color: #374151;
    margin-top: 0.5rem;
}

/* Green button */
div.stButton > button {
    background: linear-gradient(90deg, #15803d, #22c55e);
    color: white;
    border-radius: 0.5rem;
    padding: 0.75rem;
    font-size: 1rem;
    font-weight: bold;
    border: none;
    transition: background 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #166534, #16a34a);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🌱 Crop Recommendation System</h1>
    <p>AI-powered recommendations for smarter and greener farming.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Input Form (Card)
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🌿 Enter Soil & Climate Data")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("🟢 Nitrogen (N)", 0, 200, 50)
    P = st.number_input("🟡 Phosphorus (P)", 0, 200, 30)
    K = st.number_input("🟤 Potassium (K)", 0, 200, 40)
    temperature = st.number_input("🌡️ Temperature (°C)", 0.0, 50.0, 25.0)

with col2:
    humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 50.0)
    ph = st.number_input("🧪 Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 500.0, 100.0)

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
    st.subheader("🌱 Recommended Crop")
    st.markdown(
        f"""
        <div class="result-card">
            <h2>{prediction[0]}</h2>
            <p>This crop is optimal for your current environmental conditions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Info Section (Card)
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🌿 How it Works")
st.write("""
Our AI-powered model analyzes **seven environmental factors**:  

- 🟢 **Nitrogen (N)**  
- 🟡 **Phosphorus (P)**  
- 🟤 **Potassium (K)**  
- 🌡️ **Temperature**  
- 💧 **Humidity**  
- 🌧️ **Rainfall**  
- 🧪 **Soil pH**  

The system is trained on agricultural datasets to provide **region-specific crop recommendations**.
""")
st.markdown('</div>', unsafe_allow_html=True)
