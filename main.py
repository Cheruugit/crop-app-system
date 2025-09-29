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

# Custom CSS with green theme
st.markdown("""
<style>
/* Hero section */
.hero {
    position: relative;
    background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.4)),
                url('https://images.unsplash.com/photo-1600320841810-3b2a2d80a3c3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
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
}
.hero p {
    font-size: 1.2rem;
    opacity: 0.9;
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
h2, .st-subheader, .stMarkdown h2 {
    color: #166534 !important; /* deep green */
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #dcfce7, #f0fdf4);
    border: 1px solid #86efac;
    border-radius: 1rem;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease-in-out;
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
    background: linear-gradient(90deg, #16a34a, #22c55e);
    color: white;
    border-radius: 0.5rem;
    padding: 0.75rem;
    font-size: 1rem;
    font-weight: bold;
    border: none;
    transition: background 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #15803d, #16a34a);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🌱 Soil Whisperer</h1>
    <p>Smarter crop recommendations for healthier yields.</p>
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
st.subheader("ℹ️ How it Works")
st.write("""
Our AI-powered model analyzes **seven environmental factors**:  

- 🌿 **Soil nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)  
- 🌦️ **Climate**: Temperature, Humidity, Rainfall  
- 🧪 **Soil chemistry**: pH value  

The system is trained on agricultural datasets to provide **region-specific crop recommendations**.
""")
st.markdown('</div>', unsafe_allow_html=True)
