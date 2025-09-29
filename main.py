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
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Crop Recommendation System", layout="wide")

st.title("🌱 Crop Recommendation System")
st.write("Enter soil and climate conditions to get the most suitable crop recommendation.")

# Input fields in two columns
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

# Predict button
if st.button("🌾 Recommend Crop", use_container_width=True):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    # Result card
    st.success("✅ Recommended Crop")
    st.markdown(
        f"""
        <div style="
            padding: 1.5rem;
            border-radius: 0.5rem;
            background: linear-gradient(135deg, #e0f7e9, #f9fff9);
            border: 1px solid #b2dfdb;
            text-align: center;">
            <h2 style="color:#2e7d32; margin:0;">{prediction[0]}</h2>
            <p style="color:#555;">This crop is optimal for your current conditions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Info card
with st.expander("ℹ️ How it works"):
    st.write(
        """
        The system analyzes seven key environmental factors:
        - **Soil nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)  
        - **Climate conditions**: Temperature, Humidity, Rainfall  
        - **Soil chemistry**: pH value  

        Our model is trained on agricultural data to provide context-specific crop recommendations.
        """
    )
