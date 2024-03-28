"""import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load data
crop = pd.read_csv("Crop_recommendation.csv")

# Drop duplicate rows
crop.drop_duplicates(inplace=True)

# Transform labels
crop_dict = {
    'rice': 1,
    'maize': 2,
    'Soyabeans': 3,
    'beans': 4,
    'peas': 5,
    'groundnuts': 6,
    'cowpeas': 7,
    'banana': 8,
    'mango': 9,
    'grapes': 10,
    'watermelon': 11,
    'apple': 12,
    'orange': 13,
    'cotton': 14,
    'coffee': 15
}

crop['label'] = crop['label'].map(crop_dict)

# Preprocess data
X = crop.drop(['label'], axis=1)
y = crop['label']

# Fit scalers
ms = MinMaxScaler()
X_scaled = ms.fit_transform(X)

sc = StandardScaler()
X_scaled = sc.fit_transform(X_scaled)

# Train the model
rfc = RandomForestClassifier()
rfc.fit(X_scaled, y)

def recommendation(N, P, k, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features)
    transformed_features = sc.transform(transformed_features)
    prediction = rfc.predict(transformed_features).reshape(1, -1)
    return prediction[0]

# Streamlit UI
st.title("Crop Recommendation System")

# Input features using sliders
N = st.slider("Nitrogen (N)", min_value=0, max_value=100, value=40)
P = st.slider("Phosphorus (P)", min_value=0, max_value=100, value=50)
k = st.slider("Potassium (k)", min_value=0, max_value=100, value=50)
temperature = st.slider("Temperature", min_value=0.0, max_value=50.0, value=40.0)
humidity = st.slider("Humidity", min_value=0, max_value=100, value=20)
ph = st.slider("pH", min_value=0, max_value=14, value=7)
rainfall = st.slider("Rainfall", min_value=0, max_value=500, value=100)

if st.button("Recommend Crop"):
    prediction = recommendation(N, P, k, temperature, humidity, ph, rainfall)
    if prediction in crop_dict.values():
        recommended_crop = [key for key, value in crop_dict.items() if value == prediction][0]
        st.write(f"The recommended crop is: {recommended_crop}")
    else:
        st.write("Sorry, we are not able to recommend a proper crop for this environment")"""


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load data
crop = pd.read_csv("Crop_recommendation.csv")

# Drop duplicate rows to ensure data consistency
crop.drop_duplicates(inplace=True)

# Transform labels to numerical values for model training
crop_dict = {
    'rice': 1,
    'maize': 2,
    'Soyabeans': 3,
    'beans': 4,
    'peas': 5,
    'groundnuts': 6,
    'cowpeas': 7,
    'banana': 8,
    'mango': 9,
    'grapes': 10,
    'watermelon': 11,
    'apple': 12,
    'orange': 13,
    'cotton': 14,
    'coffee': 15
}

crop['label'] = crop['label'].map(crop_dict)

# Preprocess data for model training
X = crop.drop(['label'], axis=1)
y = crop['label']

# Fit data scalers to normalize features
ms = MinMaxScaler()
X_scaled = ms.fit_transform(X)

sc = StandardScaler()
X_scaled = sc.fit_transform(X_scaled)

# Train the Random Forest model
rfc = RandomForestClassifier()
rfc.fit(X_scaled, y)

def recommendation(N, P, k, temperature, humidity, ph, rainfall):
    # Prepare input features for prediction
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    # Scale the input features using the fitted scalers
    transformed_features = ms.transform(features)
    transformed_features = sc.transform(transformed_features)
    # Make prediction using the trained model
    prediction = rfc.predict(transformed_features).reshape(1, -1)
    return prediction[0]

# Streamlit UI
st.title("Crop Recommendation System for Nigeria")
st.markdown("This model predicts the most suitable crop to grow based on environmental factors in Nigeria.")

# Input features using sliders for user interaction
st.markdown("### Environmental Factors")
st.markdown("Enter the values for the following environmental factors:")

N = st.slider("Nitrogen (N)", min_value=0, max_value=100, value=40,
              help="Amount of nitrogen in the soil (in kg/ha)")
P = st.slider("Phosphorus (P)", min_value=0, max_value=100, value=50,
              help="Amount of phosphorus in the soil (in kg/ha)")
k = st.slider("Potassium (K)", min_value=0, max_value=100, value=50,
              help="Amount of potassium in the soil (in kg/ha)")
temperature = st.slider("Temperature", min_value=0.0, max_value=50.0, value=30.0,
                        help="Average temperature (in °C)")
humidity = st.slider("Humidity", min_value=0, max_value=100, value=50,
                     help="Relative humidity (in %)")
ph = st.slider("pH", min_value=0, max_value=14, value=7,
               help="Soil pH level")
rainfall = st.slider("Rainfall", min_value=0, max_value=500, value=100,
                     help="Average rainfall (in mm)")

# Button to trigger crop recommendation
if st.button("Recommend Crop"):
    # Call recommendation function with input features
    prediction = recommendation(N, P, k, temperature, humidity, ph, rainfall)
    # Check if predicted label exists in crop dictionary
    if prediction in crop_dict.values():
        # Retrieve the recommended crop name
        recommended_crop = [key for key, value in crop_dict.items() if value == prediction][0]
        # Display the recommended crop to the user
        st.success(f"The recommended crop is: {recommended_crop.capitalize()}")
    else:
        # Notify the user if a recommendation is not available
        st.warning("Sorry, we are not able to recommend a proper crop for this environment")


