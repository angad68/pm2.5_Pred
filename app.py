import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load model
@st.cache_resource
def load_pm25_model():
    return load_model("pm25_cnn_model.hdf5")

model = load_pm25_model()

# Title
st.title("PM2.5 Prediction from Sky Image 🌫️")
st.write("Upload a sky image to predict PM2.5 concentration (µg/m³).")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_normalized = img_array / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    # Predict
    prediction = model.predict(img_batch)
    predicted_pm25 = round(prediction[0][0], 2)

    st.subheader("Predicted PM2.5:")
    st.metric(label="µg/m³", value=predicted_pm25)

    # Add simple category interpretation
    def categorize_pm25(pm25):
        if pm25 <= 12:
            return "Good 😊"
        elif pm25 <= 35.4:
            return "Moderate 😐"
        elif pm25 <= 55.4:
            return "Unhealthy for Sensitive Groups 😷"
        elif pm25 <= 150.4:
            return "Unhealthy 😷"
        elif pm25 <= 250.4:
            return "Very Unhealthy 🤢"
        else:
            return "Hazardous ☠️"

    st.write("Air Quality Category:", categorize_pm25(predicted_pm25))
