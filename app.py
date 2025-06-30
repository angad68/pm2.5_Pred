import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the PM2.5 model
@st.cache_resource
def load_pm25_model():
    return load_model("LIME_20240506.best.hdf5")

model = load_pm25_model()

# Set page config
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")

# App Title
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a sky image to predict the PM2.5 air quality level.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Class labels (adjust if needed)
class_labels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']

# Predict button
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    st.subheader("📊 Prediction")
    st.write(f"**Air Quality Category:** {class_labels[class_idx]}")
