import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU
import os
import requests

# Configurations
MODEL_PATH = "LIME_20240506.best.hdf5"
MIN_PM25_VALUE = 20.0
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "7088853eac6948e286555436250107")
CITY = os.getenv("CITY", "Chandigarh")

# ------------------ Load Model ------------------ #
@st.cache_resource
def load_pm25_model():
    inputs = Input(shape=(224, 224, 3))

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool')(x)

    res_input = x
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, res_input])
    x = MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool')(x)

    res_input = x
    x = Conv2D(128, (3, 3), padding='same', name='block4_conv1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, res_input])
    x = MaxPooling2D((3, 3), strides=(2, 2), name='block4_pool')(x)

    res_input = x
    x = Conv2D(128, (3, 3), padding='same', name='block5_conv1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, res_input])
    x = MaxPooling2D((3, 3), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block6_conv1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (3, 3), padding='same', name='block6_conv2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='block6_pool')(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.1)(x)
    output = Dense(1, activation='linear', name='PM2.5_output')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
    model.load_weights(MODEL_PATH)

    return model

model = load_pm25_model()

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a sky image to predict PM2.5 air quality level.")

uploaded_file = st.file_uploader("Choose a sky image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File too large. Please upload an image smaller than 10MB.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if image.size[0] < 100 or image.size[1] < 100:
            st.error("Image too small. Please upload an image at least 100x100 pixels.")
            st.stop()
    except Exception as e:
        st.error(f"Invalid image file: {str(e)}")
        st.stop()

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Predicting PM2.5 level..."):
        prediction = model.predict(img_array)
        pm25_value = float(prediction[0][0])
        pm25_value = max(0.0, min(1000.0, pm25_value))

    # Categorize
    def categorize_pm25(value):
        if value <= 30:
            return "Good"
        elif value <= 60:
            return "Satisfactory"
        elif value <= 90:
            return "Moderately Polluted"
        elif value <= 120:
            return "Poor"
        elif value <= 250:
            return "Very Poor"
        else:
            return "Severe"

    category = categorize_pm25(pm25_value)
    colors = {
        "Good": "🟢",
        "Satisfactory": "🟡",
        "Moderately Polluted": "🟠",
        "Poor": "🔴",
        "Very Poor": "🟣",
        "Severe": "🔴"
    }

    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PM2.5 Level", f"{pm25_value:.1f} µg/m³")
    with col2:
        st.metric("Air Quality", f"{colors.get(category)} {category}")
