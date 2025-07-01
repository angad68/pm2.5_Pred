# ------------------ Imports ------------------ #
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import requests
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU

# ------------------ Configuration ------------------ #
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "7088853eac6948e286555436250107")
CITY = os.getenv("CITY", "Chandigarh")
MIN_PM25_VALUE = 20.0
MODEL_PATH = os.getenv("MODEL_PATH", "LIME_20240506.best.hdf5")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ------------------ Load Model ------------------ #
@st.cache_resource
def load_pm25_model():
    try:
        inputs = Input(shape=(224, 224, 3))
        x = Conv2D(64, (3, 3), padding='same')(inputs)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        pool2 = MaxPooling2D((3, 3), strides=(2, 2))(x)
        res1 = Conv2D(128, (3, 3), padding='same')(pool2)
        res1 = LeakyReLU(alpha=0.1)(res1)
        res1 = Add()([res1, pool2])
        x = MaxPooling2D((3, 3), strides=(2, 2))(res1)
        res2 = Conv2D(128, (3, 3), padding='same')(x)
        res2 = LeakyReLU(alpha=0.1)(res2)
        res2 = Add()([res2, x])
        x = MaxPooling2D((3, 3), strides=(2, 2))(res2)
        res3 = Conv2D(128, (3, 3), padding='same')(x)
        res3 = LeakyReLU(alpha=0.1)(res3)
        res3 = Add()([res3, x])
        x = MaxPooling2D((3, 3), strides=(2, 2))(res3)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.1)(x)
        output = Dense(1)(x)
        model = Model(inputs=inputs, outputs=output)
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model = load_pm25_model()

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("\U0001F2AB PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict PM2.5 air quality level. No indoor or obstructed photos.")

uploaded_file = st.file_uploader("Choose a sky image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File too large. Please upload an image smaller than 10MB.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
        if image.size[0] < 100 or image.size[1] < 100:
            st.error("Image too small. Please upload an image at least 100x100 pixels.")
            st.stop()
    except Exception as e:
        st.error(f"Invalid image file: {str(e)}")
        st.stop()

    st.image(image, caption="Uploaded Image", use_column_width=True)

    from quality_checks import comprehensive_image_check, categorize_pm25, validate_pm25_prediction, get_comprehensive_weather_info, adjust_prediction_with_weather

    with st.spinner("Analyzing image quality..."):
        quality_checks = comprehensive_image_check(image)

    failed_checks = []
    if quality_checks['is_blurry']:
        failed_checks.append("Image is blurry - try taking a clearer photo.")
    if quality_checks['is_poorly_exposed']:
        failed_checks.append("Image is too dark or bright - adjust exposure.")
    if not quality_checks['has_sufficient_sky']:
        failed_checks.append("Not enough sky visible - point camera upward.")

    cloudy_looking = quality_checks['is_cloudy']

    if failed_checks:
        st.error("\u26A0\uFE0F Image Quality Issues:")
        for i, issue in enumerate(failed_checks, 1):
            st.write(f"{i}. {issue}")
        st.info("\U0001F4A1 **Tips:** Take photos outdoors pointing upward at the sky, ensure good lighting and focus.")
        st.stop()

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Fetching weather data..."):
        weather_data = get_comprehensive_weather_info(CITY)

    with st.spinner("Predicting PM2.5 level..."):
        prediction = model.predict(img_array)
        base_pm25 = validate_pm25_prediction(prediction[0][0])
        pm25_value, weather_adjustments = adjust_prediction_with_weather(
            base_pm25, weather_data, cloudy_looking
        )

    category = categorize_pm25(pm25_value)

    st.subheader("\U0001F4CA Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PM2.5 Level", f"{pm25_value:.1f} µg/m³")
    with col2:
        colors = {
            "Good": "\U0001F7E2", "Satisfactory": "\U0001F7E1", "Moderately Polluted": "\U0001F7E0",
            "Poor": "\U0001F534", "Very Poor": "\U0001F7E3", "Severe": "\U0001F534"
        }
        st.metric("Air Quality", f"{colors.get(category, '⚪')} {category}")

    if weather_data:
        with st.expander("\U0001F324️ Weather Context", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Visibility", f"{weather_data['visibility_km']} km")
                st.metric("Humidity", f"{weather_data['humidity']}%")
            with col2:
                st.metric("Cloud Cover", f"{weather_data['cloud_cover']}%")
                st.metric("Wind", f"{weather_data['wind_kph']} kph")
            with col3:
                st.metric("Temperature", f"{weather_data['temp_c']}°C")
                st.write(f"**Condition:** {weather_data['condition']}")
            if weather_adjustments:
                st.write("**Weather Adjustments Applied:**")
                for adj in weather_adjustments:
                    st.write(f"• {adj}")

    health_advice = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities!",
        "Satisfactory": "Air quality is acceptable for most people.",
        "Moderately Polluted": "Sensitive individuals should limit prolonged outdoor activities.",
        "Poor": "Everyone should reduce prolonged outdoor activities.",
        "Very Poor": "Avoid outdoor activities. Keep windows closed.",
        "Severe": "Stay indoors. Avoid all outdoor activities."
    }

    if category in health_advice:
        st.info(f"\U0001F4A1 **Health Advice:** {health_advice[category]}")
