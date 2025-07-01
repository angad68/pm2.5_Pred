import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU, Dropout
import os
import requests

# ------------------ Streamlit Page Config & Styles ------------------ #
st.set_page_config(page_title="PM2.5 Air Quality Predictor", layout="centered")

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: #f7f9fc;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: #003366;
        }
        .stButton>button {
            background-color: #003366;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            border: none;
        }
        .stMetricValue {
            font-size: 1.4rem !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🌫️ PM2.5 Air Quality Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>AI-powered prediction from sky images using a custom CNN model</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Quality Check Functions ------------------ #
def is_blurry(pil_img, threshold=25.0):
    img_gray = np.array(pil_img.convert("L"))
    return cv2.Laplacian(img_gray, cv2.CV_64F).var() < threshold

def is_overexposed_or_underexposed(pil_img, low_thresh=35, high_thresh=220):
    img_gray = np.array(pil_img.convert("L"))
    mean_val = np.mean(img_gray)
    return mean_val < low_thresh or mean_val > high_thresh

def is_mostly_white_or_black(pil_img, white_thresh=235, black_thresh=25, percent=0.75):
    img = np.array(pil_img)
    white_pixels = np.sum(np.all(img > white_thresh, axis=2))
    black_pixels = np.sum(np.all(img < black_thresh, axis=2))
    total_pixels = img.shape[0] * img.shape[1]
    return (white_pixels + black_pixels) / total_pixels > percent

def is_sky_image(pil_img, sky_percent=0.45):
    img = pil_img.resize((256, 256)).convert("RGB")
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    blue_sky = ((h >= 100) & (h <= 130)) & (s > 40) & (v > 90)
    light_blue = ((h >= 85) & (h <= 110)) & (s > 20) & (v > 110)
    gray_sky = (s < 20) & (v > 130) & (v < 210)
    bright_areas = (v > 210) & (s < 30)

    sky_mask = blue_sky | light_blue | gray_sky | bright_areas

    height, width = sky_mask.shape
    weight_mask = np.ones_like(sky_mask, dtype=float)
    for i in range(height):
        weight_mask[i, :] = 1.0 + (height - i) / height

    weighted_sky_pixels = np.sum(sky_mask * weight_mask)
    total_weighted_pixels = np.sum(weight_mask)
    weighted_sky_ratio = weighted_sky_pixels / total_weighted_pixels

    return weighted_sky_ratio > sky_percent

# ------------------ Load Model ------------------ #
MODEL_PATH = "LIME_20240506.best.hdf5"
MIN_PM25_VALUE = 20.0
MAX_FILE_SIZE = 10 * 1024 * 1024
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "7088853eac6948e286555436250107")
CITY = os.getenv("CITY", "Chandigarh")

@st.cache_resource
def load_pm25_model():
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
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    for _ in range(3):
        res_input = x
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Add()([x, res_input])
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Dropout(0.3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(1024)(x)
    x = Dropout(0.3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
    model.load_weights(MODEL_PATH)
    return model

model = load_pm25_model()

# ------------------ Upload & Predict ------------------ #
with st.container():
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a sky image (JPG/PNG, < 10MB)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("File too large. Please upload an image smaller than 10MB.")
            st.stop()

        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if image.size[0] < 100 or image.size[1] < 100:
                st.error("Image too small. Minimum 100x100 pixels required.")
                st.stop()
        except Exception as e:
            st.error(f"Invalid image file: {str(e)}")
            st.stop()

        # Block prediction on quality issues
        if is_blurry(image):
            st.error("Prediction aborted: image is too blurry.")
            st.stop()
        if is_overexposed_or_underexposed(image):
            st.error("Prediction aborted: overexposed or underexposed.")
            st.stop()
        if is_mostly_white_or_black(image):
            st.error("Prediction aborted: mostly white/black.")
            st.stop()
        if not is_sky_image(image):
            st.error("Prediction aborted: image does not appear to show sky.")
            st.stop()

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Predicting PM2.5 level with uncertainty estimation..."):
            preds = [model(img_array, training=True).numpy().squeeze() for _ in range(30)]
            pm25_value = float(np.mean(preds))
            pm25_std = float(np.std(preds))
            pm25_value = max(0.0, min(1000.0, pm25_value))

        # Weather API
        try:
            weather_url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={CITY}"
            weather_data = requests.get(weather_url).json()
            condition = weather_data.get("current", {}).get("condition", {}).get("text", "").lower()
            if "cloud" in condition or "overcast" in condition:
                st.info(f"☁️ Detected cloudy/overcast weather in {CITY.title()}. Prediction adjusted.")
                pm25_value = max(pm25_value, MIN_PM25_VALUE)
        except:
            st.warning("⚠️ Could not fetch weather data. Skipping weather adjustment.")

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

        st.markdown("---")
        st.subheader("📊 Prediction Results")
        col1, col2 = st.columns(2)
        col1.metric("PM2.5 Level", f"{pm25_value:.1f} µg/m³")
        col2.metric("Uncertainty (±)", f"{pm25_std:.1f}")
        st.markdown(f"**Air Quality:** {colors.get(category)} {category}")
