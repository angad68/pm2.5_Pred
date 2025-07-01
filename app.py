import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU
import os
import requests

# ------------------ Quality Checks ------------------ #
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

def is_sky_image(pil_img, sky_percent=0.3):
    img = pil_img.resize((256, 256)).convert("RGB")
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    blue_sky = ((h >= 100) & (h <= 130)) & (s > 30) & (v > 80)
    light_blue = ((h >= 80) & (h <= 110)) & (s > 15) & (v > 100)
    gray_sky = (s < 30) & (v > 120) & (v < 220)
    bright_areas = (v > 200) & (s < 50)

    sky_mask = blue_sky | light_blue | gray_sky | bright_areas

    height, width = sky_mask.shape
    weight_mask = np.ones_like(sky_mask, dtype=float)
    for i in range(height):
        weight_factor = 1.0 + (height - i) / height
        weight_mask[i, :] = weight_factor

    weighted_sky_pixels = np.sum(sky_mask * weight_mask)
    total_weighted_pixels = np.sum(weight_mask)
    weighted_sky_ratio = weighted_sky_pixels / total_weighted_pixels

    return weighted_sky_ratio > sky_percent

# ------------------ Configurations ------------------ #
MODEL_PATH = "LIME_20240506.best.hdf5"
MIN_PM25_VALUE = 20.0
MAX_FILE_SIZE = 10 * 1024 * 1024
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

    warnings = []
    if is_blurry(image):
        warnings.append("Image appears blurry.")
    if is_overexposed_or_underexposed(image):
        warnings.append("Image may be overexposed or underexposed.")
    if is_mostly_white_or_black(image):
        warnings.append("Image has large white/black areas.")
    if not is_sky_image(image):
        warnings.append("Image may not contain enough visible sky.")

    if warnings:
        st.warning("\n".join(warnings))

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting PM2.5 level..."):
        prediction = model.predict(img_array)
        pm25_value = float(prediction[0][0])
        pm25_value = max(0.0, min(1000.0, pm25_value))

    # Weather API Adjustment
    if "cloudy" in CITY.lower() or ("sky" in CITY.lower() and "cloud" in CITY.lower()):
        pass  # Skip redundant checks
    else:
        try:
            weather_url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={CITY}"
            weather_data = requests.get(weather_url).json()
            condition = weather_data.get("current", {}).get("condition", {}).get("text", "").lower()
            if "cloud" in condition or "overcast" in condition:
                st.info(f"☁️ Detected cloudy/overcast weather in {CITY.title()} via WeatherAPI. Prediction may be adjusted.")
                pm25_value = max(pm25_value, MIN_PM25_VALUE)
        except Exception as e:
            st.warning(f"Could not fetch weather data. Skipping weather adjustment.")

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
