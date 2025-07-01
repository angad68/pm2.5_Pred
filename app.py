import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU, Dropout
import os
import requests

# ------------------ Dark Theme CSS ------------------ #
st.markdown("""
    <style>
    .stApp {
        background-color: #111111;
        color: #E0E0E0;
    }
    .block-container {
        background-color: #111111;
        padding: 2rem 1rem 2rem 1rem;
    }
    .stFileUploader {
        color: #E0E0E0;
    }
    .css-1d391kg, .css-18ni7ap {
        background-color: #111111 !important;
    }
    h1, h2, h3, h4 {
        color: #F1F1F1;
    }
    .stTitle {
        text-align: center;
    }
    .stMetric {
        background-color: #222222;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

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

import numpy as np
import cv2
from PIL import Image

def is_sky_image(pil_img, base_thresh=0.40):
    """
    Enhanced sky detection using HSV heuristics + structural checks.
    Returns True if the image likely contains a significant amount of sky.
    """
    # Step 1: Resize and convert
    img = pil_img.resize((256, 256)).convert("RGB")
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Step 2: Create HSV masks for various sky types
    blue_sky = ((h >= 95) & (h <= 135)) & (s > 40) & (v > 80)
    light_sky = ((h >= 80) & (h <= 120)) & (s > 20) & (v > 120)
    gray_clouds = (s < 25) & (v > 120) & (v < 230)
    white_clouds = (v > 200) & (s < 35)

    sky_mask = blue_sky | light_sky | gray_clouds | white_clouds

    # Step 3: Top-heavy weighting
    height, width = sky_mask.shape
    weight_mask = np.ones_like(sky_mask, dtype=np.float32)
    for i in range(height):
        weight_mask[i, :] = 1.5 - (i / height)  # More weight at top

    # Step 4: Weighted sky ratio
    weighted_sky_pixels = np.sum(sky_mask * weight_mask)
    total_weight = np.sum(weight_mask)
    weighted_sky_ratio = weighted_sky_pixels / total_weight

    # Step 5: Adaptive threshold based on brightness
    avg_brightness = np.mean(v)
    adaptive_thresh = base_thresh
    if avg_brightness < 90:
        adaptive_thresh -= 0.05
    elif avg_brightness > 180:
        adaptive_thresh += 0.05

    if weighted_sky_ratio < adaptive_thresh:
        return False  # Not enough sky

    # Step 6: Hue consistency check
    sky_hues = h[sky_mask]
    if sky_hues.size > 0 and np.std(sky_hues) > 25:
        return False  # Hue too variable, not consistent sky

    # Step 7: Component size check
    num_labels, labels = cv2.connectedComponents(sky_mask.astype(np.uint8))
    if num_labels <= 1:
        return False  # No sky component
    largest_component = max(np.bincount(labels.flatten())[1:])  # Ignore background
    if largest_component < 0.02 * sky_mask.size:
        return False  # Too small to be meaningful

    return True  # Sky detected


# ------------------ Constants ------------------ #
MODEL_PATH = "LIME_20240506.best.hdf5"
MIN_PM25_VALUE = 20.0
MAX_FILE_SIZE = 10 * 1024 * 1024
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "7088853eac6948e286555436250107")
CITY = os.getenv("CITY", "Chandigarh")

# ------------------ Load CNN Model ------------------ #
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

# ------------------ Weather API ------------------ #
@st.cache_data
def fetch_weather_data(city):
    try:
        url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        return requests.get(url).json()
    except Exception as e:
        st.warning(f"⚠️ Could not fetch weather data: {str(e)}")
        return None

# ------------------ Prediction ------------------ #
def predict_pm25(image):
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = [model(img_array, training=True).numpy().squeeze() for _ in range(30)]
    return float(np.mean(preds)), float(np.std(preds))

# ------------------ App Layout ------------------ #
st.title("🌫️ PM2.5 Air Quality Estimator")

st.subheader("📤 Upload Image")
uploaded_file = st.file_uploader("Choose a sky image (JPG/PNG, < 10MB)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File too large. Upload image < 10MB.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
        if image.size[0] < 100 or image.size[1] < 100:
            st.error("Image too small. Minimum 100x100 pixels required.")
            st.stop()
    except Exception as e:
        st.error(f"Invalid image file: {str(e)}")
        st.stop()

    # Quality Checks
    if is_blurry(image):
        st.error("Too blurry.")
        st.stop()
    if is_overexposed_or_underexposed(image):
        st.error("Over/under exposed.")
        st.stop()
    if is_mostly_white_or_black(image):
        st.error("Mostly white or black.")
        st.stop()
    if not is_sky_image(image):
        st.error("Does not look like a sky image.")
        st.stop()

    # Predict
    pm25_value, pm25_std = predict_pm25(image)

    # Weather adjustment
    weather_data = fetch_weather_data(CITY)
    if weather_data:
        condition = weather_data.get("current", {}).get("condition", {}).get("text", "").lower()
        if "cloud" in condition or "overcast" in condition:
            st.info(f"☁️ Cloudy in {CITY.title()}. Adjusted prediction.")
            pm25_value = max(pm25_value, MIN_PM25_VALUE)

    # Categorize
    def categorize_pm25(val):
        if val <= 30: return "Good"
        elif val <= 60: return "Satisfactory"
        elif val <= 90: return "Moderately Polluted"
        elif val <= 120: return "Poor"
        elif val <= 250: return "Very Poor"
        return "Severe"

    colors = {
        "Good": "🟢", "Satisfactory": "🟡", "Moderately Polluted": "🟠",
        "Poor": "🔴", "Very Poor": "🟣", "Severe": "🔴"
    }

    # Display
    display_img = image.copy()
    display_img.thumbnail((500, 500))

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(display_img, caption="Uploaded Image")
    with col2:
        st.subheader("📊 Prediction Results")
        st.metric("PM2.5 Level", f"{pm25_value:.1f} µg/m³")
        st.metric("Uncertainty (±)", f"{pm25_std:.1f}")
        st.markdown(f"**Air Quality:** {colors[categorize_pm25(pm25_value)]} {categorize_pm25(pm25_value)}")
