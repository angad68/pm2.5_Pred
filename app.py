import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU, Dropout
import os
import requests

# ------------------ Theme ------------------ #
st.markdown("""
<style>
.stApp { background-color: #111111; color: #E0E0E0; }
h1, h2, h3, h4 { color: #F1F1F1; }
.stMetric { background-color: #222222; border-radius: 0.5rem; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ------------------ Constants ------------------ #
MODEL_PATH = "LIME_20240506.best.hdf5"
MIN_PM25_VALUE = 20.0
MAX_FILE_SIZE = 10 * 1024 * 1024
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "your_api_key_here")
CITY = os.getenv("CITY", "Chandigarh")
USE_UNCERTAINTY = True


# ------------------ Image Quality Checks ------------------ #
def is_blurry(pil_img, threshold=25.0):
    return cv2.Laplacian(np.array(pil_img.convert("L")), cv2.CV_64F).var() < threshold

def is_overexposed_or_underexposed(pil_img, low=35, high=220):
    mean_val = np.mean(np.array(pil_img.convert("L")))
    return mean_val < low or mean_val > high

def is_mostly_white_or_black(pil_img, white_thresh=235, black_thresh=25, percent=0.75):
    img = np.array(pil_img)
    white = np.sum(np.all(img > white_thresh, axis=2))
    black = np.sum(np.all(img < black_thresh, axis=2))
    return (white + black) / (img.shape[0] * img.shape[1]) > percent

def is_sky_image(pil_img, base_thresh=0.40):
    img = pil_img.resize((256, 256)).convert("RGB")
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    blue_sky = ((h >= 95) & (h <= 135)) & (s > 40) & (v > 80)
    light_sky = ((h >= 80) & (h <= 120)) & (s > 20) & (v > 120)
    gray_clouds = (s < 25) & (v > 120) & (v < 230)
    white_clouds = (v > 200) & (s < 35)
    sky_mask = blue_sky | light_sky | gray_clouds | white_clouds

    weight_mask = np.array([[1.5 - (i / 256.0)] * 256 for i in range(256)])
    weighted_ratio = np.sum(sky_mask * weight_mask) / np.sum(weight_mask)

    avg_brightness = np.mean(v)
    adaptive_thresh = base_thresh + 0.05 if avg_brightness > 180 else base_thresh - 0.05 if avg_brightness < 90 else base_thresh

    if weighted_ratio < adaptive_thresh: return False
    if np.std(h[sky_mask]) > 25: return False
    _, labels = cv2.connectedComponents(sky_mask.astype(np.uint8))
    if labels.max() <= 1: return False
    if max(np.bincount(labels.flatten())[1:]) < 0.02 * sky_mask.size: return False
    return True

def is_cloudy_image(pil_img, cloudy_thresh=0.25):
    img = pil_img.resize((256, 256)).convert("RGB")
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    cloud_mask = (s < 40) & (v > 80) & (v < 200)
    dark_cloud_mask = (s < 30) & (v >= 40) & (v <= 90)
    cloudy_combined = cloud_mask | dark_cloud_mask
    return np.mean(cloudy_combined) > cloudy_thresh

# ------------------ Model Loader ------------------ #
@st.cache_resource
def load_pm25_model():
    inputs = Input(shape=(224, 224, 3))
    x = inputs
    for f in [64, 64]:
        x = Conv2D(f, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    for f in [128, 128]:
        x = Conv2D(f, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    for _ in range(3):
        skip = x
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Add()([x, skip])
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    for f in [256, 256]:
        x = Conv2D(f, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Flatten()(x)
    for _ in range(2):
        x = Dense(1024)(x)
        x = Dropout(0.3)(x)
        x = LeakyReLU(alpha=0.1)(x)

    output = Dense(1, activation='linear')(x)
    model = Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mae')
    model.load_weights(MODEL_PATH)
    return model

model = load_pm25_model()

# ------------------ Weather API ------------------ #
def fetch_weather_data(city):
    try:
        url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"🌐 WeatherAPI error: {str(e)}")
        return None

# ------------------ Prediction ------------------ #
def predict_pm25(image):
    resized = image.resize((224, 224))
    img_array = np.array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    if USE_UNCERTAINTY:
        preds = [model(img_array, training=True).numpy().squeeze() for _ in range(30)]
        return float(np.mean(preds)), float(np.std(preds))
    else:
        val = float(model(img_array, training=False).numpy().squeeze())
        return val, 0.0

# ------------------ Categorization ------------------ #
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

# ------------------ App ------------------ #
st.title("🌫️ PM2.5 Air Quality Estimator")

input_mode = st.radio("Select input method:", ["📷 Use Webcam", "📁 Upload Image"])
image = None

if input_mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a sky image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("❌ File too large. Please upload < 10MB.")
            st.stop()
        try:
            image = Image.open(uploaded_file).convert("RGB")
            if min(image.size) < 100:
                st.error("❌ Image too small. Minimum 100x100 pixels required.")
                st.stop()
        except Exception as e:
            st.error(f"Invalid image: {str(e)}")
            st.stop()

elif input_mode == "📷 Use Webcam":
    cam_input = st.camera_input("Capture sky image")
    if cam_input:
        try:
            image = Image.open(cam_input).convert("RGB")
        except Exception as e:
            st.error(f"Camera capture failed: {str(e)}")
            st.stop()

# ------------------ If image available ------------------ #
if image:
    if is_blurry(image):
        st.error("Image is too blurry.")
        st.stop()
    if is_overexposed_or_underexposed(image):
        st.error("Image is over/under exposed.")
        st.stop()
    if is_mostly_white_or_black(image):
        st.error("Image is mostly white or black.")
        st.stop()
    if not is_sky_image(image):
        st.error("Image does not look like sky.")
        st.stop()

    pm25_val, pm25_std = predict_pm25(image)

if is_cloudy_image(image):
    weather_data = fetch_weather_data(CITY)
    if weather_data:
        condition = weather_data.get("current", {}).get("condition", {}).get("text", "").lower()
        if "cloud" in condition or "overcast" in condition:
            st.info(f"☁️ WeatherAPI: Cloudy in {CITY.title()} — adjusted value.")
            pm25_val = max(pm25_val, MIN_PM25_VALUE)

    # Display
    display_img = image.copy()
    display_img.thumbnail((500, 500))

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(display_img, caption="Captured Image")
    with col2:
        st.subheader("📊 Prediction Results")
        st.metric("PM2.5 Level", f"{pm25_val:.1f} µg/m³")
        st.metric("Uncertainty (±)", f"{pm25_std:.1f}")
        category = categorize_pm25(pm25_val)
        st.markdown(f"**Air Quality:** {colors[category]} {category}")
