import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import requests
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU

# ------------------ CONFIG ------------------ #
WEATHER_API_KEY = "7088853eac6948e286555436250107"
CITY = "Chandigarh"
MIN_PM25_VALUE = 20.0
SKY_THRESHOLD = 0.3  # Minimum sky percentage for valid image

# ------------------ Weather API ------------------ #
def get_weather_cloud_info(city=CITY):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        return data["current"]["cloud"]
    except Exception as e:
        st.error(f"⚠️ Weather API error: {str(e)}")
        return None

# ------------------ Quality Checks ------------------ #
def is_blurry(pil_img, threshold=25.0):
    """Detect blur using Laplacian variance [2]"""
    img_gray = np.array(pil_img.convert("L"))
    return cv2.Laplacian(img_gray, cv2.CV_64F).var() < threshold

def is_overexposed_or_underexposed(pil_img, low_thresh=35, high_thresh=220):
    """Check exposure levels using mean intensity"""
    img_gray = np.array(pil_img.convert("L"))
    mean_val = np.mean(img_gray)
    return mean_val < low_thresh or mean_val > high_thresh

def is_mostly_white_or_black(pil_img, white_thresh=235, black_thresh=25, percent=0.75):
    """Detect monochromatic dominance"""
    img = np.array(pil_img)
    white_pixels = np.sum(np.all(img > white_thresh, axis=2))
    black_pixels = np.sum(np.all(img < black_thresh, axis=2))
    total_pixels = img.shape[0] * img.shape[1]
    return (white_pixels + black_pixels) / total_pixels > percent

def is_sky_image(pil_img):
    """Improved sky detection using variance method [3]"""
    img = pil_img.resize((256, 256)).convert("L")
    img_np = np.array(img)
    
    # Focus on top 50% where sky appears
    top_portion = img_np[:int(0.5 * img_np.shape[0]), :]
    
    # Calculate variance (sky has lower variance)
    variance = np.var(top_portion)
    return variance < 500  # Empirical threshold

# ------------------ PM2.5 Category ------------------ #
def categorize_pm25(pm_value):
    """AQI categorization"""
    if pm_value <= 30: return "Good"
    elif pm_value <= 60: return "Satisfactory"
    elif pm_value <= 90: return "Moderately Polluted"
    elif pm_value <= 120: return "Poor"
    elif pm_value <= 250: return "Very Poor"
    else: return "Severe"

# ------------------ Model Loading ------------------ #
@st.cache_resource
def load_pm25_model():
    """Load model with proper weight initialization [4]"""
    inputs = Input(shape=(224, 224, 3))
    # ... (same architecture as before) ...
    model = Model(inputs=inputs, outputs=output)
    
    try:
        model.load_weights("LIME_20240506.best.hdf5")
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None
    return model

# ------------------ UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a **clear sky image** for prediction (no obstructions)")

uploaded_file = st.file_uploader("Choose sky image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform Quality Checks
    quality_issues = []
    if is_blurry(image): 
        quality_issues.append("Image is blurry (low sharpness)")
    if is_overexposed_or_underexposed(image): 
        quality_issues.append("Exposure issues (too dark/bright)")
    if not is_sky_image(image): 
        quality_issues.append("Insufficient sky content")
    if is_mostly_white_or_black(image): 
        quality_issues.append("Monochromatic dominance")
    
    if quality_issues:
        st.error("❌ Image rejected:")
        for issue in quality_issues:
            st.write(f"- {issue}")
        st.stop()
    
    # Model prediction
    model = load_pm25_model()
    if model is None:
        st.stop()
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Cloud handling
    cloud_override = False
    if cloud_percent := get_weather_cloud_info():
        if cloud_percent > 75:
            cloud_override = True
            st.info(f"☁️ Cloudy conditions ({cloud_percent}%) - using minimum PM2.5")

    pm25_value = MIN_PM25_VALUE if cloud_override else float(model.predict(img_array)[0][0])
    
    # Results display
    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PM2.5 Level", f"{pm25_value:.2f} µg/m³")
    with col2:
        st.metric("Air Quality", categorize_pm25(pm25_value))
