import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import requests
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU

# ------------------ CONFIG ------------------ #
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "7088853eac6948e286555436250107")
CITY = os.getenv("CITY", "Chandigarh")
MIN_PM25_VALUE = 20.0
MODEL_PATH = os.getenv("MODEL_PATH", "LIME_20240506.best.hdf5")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ------------------ Weather API ------------------ #
def get_weather_cloud_info(city=CITY):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["current"]["cloud"]
    except requests.RequestException as e:
        st.warning(f"Weather API unavailable: {str(e)}")
        return None
    except KeyError:
        st.warning("Unexpected weather API response format")
        return None

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
    
    # Multiple sky detection criteria
    blue_sky = ((h >= 100) & (h <= 130)) & (s > 30) & (v > 80)
    light_blue = ((h >= 80) & (h <= 110)) & (s > 15) & (v > 100)
    gray_sky = (s < 30) & (v > 120) & (v < 220)
    bright_areas = (v > 200) & (s < 50)
    
    sky_mask = blue_sky | light_blue | gray_sky | bright_areas
    
    # Weight upper portion more heavily
    height, width = sky_mask.shape
    weight_mask = np.ones_like(sky_mask, dtype=float)
    
    for i in range(height):
        weight_factor = 1.0 + (height - i) / height
        weight_mask[i, :] = weight_factor
    
    weighted_sky_pixels = np.sum(sky_mask * weight_mask)
    total_weighted_pixels = np.sum(weight_mask)
    weighted_sky_ratio = weighted_sky_pixels / total_weighted_pixels
    
    return weighted_sky_ratio > sky_percent

def is_cloudy_image(pil_img):
    img = pil_img.resize((128, 128)).convert("RGB")
    img_np = np.array(img)
    
    # Focus on upper portion where clouds typically appear
    top_portion = img_np[:int(0.7 * img_np.shape[0]), :]
    hsv_top = cv2.cvtColor(top_portion, cv2.COLOR_RGB2HSV)
    h_top, s_top, v_top = cv2.split(hsv_top)
    
    # Cloud detection: high brightness, low saturation
    cloud_mask = (s_top < 25) & (v_top > 150)
    cloud_ratio = np.sum(cloud_mask) / cloud_mask.size
    
    return cloud_ratio > 0.4

def comprehensive_image_check(pil_img):
    """Return a dict with all check results"""
    checks = {
        'is_blurry': is_blurry(pil_img),
        'is_poorly_exposed': is_overexposed_or_underexposed(pil_img),
        'has_sufficient_sky': is_sky_image(pil_img),
        'is_mostly_uniform': is_mostly_white_or_black(pil_img),
        'is_cloudy': is_cloudy_image(pil_img)
    }
    return checks

# ------------------ PM2.5 Category ------------------ #
def categorize_pm25(pm_value):
    if pm_value <= 30:
        return "Good"
    elif pm_value <= 60:
        return "Satisfactory"
    elif pm_value <= 90:
        return "Moderately Polluted"
    elif pm_value <= 120:
        return "Poor"
    elif pm_value <= 250:
        return "Very Poor"
    else:
        return "Severe"

def validate_pm25_prediction(prediction_value):
    """Ensure PM2.5 prediction is within reasonable bounds"""
    return max(0.0, min(1000.0, float(prediction_value)))

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
        st.error("Please ensure the model file 'LIME_20240506.best.hdf5' is in the correct directory.")
        st.stop()

model = load_pm25_model()

# ------------------ UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict PM2.5 air quality level. No indoor or obstructed photos.")

uploaded_file = st.file_uploader("Choose a sky image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Validate file size
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File too large. Please upload an image smaller than 10MB.")
        st.stop()
    
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Validate image dimensions
        if image.size[0] < 100 or image.size[1] < 100:
            st.error("Image too small. Please upload an image at least 100x100 pixels.")
            st.stop()
            
    except Exception as e:
        st.error(f"Invalid image file: {str(e)}")
        st.stop()
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform comprehensive quality checks
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
        st.error("⚠️ Image Quality Issues:")
        for i, issue in enumerate(failed_checks, 1):
            st.write(f"{i}. {issue}")
        st.info("💡 **Tips:** Take photos outdoors pointing upward at the sky, ensure good lighting and focus.")
        st.stop()
    
    # Prepare image for prediction
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Cloud handling with weather API
    use_weather_override = False
    if cloudy_looking:
        cloud_percent = get_weather_cloud_info(CITY)
        if cloud_percent and cloud_percent > 75:
            use_weather_override = True
    
    # Make prediction
    with st.spinner("Predicting PM2.5 level..."):
        if use_weather_override:
            st.info("☁️ Heavy cloud cover detected. Using minimum baseline value.")
            pm25_value = MIN_PM25_VALUE
        else:
            prediction = model.predict(img_array)
            pm25_value = validate_pm25_prediction(prediction[0][0])
    
    category = categorize_pm25(pm25_value)
    
    # Display results with improved UI
    st.subheader("📊 Prediction Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PM2.5 Level", f"{pm25_value:.1f} µg/m³")
    with col2:
        # Color-coded category display
        colors = {
            "Good": "🟢", 
            "Satisfactory": "🟡", 
            "Moderately Polluted": "🟠", 
            "Poor": "🔴", 
            "Very Poor": "🟣", 
            "Severe": "🔴"
        }
        st.metric("Air Quality", f"{colors.get(category, '⚪')} {category}")
    
    # Additional information
    if cloudy_looking and not use_weather_override:
        st.info("ℹ️ Cloudy conditions detected in image. Prediction accuracy may be affected.")
    
    # Health recommendations based on category
    health_advice = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities!",
        "Satisfactory": "Air quality is acceptable for most people.",
        "Moderately Polluted": "Sensitive individuals should limit prolonged outdoor activities.",
        "Poor": "Everyone should reduce prolonged outdoor activities.",
        "Very Poor": "Avoid outdoor activities. Keep windows closed.",
        "Severe": "Stay indoors. Avoid all outdoor activities."
    }
    
    if category in health_advice:
        st.info(f"💡 **Health Advice:** {health_advice[category]}")
