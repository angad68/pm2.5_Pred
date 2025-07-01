import streamlit as st
import numpy as np
import requests
import cv2
from PIL import Image
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU

# ------------------ CONFIG ------------------ #
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "7088853eac6948e286555436250107")
CITY = os.getenv("CITY", "Chandigarh")
MIN_PM25_VALUE = 20.0
MODEL_PATH = os.getenv("MODEL_PATH", "LIME_20240506.best.hdf5")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ------------------ UTILITY FUNCTIONS ------------------ #
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
    return max(0.0, min(1000.0, float(prediction_value)))

def get_comprehensive_weather_info(city=CITY):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data["current"]
        return {
            'cloud_cover': current.get("cloud", 0),
            'visibility_km': current.get("vis_km", 10),
            'condition': current.get("condition", {}).get("text", ""),
            'humidity': current.get("humidity", 50),
            'wind_kph': current.get("wind_kph", 0),
            'temp_c': current.get("temp_c", 20)
        }
    except:
        return None

def adjust_prediction_with_weather(base_prediction, weather_data, is_cloudy_image):
    if not weather_data:
        return base_prediction, []

    adjusted_prediction = base_prediction
    adjustments = []

    if weather_data['visibility_km'] < 5:
        adjusted_prediction *= 1.3
        adjustments.append(f"Low visibility ({weather_data['visibility_km']}km)")
    elif weather_data['visibility_km'] < 2:
        adjusted_prediction *= 1.6
        adjustments.append(f"Very low visibility ({weather_data['visibility_km']}km)")

    if weather_data['humidity'] > 80:
        adjusted_prediction *= 1.1
        adjustments.append(f"High humidity ({weather_data['humidity']}%)")

    if is_cloudy_image and weather_data['cloud_cover'] > 80:
        adjusted_prediction = max(adjusted_prediction, MIN_PM25_VALUE * 1.5)
        adjustments.append(f"Heavy overcast ({weather_data['cloud_cover']}% clouds)")

    fog_conditions = ['fog', 'mist', 'haze', 'smog']
    if any(cond in weather_data['condition'].lower() for cond in fog_conditions):
        adjusted_prediction *= 1.4
        adjustments.append(f"Foggy: {weather_data['condition']}")

    return min(adjusted_prediction, 500.0), adjustments

# ------------------ IMAGE QUALITY CHECKS ------------------ #
def is_blurry(pil_img, threshold=25.0):
    img_gray = np.array(pil_img.convert("L"))
    return cv2.Laplacian(img_gray, cv2.CV_64F).var() < threshold

def is_overexposed_or_underexposed(pil_img, low_thresh=35, high_thresh=220):
    img_gray = np.array(pil_img.convert("L"))
    mean_val = np.mean(img_gray)
    return mean_val < low_thresh or mean_val > high_thresh

def is_sky_image(pil_img):
    img = pil_img.resize((224, 224)).convert("RGB")
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    sky_mask = ((h >= 90) & (h <= 140) & (s > 20) & (v > 100)) | ((s < 30) & (v > 160))
    sky_ratio = np.sum(sky_mask) / sky_mask.size
    return sky_ratio > 0.3

def comprehensive_image_check(pil_img):
    return {
        'is_blurry': is_blurry(pil_img),
        'is_poorly_exposed': is_overexposed_or_underexposed(pil_img),
        'has_sufficient_sky': is_sky_image(pil_img),
        'is_cloudy': True  # Optional placeholder if cloud detection isn't added
    }

# ------------------ LOAD MODEL ------------------ #
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
        st.error(f"Model load failed: {e}")
        st.stop()

model = load_pm25_model()

# ------------------ STREAMLIT UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict PM2.5 air quality level.")

uploaded_file = st.file_uploader("Choose a sky image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("Image file too large. Please upload <10MB.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
        if image.size[0] < 100 or image.size[1] < 100:
            st.error("Image too small. Please upload >100x100.")
            st.stop()
    except Exception as e:
        st.error(f"Invalid image: {e}")
        st.stop()

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image quality..."):
        checks = comprehensive_image_check(image)
    issues = []
    if checks['is_blurry']: issues.append("Image is blurry")
    if checks['is_poorly_exposed']: issues.append("Image is poorly exposed")
    if not checks['has_sufficient_sky']: issues.append("Not enough visible sky")

    if issues:
        st.error("⚠️ Image issues detected:")
        for i, issue in enumerate(issues, 1):
            st.write(f"{i}. {issue}")
        st.stop()

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    weather = get_comprehensive_weather_info()
    prediction = model.predict(img_array)
    base_pm25 = validate_pm25_prediction(prediction[0][0])
    pm25_value, weather_notes = adjust_prediction_with_weather(base_pm25, weather, checks['is_cloudy'])

    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PM2.5 Level", f"{pm25_value:.1f} µg/m³")
    with col2:
        st.metric("Category", categorize_pm25(pm25_value))

    if weather_notes:
        st.write("**Weather adjustments applied:**")
        for note in weather_notes:
            st.write(f"- {note}")
