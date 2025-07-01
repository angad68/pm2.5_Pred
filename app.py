import streamlit as st
import numpy as np
from PIL import Image
import cv2
import requests
from tensorflow.keras.models import load_model

# ------------------ CONFIG ------------------ #
WEATHER_API_KEY = "7088853eac6948e286555436250107"  # Replace with your key
CITY = "Chandigarh"
MIN_PM25_VALUE = 20.0

# ------------------ Weather API ------------------ #
def get_weather_cloud_info(city=CITY):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        response = requests.get(url)
        data = response.json()
        return data["current"]["cloud"]
    except:
        return None

# ------------------ Quality Checks ------------------ #
def is_blurry(pil_img, threshold=25.0):  # Softened
    img_gray = np.array(pil_img.convert("L"))
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return laplacian_var < threshold

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

def is_sky_like(pil_img):
    img = np.array(pil_img.resize((128, 128)))
    b = img[:, :, 2]
    g = img[:, :, 1]
    r = img[:, :, 0]
    sky_mask = (b > r + 10) & (b > g + 10)
    sky_coverage = np.sum(sky_mask) / sky_mask.size
    return sky_coverage > 0.15  # relaxed threshold

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

# ------------------ Load Model ------------------ #
@st.cache_resource
def load_pm25_model():
    return load_model("LIME_20240506.best.hdf5")

model = load_pm25_model()

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict PM2.5 air quality.")

uploaded_file = st.file_uploader("Choose a sky image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- Quality Checks ---
    issues = []
    should_abort = False
    cloudy_looking = False

    if not is_sky_like(image):
        issues.append("Image doesn't look like a sky photo.")
        should_abort = True

    if is_blurry(image):
        issues.append("Image might be blurry.")
        should_abort = True

    if is_overexposed_or_underexposed(image):
        issues.append("Image is too bright or too dark.")
        should_abort = True

    if is_mostly_white_or_black(image):
        issues.append("Sky seems fully covered (cloudy/washed).")
        cloudy_looking = True

    if issues:
        st.warning("⚠️ Image Quality Issues Detected:")
        for issue in issues:
            st.write(f"- {issue}")

    if should_abort:
        st.error("⛔ This image is unsuitable for prediction. Please upload a clearer sky image.")
    else:
        # --- Prediction Block ---
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        use_weather_adjustment = False
        if cloudy_looking:
            cloud_percent = get_weather_cloud_info(CITY)
            if cloud_percent and cloud_percent > 75:
                use_weather_adjustment = True

        if use_weather_adjustment:
            st.info("☁️ Cloudy conditions detected in your region. Prediction might be affected. Using minimum plausible value.")
            pm25_value = MIN_PM25_VALUE
        else:
            prediction = model.predict(img_array)
            pm25_value = float(prediction[0][0])

        category = categorize_pm25(pm25_value)

        st.subheader("📊 Prediction")
        st.write(f"**Predicted PM2.5:** {pm25_value:.2f} µg/m³")
        st.write(f"**Air Quality Category:** {category}")
