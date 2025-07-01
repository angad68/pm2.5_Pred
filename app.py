import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import requests
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU
from skimage import measure
from scipy.ndimage import gaussian_filter
import logging

# ------------------ CONFIG ------------------ #
CONFIG = {
    "WEATHER_API_KEY": "7088853eac6948e286555436250107",
    "CITY": "Chandigarh",
    "MIN_PM25_VALUE": 20.0,
    "IMAGE_SIZE": (224, 224),
    "SKY_PERCENT": 0.3,
    "BLUR_THRESHOLD": 25.0,
    "LOW_EXPOSURE_THRESH": 35,
    "HIGH_EXPOSURE_THRESH": 220,
    "WHITE_THRESH": 235,
    "BLACK_THRESH": 25,
    "CLOUDY_PERCENT": 0.75,
    "CLOUD_COVER_THRESH": 75
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Weather API ------------------ #
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_weather_cloud_info(city=CONFIG["CITY"]):
    url = f"http://api.weatherapi.com/v1/current.json?key={CONFIG['WEATHER_API_KEY']}&q={city}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        cloud_percent = data["current"]["cloud"]
        logger.info(f"Retrieved cloud cover for {city}: {cloud_percent}%")
        return cloud_percent
    except Exception as e:
        logger.error(f"Failed to fetch weather data: {str(e)}")
        return None

# ------------------ Quality Checks ------------------ #
def is_blurry(pil_img, threshold=CONFIG["BLUR_THRESHOLD"]):
    img_gray = np.array(pil_img.convert("L"))
    blur_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    logger.info(f"Blur variance: {blur_var:.2f}")
    return blur_var < threshold

def is_overexposed_or_underexposed(pil_img, low_thresh=CONFIG["LOW_EXPOSURE_THRESH"], high_thresh=CONFIG["HIGH_EXPOSURE_THRESH"]):
    img_gray = np.array(pil_img.convert("L"))
    mean_val = np.mean(img_gray)
    logger.info(f"Image mean intensity: {mean_val:.2f}")
    return mean_val < low_thresh or mean_val > high_thresh

def is_mostly_white_or_black(pil_img, white_thresh=CONFIG["WHITE_THRESH"], black_thresh=CONFIG["BLACK_THRESH"], percent=CONFIG["CLOUDY_PERCENT"]):
    img = np.array(pil_img)
    white_pixels = np.sum(np.all(img > white_thresh, axis=2))
    black_pixels = np.sum(np.all(img < black_thresh, axis=2))
    total_pixels = img.shape[0] * img.shape[1]
    ratio = (white_pixels + black_pixels) / total_pixels
    logger.info(f"White/black pixel ratio: {ratio:.3f}")
    return ratio > percent

def is_sky_image(pil_img, sky_percent=CONFIG["SKY_PERCENT"], blue_hue_range=(90, 150)):
    img = pil_img.resize((256, 256)).convert("RGB")
    img_np = np.array(img)

    # Convert to HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Sky conditions
    overcast_mask = (s < 50) & (v > 50)
    blue_sky_mask = (h >= blue_hue_range[0]) & (h <= blue_hue_range[1]) & (v > 50)
    sky_mask = overcast_mask | blue_sky_mask

    # Apply Gaussian blur to reduce noise
    sky_mask = gaussian_filter(sky_mask.astype(float), sigma=2) > 0.5

    # Focus on upper half of the image
    height, width = sky_mask.shape
    upper_half_mask = np.zeros_like(sky_mask)
    upper_half_mask[:height//2, :] = sky_mask[:height//2, :]

    # Filter small regions
    labels = measure.label(upper_half_mask, background=0)
    if labels.max() > 0:
        largest_region = np.argmax(np.bincount(labels.flat)[1:]) + 1
        sky_mask = (labels == largest_region)

    # Calculate sky ratio
    sky_ratio = np.sum(sky_mask) / sky_mask.size
    logger.info(f"Sky ratio: {sky_ratio:.3f}, Overcast pixels: {np.sum(overcast_mask)/overcast_mask.size:.3f}, "
                f"Blue sky pixels: {np.sum(blue_sky_mask)/blue_sky_mask.size:.3f}")

    return sky_ratio > sky_percent

# ------------------ PM2.5 Category ------------------ #
def categorize_pm25(pm_value):
    thresholds = [
        (30, "Good"),
        (60, "Satisfactory"),
        (90, "Moderately Polluted"),
        (120, "Poor"),
        (250, "Very Poor"),
        (float("inf"), "Severe")
    ]
    for thresh, category in thresholds:
        if pm_value <= thresh:
            return category
    return "Severe"

# ------------------ Load Model ------------------ #
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
    try:
        model.load_weights("LIME_20240506.best.hdf5")
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model weights: {str(e)}")
        st.error("Failed to load model weights. Please check the file path.")
        st.stop()
    return model

# ------------------ UI ------------------ #
def main():
    st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
    st.title("🌫️ PM2.5 Level Predictor")
    st.write("Upload a **sky image** to predict PM2.5 air quality level. Ensure the image is clear, well-lit, and contains significant sky.")

    uploaded_file = st.file_uploader("Choose a sky image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Perform Quality Checks
            failed_checks = []
            if is_blurry(image):
                failed_checks.append("Image is blurry.")
            if is_overexposed_or_underexposed(image):
                failed_checks.append("Image is too dark or too bright.")
            if not is_sky_image(image):
                failed_checks.append("Image doesn't contain enough sky.")
            cloudy_looking = is_mostly_white_or_black(image)

            if failed_checks:
                st.error("⚠️ Image Quality Issues Detected:")
                for issue in failed_checks:
                    st.write(f"- {issue}")
                st.stop()

            # Prepare image for model
            img = image.resize(CONFIG["IMAGE_SIZE"])
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Cloud handling
            use_weather_override = False
            if cloudy_looking:
                cloud_percent = get_weather_cloud_info()
                if cloud_percent and cloud_percent > CONFIG["CLOUD_COVER_THRESH"]:
                    use_weather_override = True

            # Prediction
            model = load_pm25_model()
            if use_weather_override:
                st.info(f"☁️ Cloudy conditions detected (cloud cover: {cloud_percent}%). Prediction may be less accurate.\nReturning minimum plausible PM2.5 value.")
                pm25_value = CONFIG["MIN_PM25_VALUE"]
            else:
                prediction = model.predict(img_array, verbose=0)
                pm25_value = max(float(prediction[0][0]), CONFIG["MIN_PM25_VALUE"])  # Ensure non-negative

            category = categorize_pm25(pm25_value)

            # Display results
            st.subheader("📊 Prediction Results")
            st.metric("Predicted PM2.5 Value", f"{pm25_value:.2f} µg/m³")
            st.metric("Air Quality Category (India)", category)

            # Add visualization of sky mask for debugging
            if st.checkbox("Show sky detection mask (for debugging)"):
                img_np = np.array(image.resize((256, 256)).convert("RGB"))
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                sky_mask = ((h >= 90) & (h <= 150) & (v > 50)) | ((s < 50) & (v > 50))
                sky_mask = gaussian_filter(sky_mask.astype(float), sigma=2) > 0.5
                st.image(sky_mask * 255, caption="Sky Detection Mask (White = Sky)", use_column_width=True)

if __name__ == "__main__":
    main()
