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

# ------------------ Weather API ------------------ #
def get_weather_cloud_info(city=CITY):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    try:
        response = requests.get(url)
        data = response.json()
        return data["current"]["cloud"]
    except Exception:
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
    # Resize for faster processing
    img = pil_img.resize((256, 256)).convert("RGB")
    img_np = np.array(img)
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Multiple sky detection criteria
    sky_mask = np.zeros_like(h, dtype=bool)
    
    # 1. Blue sky (clear sky)
    blue_sky = ((h >= 100) & (h <= 130)) & (s > 30) & (v > 80)
    
    # 2. Light blue/cyan sky
    light_blue = ((h >= 80) & (h <= 110)) & (s > 15) & (v > 100)
    
    # 3. Overcast/gray sky (low saturation, medium-high brightness)
    gray_sky = (s < 30) & (v > 120) & (v < 220)
    
    # 4. White/very bright areas (clouds, bright overcast)
    bright_areas = (v > 200) & (s < 50)
    
    # Combine all sky conditions
    sky_mask = blue_sky | light_blue | gray_sky | bright_areas
    
    # Weight upper portion of image more heavily (sky is usually at top)
    height, width = sky_mask.shape
    weight_mask = np.ones_like(sky_mask, dtype=float)
    
    # Create gradient weight (top gets more weight)
    for i in range(height):
        weight_factor = 1.0 + (height - i) / height  # Top rows get higher weight
        weight_mask[i, :] = weight_factor
    
    # Calculate weighted sky ratio
    weighted_sky_pixels = np.sum(sky_mask * weight_mask)
    total_weighted_pixels = np.sum(weight_mask)
    weighted_sky_ratio = weighted_sky_pixels / total_weighted_pixels
    
    return weighted_sky_ratio > sky_percent

# Alternative simpler version focusing on spatial analysis
def is_sky_image_simple(pil_img, sky_percent=0.4):
    img = pil_img.resize((128, 128)).convert("RGB")
    img_np = np.array(img)
    
    # Focus on top 60% of image where sky usually is
    top_portion = img_np[:int(0.6 * img_np.shape[0]), :]
    
    hsv_top = cv2.cvtColor(top_portion, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_top)
    
    # Multiple sky conditions
    blue_sky = ((h >= 90) & (h <= 140)) & (s > 20)
    gray_sky = (s < 40) & (v > 100)
    bright_sky = (v > 180) & (s < 60)
    
    sky_mask = blue_sky | gray_sky | bright_sky
    sky_ratio = np.sum(sky_mask) / sky_mask.size
    
    return sky_ratio > sky_percent


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
    model.load_weights("LIME_20240506.best.hdf5")
    return model

model = load_pm25_model()

# ------------------ UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict PM2.5 air quality level. No indoor or obstructed photos.")

uploaded_file = st.file_uploader("Choose a sky image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform Quality Checks
    failed_checks = []

    if is_blurry(image): failed_checks.append("Image is blurry.")
    if is_overexposed_or_underexposed(image): failed_checks.append("Image is too dark or too bright.")
    if not is_sky_image(image): failed_checks.append("Image doesn't seem to contain enough sky.")
    if is_mostly_white_or_black(image): cloudy_looking = True
    else: cloudy_looking = False

    if failed_checks:
        st.error("⚠️ Image Quality Issues Detected:")
        for issue in failed_checks:
            st.write(f"- {issue}")
        st.stop()

    # Prepare image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Cloud handling
    use_weather_override = False
    if cloudy_looking:
        cloud_percent = get_weather_cloud_info(CITY)
        if cloud_percent and cloud_percent > 75:
            use_weather_override = True

    if use_weather_override:
        st.info("☁️ It's cloudy in this region. Prediction may not be accurate.\nReturning minimum plausible PM2.5 value.")
        pm25_value = MIN_PM25_VALUE
    else:
        prediction = model.predict(img_array)
        pm25_value = float(prediction[0][0])

    category = categorize_pm25(pm25_value)

    st.subheader("📊 Prediction")
    st.write(f"**Predicted PM2.5 Value:** {pm25_value:.2f} µg/m³")
    st.write(f"**Air Quality Category (India):** {category}")
