import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import requests
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU

# ------------------ CONFIG ------------------ #
WEATHER_API_KEY = "7088853eac6948e286555436250107"  # Replace with your WeatherAPI key
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

# ------------------ Image Quality Checks ------------------ #
def is_blurry(pil_img, threshold=20.0):  # softened
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

def is_sky_image(pil_img, threshold=0.3):
    img = np.array(pil_img.resize((100, 100)))
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    blue_ratio = b / (r + g + b + 1e-5)
    return np.mean(blue_ratio) > threshold

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

# ------------------ Model Definition ------------------ #
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

    residual = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(residual)
    x = Add()([x, residual])
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    residual = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(residual)
    x = Add()([x, residual])
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    residual = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(residual)
    x = Add()([x, residual])
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

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
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=output)
    model.load_weights("LIME_20240506.best.hdf5")
    return model

model = load_pm25_model()

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("\U0001F32B️ PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict the PM2.5 air quality level.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- Validity Check ---
    if not is_sky_image(image):
        st.error("⚠️ Uploaded image doesn't appear to be a sky image. Please upload a clear sky photo.")
        st.stop()

    # --- Quality Checks ---
    issues = []
    cloudy_looking = False

    if is_blurry(image):
        issues.append("Image might be slightly blurry.")
    if is_overexposed_or_underexposed(image):
        issues.append("Image is a bit too dark or too bright.")
    if is_mostly_white_or_black(image):
        cloudy_looking = True

    if issues:
        st.warning("⚠️ Image Quality Notes:")
        for issue in issues:
            st.write(f"- {issue}")

    # --- Prediction ---
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    use_weather_adjustment = False
    if cloudy_looking:
        cloud_percent = get_weather_cloud_info(CITY)
        if cloud_percent is not None and cloud_percent > 75:
            use_weather_adjustment = True

    if use_weather_adjustment:
        st.info("☁️ It's currently cloudy in your region. Prediction may be affected — showing minimum plausible value.")
        pm25_value = MIN_PM25_VALUE
    else:
        prediction = model.predict(img_array)
        pm25_value = float(prediction[0][0])

    category = categorize_pm25(pm25_value)

    st.subheader("\U0001F4CA Prediction")
    st.write(f"**Predicted PM2.5 Value:** {pm25_value:.2f} µg/m³")
    st.write(f"**Air Quality Category (India):** {category}")
