import streamlit as st
import numpy as np
from PIL import Image
import cv2
import requests
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU

# ------------------ CONFIG ------------------ #

WEATHER_API_KEY = "7088853eac6948e286555436250107"  # Replace with your WeatherAPI key
CITY = "Chandigarh"

# ------------------ Weather API ------------------ #

def get_weather_data(city=CITY):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    try:
        response = requests.get(url)
        data = response.json()
        return {
            "cloud": data["current"]["cloud"],
            "humidity": data["current"]["humidity"],
            "wind_kph": data["current"]["wind_kph"],
            "desc": data["current"]["condition"]["text"]
        }
    except Exception:
        return None

# ------------------ Image Quality Checks ------------------ #

def is_blurry(pil_img, threshold=50.0):
    img_gray = np.array(pil_img.convert("L"))
    return cv2.Laplacian(img_gray, cv2.CV_64F).var() < threshold

def is_overexposed_or_underexposed(pil_img, low_thresh=30, high_thresh=240):
    img_gray = np.array(pil_img.convert("L"))
    mean_val = np.mean(img_gray)
    return mean_val < low_thresh or mean_val > high_thresh

def is_mostly_white_or_black(pil_img, white_thresh=240, black_thresh=20, percent=0.75):
    img = np.array(pil_img)
    white_pixels = np.sum(np.all(img > white_thresh, axis=2))
    black_pixels = np.sum(np.all(img < black_thresh, axis=2))
    total_pixels = img.shape[0] * img.shape[1]
    return (white_pixels + black_pixels) / total_pixels > percent

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

# ------------------ Model Loader ------------------ #

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

    x = Conv2D(128, (3, 3), padding='same')(pool2)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, pool2])
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, pool2])
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, pool2])
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

# ------------------ Streamlit App ------------------ #

st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict the PM2.5 air quality level. Try to avoid blurry, dark or blocked images.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Image Quality Checks
    issues = []
    if is_blurry(image): issues.append("Image appears blurry.")
    if is_overexposed_or_underexposed(image): issues.append("Image is too dark or too bright.")
    if is_mostly_white_or_black(image): issues.append("Image contains too much black or white — possibly blocked or indoor.")

    if issues:
        st.error("⚠️ Image Quality Issues Detected:")
        for issue in issues:
            st.write(f"- {issue}")
        st.stop()

    # Run Prediction
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = load_pm25_model()
    prediction = model.predict(img_array)
    pm25_value = float(prediction[0][0])
    category = categorize_pm25(pm25_value)

    st.subheader("📊 Prediction")
    st.write(f"**Predicted PM2.5 Value:** {pm25_value:.2f} µg/m³")
    st.write(f"**Air Quality Category (India):** {category}")

    # Weather Data & Cloud Warning
    weather = get_weather_data(CITY)
    if weather:
        st.subheader(f"🌦️ Current Weather: {CITY}")
        st.write(f"**Condition:** {weather['desc']}")
        st.write(f"**Cloud Cover:** {weather['cloud']}%")
        st.write(f"**Humidity:** {weather['humidity']}%")
        st.write(f"**Wind:** {weather['wind_kph']} km/h")

        if weather['cloud'] > 80 and pm25_value > 100:
            st.warning("⚠️ Heavy cloud cover detected. Prediction might overestimate PM2.5 due to cloudy appearance.")
    else:
        st.info("Live weather data not available at the moment.")
