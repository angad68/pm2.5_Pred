import streamlit as st
import numpy as np
from PIL import Image
import cv2
import requests
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Add

# ------------------ CONFIG ------------------ #
WEATHER_API_KEY = "7088853eac6948e286555436250107"
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

# ------------------ Image Quality Checks (Soft Warnings) ------------------ #
def is_blurry(pil_img, threshold=25.0):
    img_gray = np.array(pil_img.convert("L"))
    return cv2.Laplacian(img_gray, cv2.CV_64F).var() < threshold

def is_dark_or_bright(pil_img, low_thresh=35, high_thresh=220):
    mean_val = np.mean(np.array(pil_img.convert("L")))
    return mean_val < low_thresh or mean_val > high_thresh

def is_mostly_white_or_black(pil_img, white_thresh=235, black_thresh=25, percent=0.75):
    img = np.array(pil_img)
    white_pixels = np.sum(np.all(img > white_thresh, axis=2))
    black_pixels = np.sum(np.all(img < black_thresh, axis=2))
    total = img.shape[0] * img.shape[1]
    return (white_pixels + black_pixels) / total > percent

# ------------------ CNN Model Loader ------------------ #
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
    res2 = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(res2)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, res2])
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, x])
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, x])
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
    output = Dense(1, activation='linear', name="PM2.5_output")(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
    model.load_weights("LIME_20240506.best.hdf5")
    return model

model = load_pm25_model()

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict PM2.5 air quality.")

uploaded_file = st.file_uploader("Choose a sky image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- Warnings (not blocking) ---
    warnings = []
    if is_blurry(image):
        warnings.append("Image might be blurry.")
    if is_dark_or_bright(image):
        warnings.append("Image may be over/underexposed.")
    if is_mostly_white_or_black(image):
        warnings.append("Image appears washed out or cloudy.")

    if warnings:
        st.warning("⚠️ Image Quality Notes:")
        for w in warnings:
            st.write(f"- {w}")

    # --- Resize and Predict ---
    img = image.resize((224, 224))  # keep exactly how you trained
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    use_weather_adjustment = False
    if is_mostly_white_or_black(image):
        cloud_percent = get_weather_cloud_info(CITY)
        if cloud_percent and cloud_percent > 75:
            use_weather_adjustment = True

    if use_weather_adjustment:
        st.info("☁️ Cloudy weather detected via API. Using minimum PM2.5 value.")
        pm25_value = MIN_PM25_VALUE
    else:
        prediction = model.predict(img_array, verbose=0)
        pm25_value = float(prediction[0][0])

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

    category = categorize_pm25(pm25_value)

    st.subheader("📊 Prediction")
    st.write(f"**Predicted PM2.5:** {pm25_value:.2f} µg/m³")
    st.write(f"**Air Quality Category:** {category}")
