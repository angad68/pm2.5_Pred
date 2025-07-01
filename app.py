import streamlit as st
import numpy as np
from PIL import Image
import cv2
import requests
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU
from sklearn.cluster import KMeans

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

# ------------------ Image Quality & Sky Validity Checks ------------------ #
def is_blurry(pil_img, threshold=25.0):
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

def is_sky_image(pil_img, blue_thresh=100, sky_percent=0.25):
    img = np.array(pil_img.resize((128, 128)))
    b = img[:, :, 2]
    g = img[:, :, 1]
    r = img[:, :, 0]
    sky_mask = (b > r + 15) & (b > g + 15) & (b > blue_thresh)
    return np.sum(sky_mask) / sky_mask.size > sky_percent

def is_low_texture(pil_img, edge_thresh=10.0):
    gray = np.array(pil_img.convert("L"))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density < edge_thresh / 100.0

def is_color_uniform(pil_img, k=3, threshold=0.85):
    img = np.array(pil_img.resize((64, 64)))
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant_ratio = np.max(counts) / np.sum(counts)
    return dominant_ratio > threshold

def is_valid_sky_image(pil_img):
    return (
        is_sky_image(pil_img)
        and not is_low_texture(pil_img)
        and not is_color_uniform(pil_img)
    )

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
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    res = Conv2D(128, (3, 3), padding='same')(x)
    res = LeakyReLU(0.1)(res)
    x = Add()([res, x])
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(0.1)(x)
    output = Dense(1)(x)

    model = Model(inputs, output)
    model.load_weights("LIME_20240506.best.hdf5")
    return model

model = load_pm25_model()

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict the PM2.5 air quality level.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Image Checks
    quality_issues = []
    if is_blurry(image): quality_issues.append("Blurry image")
    if is_overexposed_or_underexposed(image): quality_issues.append("Over/under exposed")
    if is_mostly_white_or_black(image): quality_issues.append("Mostly white or black")

    if not is_valid_sky_image(image):
        st.error("🚫 This image does not appear to be a valid sky photo. Please upload a clear outdoor sky image.")
        st.stop()

    if quality_issues:
        st.warning("⚠️ Image Quality Issues:")
        for issue in quality_issues:
            st.write(f"- {issue}")
        st.stop()

    # Prediction
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    cloud_percent = get_weather_cloud_info(CITY)
    if cloud_percent is not None and cloud_percent > 75:
        st.info("☁️ Cloudy conditions detected in your region. Prediction might be affected. Returning minimum plausible PM2.5 value.")
        pm25_value = MIN_PM25_VALUE
    else:
        prediction = model.predict(img_array)
        pm25_value = float(prediction[0][0])

    category = categorize_pm25(pm25_value)

    st.subheader("📊 Prediction")
    st.write(f"**Predicted PM2.5 Value:** {pm25_value:.2f} µg/m³")
    st.write(f"**Air Quality Category (India):** {category}")
