import streamlit as st
import numpy as np
from PIL import Image
import cv2
import requests
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU

# ------------------ CONFIG ------------------ #
WEATHER_API_KEY = "7088853eac6948e286555436250107"  # Replace with your own
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

def is_sky_like(pil_img):
    img = np.array(pil_img.resize((128, 128)))
    b = img[:, :, 2]
    g = img[:, :, 1]
    r = img[:, :, 0]
    sky_mask = (b > r + 10) & (b > g + 10)
    sky_coverage = np.sum(sky_mask) / sky_mask.size
    return sky_coverage > 0.15

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

# ------------------ CNN Architecture ------------------ #
@st.cache_resource
def load_pm25_model():
    inputs = Input(shape=(224, 224, 3))

    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    leak1 = LeakyReLU(alpha=0.1)(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same')(leak1)
    leak2 = LeakyReLU(alpha=0.1)(conv2)
    pool1 = MaxPooling2D((3, 3), strides=(2, 2))(leak2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool1)
    leak3 = LeakyReLU(alpha=0.1)(conv3)
    conv4 = Conv2D(128, (3, 3), padding='same')(leak3)
    leak4 = LeakyReLU(alpha=0.1)(conv4)
    pool2 = MaxPooling2D((3, 3), strides=(2, 2))(leak4)

    conv5 = Conv2D(128, (3, 3), padding='same')(pool2)
    leak5 = LeakyReLU(alpha=0.1)(conv5)
    res2 = Add()([leak5, pool2])
    pool3 = MaxPooling2D((3, 3), strides=(2, 2))(res2)

    conv7 = Conv2D(128, (3, 3), padding='same')(pool3)
    leak7 = LeakyReLU(alpha=0.1)(conv7)
    res3 = Add()([leak7, pool3])
    pool4 = MaxPooling2D((3, 3), strides=(2, 2))(res3)

    conv9 = Conv2D(128, (3, 3), padding='same')(pool4)
    leak9 = LeakyReLU(alpha=0.1)(conv9)
    res4 = Add()([leak9, pool4])
    pool5 = MaxPooling2D((3, 3), strides=(2, 2))(res4)

    conv11 = Conv2D(256, (3, 3), padding='same')(pool5)
    leak11 = LeakyReLU(alpha=0.1)(conv11)
    conv12 = Conv2D(256, (3, 3), padding='same')(leak11)
    leak12 = LeakyReLU(alpha=0.1)(conv12)
    pool6 = MaxPooling2D((3, 3), strides=(2, 2))(leak12)

    flatten = Flatten()(pool6)
    dense1 = Dense(1024)(flatten)
    fcLeak1 = LeakyReLU(alpha=0.1)(dense1)
    dense2 = Dense(1024)(fcLeak1)
    fcLeak2 = LeakyReLU(alpha=0.1)(dense2)
    pm25 = Dense(1, activation='linear')(fcLeak2)

    model = Model(inputs=inputs, outputs=pm25)
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    model.load_weights("LIME_20240506.best.hdf5")
    return model

# ------------------ Image Preprocessing ------------------ #
def preprocess_uploaded_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("\U0001F32B\uFE0F PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict PM2.5 air quality.")

uploaded_file = st.file_uploader("Choose a sky image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array, display_img = preprocess_uploaded_image(uploaded_file)
    st.image(display_img, caption="Uploaded Image", use_container_width=True)

    issues = []
    should_abort = False
    cloudy_looking = False

    if not is_sky_like(display_img):
        issues.append("Image doesn't look like a sky photo.")
        should_abort = True

    if is_blurry(display_img):
        issues.append("Image might be blurry.")
        should_abort = True

    if is_overexposed_or_underexposed(display_img):
        issues.append("Image is too bright or too dark.")
        should_abort = True

    if is_mostly_white_or_black(display_img):
        issues.append("Sky seems fully covered (cloudy/washed).")
        cloudy_looking = True

    if issues:
        st.warning("\u26A0\uFE0F Image Quality Issues Detected:")
        for issue in issues:
            st.write(f"- {issue}")

    if should_abort:
        st.error("\u26D4\uFE0F This image is unsuitable for prediction. Please upload a clearer sky image.")
    else:
        model = load_pm25_model()
        use_weather_adjustment = False
        if cloudy_looking:
            cloud_percent = get_weather_cloud_info(CITY)
            if cloud_percent and cloud_percent > 75:
                use_weather_adjustment = True

        if use_weather_adjustment:
            st.info("\u2601\uFE0F Cloudy conditions detected in your region. Prediction might be affected. Using minimum plausible value.")
            pm25_value = MIN_PM25_VALUE
        else:
            prediction = model.predict(img_array)
            pm25_value = float(prediction[0][0])

        category = categorize_pm25(pm25_value)

        st.subheader("\U0001F4CA Prediction")
        st.write(f"**Predicted PM2.5:** {pm25_value:.2f} µg/m³")
        st.write(f"**Air Quality Category:** {category}")
