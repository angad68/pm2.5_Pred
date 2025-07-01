import streamlit as st
import numpy as np
from PIL import Image
import cv2
import requests
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU

# ------------------ CONFIG ------------------ #
WEATHER_API_KEY = "7088853eac6948e286555436250107"  # Replace with your key
CITY = "Chandigarh"

# ------------------ Weather API ------------------ #
def get_weather_data(city=CITY):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        response = requests.get(url)
        data = response.json()
        return {
            "cloud": data["current"]["cloud"],
            "humidity": data["current"]["humidity"],
            "wind_kph": data["current"]["wind_kph"],
            "desc": data["current"]["condition"]["text"]
        }
    except:
        return None

# ------------------ Image Quality Checks ------------------ #
def is_blurry(pil_img, threshold=30.0):
    img_gray = np.array(pil_img.convert("L"))
    return cv2.Laplacian(img_gray, cv2.CV_64F).var() < threshold

def is_exposed(pil_img, low_thresh=30, high_thresh=230):
    img_gray = np.array(pil_img.convert("L"))
    mean = np.mean(img_gray)
    return mean < low_thresh or mean > high_thresh

def is_obstructed(pil_img, edge_thresh=0.2):
    img = np.array(pil_img.resize((128, 128)))
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density > edge_thresh

# ------------------ PM2.5 Categorization ------------------ #
def categorize_pm25(pm):
    if pm <= 30: return "Good"
    elif pm <= 60: return "Satisfactory"
    elif pm <= 90: return "Moderately Polluted"
    elif pm <= 120: return "Poor"
    elif pm <= 250: return "Very Poor"
    else: return "Severe"

# ------------------ Model Loader ------------------ #
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
    leak13 = LeakyReLU(alpha=0.1)(dense1)
    dense2 = Dense(1024)(leak13)
    leak14 = LeakyReLU(alpha=0.1)(dense2)
    output = Dense(1)(leak14)

    model = Model(inputs=inputs, outputs=output)
    model.load_weights("LIME_20240506.best.hdf5")
    return model

# ------------------ Streamlit App ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict the PM2.5 air quality level. Avoid indoor or heavily obstructed images.")

uploaded_file = st.file_uploader("Upload a sky image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Quality Warnings
    warnings = []
    if is_blurry(image): warnings.append("• Image appears slightly blurry.")
    if is_exposed(image): warnings.append("• Image is underexposed or overexposed.")
    if is_obstructed(image): warnings.append("• Sky may be obstructed (trees/buildings).")

    if warnings:
        st.warning("⚠️ Image Quality Notice:\n" + "\n".join(warnings))

    # Run prediction
    img = image.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    model = load_pm25_model()
    prediction = model.predict(arr)
    pm25 = float(prediction[0][0])
    category = categorize_pm25(pm25)

    st.subheader("📊 Prediction")
    st.write(f"**Predicted PM2.5 Value:** {pm25:.2f} µg/m³")
    st.write(f"**Air Quality Category:** {category}")

    # Weather info
    weather = get_weather_data()
    if weather:
        st.subheader(f"🌦️ Weather in {CITY}")
        st.write(f"**Condition:** {weather['desc']}")
        st.write(f"**Cloud Cover:** {weather['cloud']}%")
        st.write(f"**Humidity:** {weather['humidity']}%")
        st.write(f"**Wind Speed:** {weather['wind_kph']} km/h")
        if weather['cloud'] > 80 and pm25 > 90:
            st.info("☁️ Heavy cloud cover may influence predictions due to reduced sky clarity.")
    else:
        st.info("Live weather data unavailable.")

