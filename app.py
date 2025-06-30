import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="AirVision - PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Air Quality Prediction")

# -----------------------------
# Image Validation Functions
# -----------------------------
def is_blurry(pil_img, threshold=100):
    img = np.array(pil_img.convert("L"))
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return lap_var < threshold

def is_dark(pil_img, threshold=20):
    img = np.array(pil_img.convert("L"))
    brightness = np.mean(img)
    return brightness < threshold

def is_overexposed(pil_img, threshold=240, white_ratio=0.5):
    img = np.array(pil_img.convert("L"))
    white_pixels = np.sum(img > threshold)
    return (white_pixels / img.size) > white_ratio

def is_sky_like(pil_img, blue_ratio_thresh=0.08):
    img = np.array(pil_img.resize((224, 224)))
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    sky_mask = (b > 90) & (b > r + 15) & (b > g + 15)
    sky_pixels = np.sum(sky_mask)
    return (sky_pixels / img.size) > blue_ratio_thresh

def is_low_resolution(pil_img, min_size=(100, 100)):
    return pil_img.size[0] < min_size[0] or pil_img.size[1] < min_size[1]

# -----------------------------
# PM2.5 Advisory Generator
# -----------------------------
def pm25_advisory(pm25):
    if pm25 <= 12:
        return "Good air quality. Breathe freely."
    elif pm25 <= 35.4:
        return "Moderate air quality. Some pollution is present."
    elif pm25 <= 55.4:
        return "Unhealthy for sensitive groups. Consider reducing outdoor activity."
    elif pm25 <= 150.4:
        return "Unhealthy. Limit prolonged outdoor exertion."
    elif pm25 <= 250.4:
        return "Very unhealthy. Avoid outdoor activity. Use masks and purifiers."
    else:
        return "Hazardous. Stay indoors. Critical air pollution level."

# -----------------------------
# Model Loader
# -----------------------------
@st.cache_resource
def load_model(weight_path="pm25_model.hdf5"):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mae')
    model.load_weights(weight_path)
    return model

model = load_model()

# -----------------------------
# File Upload + Prediction
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload a **sky image** (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Image Checks
    if is_low_resolution(image_pil):
        st.error("❌ The image resolution is too low. Please upload a higher-quality image.")
    elif is_dark(image_pil):
        st.error("❌ The image appears too dark. Please upload a daytime sky image.")
    elif is_overexposed(image_pil):
        st.error("❌ The image is overexposed. Avoid glare or direct sunlight.")
    elif is_blurry(image_pil):
        st.warning("⚠️ The image seems blurry. Retake a clearer photo of the sky.")
    elif not is_sky_like(image_pil):
        st.warning("⚠️ This doesn't look like a sky image. Please upload a clearer sky photo.")
    else:
        # Predict
        img = image_pil.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        pm25_value = float(prediction[0][0])
        
        st.markdown(f"### 🌬️ Predicted PM2.5: `{pm25_value:.2f} µg/m³`")
        st.markdown(f"### 🛡️ Advisory: {pm25_advisory(pm25_value)}")
