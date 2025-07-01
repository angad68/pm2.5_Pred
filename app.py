import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU

# Define the custom model (must match training time)
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
    model.load_weights("LIME_20240506.best.hdf5")

    return model

model = load_pm25_model()

# Streamlit UI setup
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a sky image to predict the PM2.5 air quality level.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# PM2.5 to AQI category bins (adjust if needed)
def categorize_pm25(pm_value):
    if pm_value <= 30:
        return "Good"
    elif pm_value <= 60:
        return "Moderate"
    elif pm_value <= 90:
        return "Unhealthy for Sensitive Groups"
    elif pm_value <= 120:
        return "Unhealthy"
    elif pm_value <= 250:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pm25_value = float(prediction[0][0])
    category = categorize_pm25(pm25_value)

    st.subheader("📊 Prediction")
    st.write(f"**Predicted PM2.5 Value:** {pm25_value:.2f} µg/m³")
    st.write(f"**Air Quality Category:** {category}")
