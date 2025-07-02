import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU, Dropout
import os
import requests

# ------------------ Theme ------------------ #
st.markdown("""
<style>
.stApp { background-color: #111111; color: #E0E0E0; }
h1, h2, h3, h4 { color: #F1F1F1; }
.stMetric { background-color: #222222; border-radius: 0.5rem; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ------------------ Constants ------------------ #
MODEL_PATH = "LIME_20240506.best.hdf5"
MIN_PM25_VALUE = 20.0
MAX_FILE_SIZE = 10 * 1024 * 1024
CITY = os.getenv("CITY", "Chandigarh")
WEATHER_API_KEY="7088853eac6948e286555436250107"
USE_UNCERTAINTY = True
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")


# ------------------ Image Quality Checks ------------------ #
def is_blurry(pil_img, threshold=25.0):
    return cv2.Laplacian(np.array(pil_img.convert("L")), cv2.CV_64F).var() < threshold

def is_overexposed_or_underexposed(pil_img, low=35, high=220):
    mean_val = np.mean(np.array(pil_img.convert("L")))
    return mean_val < low or mean_val > high

def is_mostly_white_or_black(pil_img, white_thresh=235, black_thresh=25, percent=0.75):
    img = np.array(pil_img)
    white = np.sum(np.all(img > white_thresh, axis=2))
    black = np.sum(np.all(img < black_thresh, axis=2))
    return (white + black) / (img.shape[0] * img.shape[1]) > percent
import numpy as np
import cv2
from PIL import Image

def is_sky_image(pil_img, base_thresh=0.38, min_sky_region=0.15):
    """
    Improved sky detection function with better robustness and edge case handling.
    
    Args:
        pil_img: PIL Image object
        base_thresh: Base threshold for sky detection (0-1)
        min_sky_region: Minimum ratio of image that must be sky-like
    
    Returns:
        bool: True if image contains significant sky content
    """
    # Preserve aspect ratio while resizing
    original_size = pil_img.size
    max_dim = 384
    if max(original_size) > max_dim:
        ratio = max_dim / max(original_size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    else:
        img = pil_img
    
    img = img.convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    
    # Convert to multiple color spaces for better analysis
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    
    h_hsv, s, v = cv2.split(hsv)
    l_lab, a_lab, b_lab = cv2.split(lab)
    
    # More refined sky detection with multiple criteria
    
    # 1. Blue sky (clear conditions)
    blue_sky = (
        ((h_hsv >= 90) & (h_hsv <= 140)) &  # Blue hue range
        (s > 30) & (s < 200) &              # Moderate saturation
        (v > 60) & (v < 240)                # Avoid pure white/black
    )
    
    # 2. Light/pale sky (overcast or dawn/dusk)
    light_sky = (
        ((h_hsv >= 75) & (h_hsv <= 150)) &  # Broader blue range
        (s > 10) & (s < 80) &               # Lower saturation
        (v > 100) & (v < 250) &             # Bright but not pure white
        (b_lab < 140)                       # Not too yellow (using LAB)
    )
    
    # 3. Gray/white clouds
    gray_white_clouds = (
        (s < 40) &                          # Low saturation
        (v > 130) & (v < 245) &             # High brightness
        (np.abs(a_lab - 128) < 15) &        # Near neutral in LAB a* channel
        (np.abs(b_lab - 128) < 20)          # Near neutral in LAB b* channel
    )
    
    # 4. Sunset/sunrise sky (warmer tones)
    warm_sky = (
        (((h_hsv >= 0) & (h_hsv <= 30)) | ((h_hsv >= 150) & (h_hsv <= 180))) &  # Orange/pink range
        (s > 20) & (s < 180) &
        (v > 80) & (v < 240)
    )
    
    # Combine all sky masks
    sky_mask = blue_sky | light_sky | gray_white_clouds | warm_sky
    
    # Create more sophisticated weight mask
    # Give more weight to upper portions, but with smoother transition
    y_coords = np.arange(h).reshape(-1, 1)
    weight_mask = np.ones((h, w))
    
    # Exponential decay from top to bottom
    weight_mask = 2.0 * np.exp(-2.5 * y_coords / h) + 0.3
    
    # Apply Gaussian blur to weight mask for smoother transitions
    weight_mask = cv2.GaussianBlur(weight_mask, (5, 5), 1.0)
    
    # Calculate weighted sky ratio
    weighted_sky_pixels = np.sum(sky_mask * weight_mask)
    total_weight = np.sum(weight_mask)
    weighted_ratio = weighted_sky_pixels / total_weight
    
    # Additional checks for robustness
    
    # 1. Check if there's a significant sky region in upper third
    upper_third = sky_mask[:h//3, :]
    upper_sky_ratio = np.sum(upper_third) / (h//3 * w)
    
    # 2. Calculate overall brightness and contrast
    avg_brightness = np.mean(v)
    brightness_std = np.std(v)
    
    # 3. Check for texture consistency (sky should be relatively smooth)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_penalty = min(laplacian_var / 1000.0, 0.2)  # Cap penalty
    
    # Adaptive threshold based on image characteristics
    adaptive_thresh = base_thresh
    
    # Adjust for brightness
    if avg_brightness > 200:
        adaptive_thresh += 0.08  # Bright images need higher threshold
    elif avg_brightness < 80:
        adaptive_thresh -= 0.08  # Dark images get lower threshold
    
    # Adjust for contrast
    if brightness_std < 30:
        adaptive_thresh -= 0.05  # Low contrast (overcast) gets easier threshold
    
    # Apply texture penalty
    final_threshold = max(adaptive_thresh - texture_penalty, 0.1)
    
    # Final decision with multiple criteria
    is_sky = (
        (weighted_ratio > final_threshold) and 
        (upper_sky_ratio > min_sky_region) and
        (weighted_ratio > 0.1)  # Minimum sky content
    )
    
    return is_sky

# Optional: Function to visualize the detection for debugging
def visualize_sky_detection(pil_img, save_path=None):
    """
    Visualize the sky detection process for debugging.
    """
    img = pil_img.resize((384, 384), Image.Resampling.LANCZOS).convert("RGB")
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h_hsv, s, v = cv2.split(hsv)
    
    # Recreate the sky detection logic
    blue_sky = ((h_hsv >= 90) & (h_hsv <= 140)) & (s > 30) & (s < 200) & (v > 60) & (v < 240)
    light_sky = ((h_hsv >= 75) & (h_hsv <= 150)) & (s > 10) & (s < 80) & (v > 100) & (v < 250)
    gray_white_clouds = (s < 40) & (v > 130) & (v < 245)
    
    sky_mask = blue_sky | light_sky | gray_white_clouds
    
    # Create visualization
    result = img_np.copy()
    result[sky_mask] = [0, 255, 0]  # Highlight detected sky in green
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    return result

    if weighted_ratio < adaptive_thresh: return False
    if np.std(h[sky_mask]) > 25: return False
    _, labels = cv2.connectedComponents(sky_mask.astype(np.uint8))
    if labels.max() <= 1: return False
    if max(np.bincount(labels.flatten())[1:]) < 0.02 * sky_mask.size: return False
    return True

def is_cloudy_image(pil_img, cloudy_thresh=0.25):
    img = pil_img.resize((256, 256)).convert("RGB")
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    cloud_mask = (s < 40) & (v > 80) & (v < 200)
    dark_cloud_mask = (s < 30) & (v >= 40) & (v <= 90)
    cloudy_combined = cloud_mask | dark_cloud_mask
    return np.mean(cloudy_combined) > cloudy_thresh

# ------------------ Model Loader ------------------ #
@st.cache_resource
def load_pm25_model():
    inputs = Input(shape=(224, 224, 3))
    x = inputs
    for f in [64, 64]:
        x = Conv2D(f, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    for f in [128, 128]:
        x = Conv2D(f, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    for _ in range(3):
        skip = x
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Add()([x, skip])
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    for f in [256, 256]:
        x = Conv2D(f, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Flatten()(x)
    for _ in range(2):
        x = Dense(1024)(x)
        x = Dropout(0.3)(x)
        x = LeakyReLU(alpha=0.1)(x)

    output = Dense(1, activation='linear')(x)
    model = Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mae')
    model.load_weights(MODEL_PATH)
    return model

model = load_pm25_model()

# ------------------ Weather API ------------------ #
def fetch_weather_data(city):
    try:
        url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"🌐 WeatherAPI error: {str(e)}")
        return None

# ------------------ Prediction ------------------ #
def predict_pm25(image):
    resized = image.resize((224, 224))
    img_array = np.array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    if USE_UNCERTAINTY:
        preds = [model(img_array, training=True).numpy().squeeze() for _ in range(30)]
        return float(np.mean(preds)), float(np.std(preds))
    else:
        val = float(model(img_array, training=False).numpy().squeeze())
        return val, 0.0

# ------------------ Categorization ------------------ #
def categorize_pm25(val):
    if val <= 30: return "Good"
    elif val <= 60: return "Satisfactory"
    elif val <= 90: return "Moderately Polluted"
    elif val <= 120: return "Poor"
    elif val <= 250: return "Very Poor"
    return "Severe"

colors = {
    "Good": "🟢", "Satisfactory": "🟡", "Moderately Polluted": "🟠",
    "Poor": "🔴", "Very Poor": "🟣", "Severe": "🔴"
}

# ------------------ App ------------------ #
st.title("🌫️ PM2.5 Air Quality Estimator")

input_mode = st.radio("Select input method:", ["📷 Use Webcam", "📁 Upload Image"])
image = None

if input_mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a sky image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("❌ File too large. Please upload < 10MB.")
            st.stop()
        try:
            image = Image.open(uploaded_file).convert("RGB")
            if min(image.size) < 100:
                st.error("❌ Image too small. Minimum 100x100 pixels required.")
                st.stop()
        except Exception as e:
            st.error(f"Invalid image: {str(e)}")
            st.stop()

elif input_mode == "📷 Use Webcam":
    cam_input = st.camera_input("Capture sky image")
    if cam_input:
        try:
            image = Image.open(cam_input).convert("RGB")
        except Exception as e:
            st.error(f"Camera capture failed: {str(e)}")
            st.stop()

# ------------------ If image available ------------------ #
if image:
    if is_blurry(image):
        st.error("Image is too blurry.")
        st.stop()
    if is_overexposed_or_underexposed(image):
        st.error("Image is over/under exposed.")
        st.stop()
    if is_mostly_white_or_black(image):
        st.error("Image is mostly white or black.")
        st.stop()
    if not is_sky_image(image):
        st.error("Image does not look like sky.")
        st.stop()

    pm25_val, pm25_std = predict_pm25(image)

    if is_cloudy_image(image):
        weather_data = fetch_weather_data(CITY)
        if weather_data:
            condition = weather_data.get("current", {}).get("condition", {}).get("text", "").lower()
            if "cloud" in condition or "overcast" in condition:
                st.info(f"☁️ WeatherAPI: Cloudy in {CITY.title()} — adjusted value.")
                pm25_val = max(pm25_val, MIN_PM25_VALUE)

    # Display
    display_img = image.copy()
    display_img.thumbnail((500, 500))

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(display_img, caption="Captured Image")
    with col2:
        st.subheader("📊 Prediction Results")
        st.metric("PM2.5 Level", f"{pm25_val:.1f} µg/m³")
        st.metric("Uncertainty (±)", f"{pm25_std:.1f}")
        category = categorize_pm25(pm25_val)
        st.markdown(f"**Air Quality:** {colors[category]} {category}")


