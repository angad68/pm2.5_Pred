import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import requests
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU

# ------------------ CONFIG ------------------ #
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "7088853eac6948e286555436250107")
CITY = os.getenv("CITY", "Chandigarh")
MIN_PM25_VALUE = 20.0
MODEL_PATH = os.getenv("MODEL_PATH", "LIME_20240506.best.hdf5")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ------------------ Weather API ------------------ #
# Enhanced weather API function
def get_comprehensive_weather_info(city=CITY):
    """Get comprehensive weather data including visibility and conditions"""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data["current"]
        
        return {
            'cloud_cover': current.get("cloud", 0),
            'visibility_km': current.get("vis_km", 10),
            'condition': current.get("condition", {}).get("text", ""),
            'humidity': current.get("humidity", 50),
            'wind_kph': current.get("wind_kph", 0),
            'temp_c': current.get("temp_c", 20)
        }
    except requests.RequestException as e:
        st.warning(f"Weather API unavailable: {str(e)}")
        return None
    except KeyError as e:
        st.warning(f"Unexpected weather API response format: {str(e)}")
        return None

def adjust_prediction_with_weather(base_prediction, weather_data, is_cloudy_image):
    """Adjust PM2.5 prediction based on weather conditions"""
    if not weather_data:
        return base_prediction
    
    adjusted_prediction = base_prediction
    adjustments = []
    
    # Low visibility suggests high pollution
    if weather_data['visibility_km'] < 5:
        adjusted_prediction *= 1.3
        adjustments.append(f"Low visibility ({weather_data['visibility_km']}km)")
    elif weather_data['visibility_km'] < 2:
        adjusted_prediction *= 1.6
        adjustments.append(f"Very low visibility ({weather_data['visibility_km']}km)")
    
    # High humidity can correlate with higher PM2.5
    if weather_data['humidity'] > 80:
        adjusted_prediction *= 1.1
        adjustments.append(f"High humidity ({weather_data['humidity']}%)")
    
    # Cloudy conditions - use weather data to refine
    if is_cloudy_image and weather_data['cloud_cover'] > 80:
        # Heavy overcast might indicate pollution
        adjusted_prediction = max(adjusted_prediction, MIN_PM25_VALUE * 1.5)
        adjustments.append(f"Heavy overcast ({weather_data['cloud_cover']}% clouds)")
    
    # Foggy/misty conditions
    fog_conditions = ['fog', 'mist', 'haze', 'smog']
    if any(condition in weather_data['condition'].lower() for condition in fog_conditions):
        adjusted_prediction *= 1.4
        adjustments.append(f"Foggy conditions: {weather_data['condition']}")
    
    return min(adjusted_prediction, 500.0), adjustments  # Cap at 500

# In your main prediction section, replace with:
# Make prediction with weather adjustment
with st.spinner("Predicting PM2.5 level..."):
    weather_data = get_comprehensive_weather_info(CITY)
    
    if use_weather_override:
        st.info("☁️ Heavy cloud cover detected. Using weather-adjusted baseline.")
        pm25_value = MIN_PM25_VALUE * 1.2  # Slightly higher baseline
        weather_adjustments = ["Heavy cloud cover override"]
    else:
        prediction = model.predict(img_array)
        base_pm25 = validate_pm25_prediction(prediction[0][0])
        
        # Apply weather adjustments
        pm25_value, weather_adjustments = adjust_prediction_with_weather(
            base_pm25, weather_data, cloudy_looking
        )

# Show weather information in results
if weather_data:
    with st.expander("🌤️ Weather Context", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Visibility", f"{weather_data['visibility_km']} km")
            st.metric("Humidity", f"{weather_data['humidity']}%")
        with col2:
            st.metric("Cloud Cover", f"{weather_data['cloud_cover']}%")
            st.metric("Wind", f"{weather_data['wind_kph']} kph")
        with col3:
            st.metric("Temperature", f"{weather_data['temp_c']}°C")
            st.write(f"**Condition:** {weather_data['condition']}")
        
        if weather_adjustments:
            st.write("**Weather Adjustments Applied:**")
            for adj in weather_adjustments:
                st.write(f"• {adj}")

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

import cv2
import numpy as np
from PIL import Image

def is_sky_image(pil_img, sky_percent=0.4):
    """
    Improved sky detection with multiple criteria and stricter thresholds
    """
    img = pil_img.resize((256, 256)).convert("RGB")
    img_np = np.array(img)
    
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    h, s, v = cv2.split(hsv)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 1. Classic blue sky detection (more restrictive)
    blue_sky = ((h >= 95) & (h <= 135)) & (s > 40) & (v > 100)
    
    # 2. Light blue/cyan sky
    light_blue = ((h >= 85) & (h <= 105)) & (s > 25) & (v > 120)
    
    # 3. Gray/overcast sky (more restrictive)
    gray_sky = (s < 20) & (v > 140) & (v < 200)
    
    # 4. Very bright areas (clouds) - but not too bright (avoid white walls)
    bright_clouds = (v > 180) & (s < 40) & (v < 245)
    
    # 5. LAB color space check for sky-like colors
    # Sky typically has negative 'a' (green-magenta axis) and negative 'b' (blue-yellow axis)
    lab_sky = (a_channel < 128) & (b_channel < 125) & (l_channel > 120)
    
    # Combine all sky detection criteria
    sky_mask = blue_sky | light_blue | gray_sky | bright_clouds | lab_sky
    
    # Apply spatial weighting - sky should be in upper portion
    height, width = sky_mask.shape
    weight_mask = np.zeros_like(sky_mask, dtype=float)
    
    # Create gradient weight: top gets more weight, bottom gets less
    for i in range(height):
        # Top 30% gets full weight, middle 40% gets medium weight, bottom 30% gets minimal weight
        if i < height * 0.3:  # Top 30%
            weight_mask[i, :] = 2.0
        elif i < height * 0.7:  # Middle 40%
            weight_mask[i, :] = 1.0
        else:  # Bottom 30%
            weight_mask[i, :] = 0.3
    
    # Calculate weighted sky ratio
    weighted_sky_pixels = np.sum(sky_mask * weight_mask)
    total_weighted_pixels = np.sum(weight_mask)
    weighted_sky_ratio = weighted_sky_pixels / total_weighted_pixels
    
    return weighted_sky_ratio > sky_percent

def detect_indoor_or_objects(pil_img):
    """
    Detect if image likely contains indoor scenes or large objects
    Returns True if image appears to be indoor/contains objects
    """
    img = pil_img.resize((256, 256)).convert("RGB")
    img_np = np.array(img)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Edge detection to find structured objects
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # High edge density suggests man-made objects/structures
    if edge_density > 0.15:
        return True
    
    # Check for dominant colors that suggest indoor scenes
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Brown/wooden colors (furniture, floors)
    brown_mask = ((h >= 8) & (h <= 25)) & (s > 50) & (v > 50)
    brown_ratio = np.sum(brown_mask) / brown_mask.size
    
    # Green vegetation (trees, grass) - might indicate ground-level photo
    green_mask = ((h >= 35) & (h <= 85)) & (s > 40) & (v > 40)
    green_ratio = np.sum(green_mask) / green_mask.size
    
    # Check if bottom half has too much green (ground vegetation)
    bottom_half = green_mask[int(h.shape[0] * 0.5):, :]
    bottom_green_ratio = np.sum(bottom_half) / bottom_half.size
    
    if brown_ratio > 0.3 or bottom_green_ratio > 0.4:
        return True
    
    return False

def has_horizon_line(pil_img):
    """
    Detect if image has a clear horizon line (suggests ground-level photo)
    """
    img = pil_img.resize((256, 256)).convert("L")
    img_np = np.array(img)
    
    # Apply Hough line detection
    edges = cv2.Canny(img_np, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
    
    if lines is not None:
        horizontal_lines = 0
        for rho, theta in lines[:, 0]:
            # Check for near-horizontal lines (horizon)
            angle = theta * 180 / np.pi
            if abs(angle) < 20 or abs(angle - 180) < 20:
                horizontal_lines += 1
        
        # If we find multiple horizontal lines, likely has horizon
        if horizontal_lines > 2:
            return True
    
    return False

def comprehensive_image_check(pil_img):
    """Return a dict with all check results using improved sky detection"""
    
    # Use the improved sky detection
    sky_results = comprehensive_sky_check(pil_img)
    
    checks = {
        'is_blurry': is_blurry(pil_img),
        'is_poorly_exposed': is_overexposed_or_underexposed(pil_img),
        'has_sufficient_sky': sky_results['has_sufficient_sky'],
        'is_mostly_uniform': is_mostly_white_or_black(pil_img),
        'is_cloudy': is_cloudy_image(pil_img),
        # Additional detailed info for debugging
        'sky_details': sky_results
    }
    return checks

# Updated error message section in your main code:
def show_detailed_sky_feedback(sky_details):
    """Show detailed feedback about why sky detection failed"""
    if not sky_details['sky_detected']:
        st.write("   - No sky colors detected (blue, gray, or white clouds)")
    if sky_details['objects_detected']:
        st.write("   - Indoor objects or structures detected")
    if sky_details['horizon_detected']:
        st.write("   - Horizon line detected (suggests ground-level photo)")

# In your main UI code, replace the failed checks section with:
if failed_checks:
    st.error("⚠️ Image Quality Issues:")
    for i, issue in enumerate(failed_checks, 1):
        st.write(f"{i}. {issue}")
    
    # Show detailed sky feedback if sky detection failed
    if not quality_checks['has_sufficient_sky']:
        st.write("**Sky Detection Details:**")
        show_detailed_sky_feedback(quality_checks['sky_details'])
    
    st.info("💡 **Tips:** Take photos outdoors pointing upward at the sky, ensure good lighting and focus.")
    st.stop()


def comprehensive_sky_check(pil_img):
    """
    Comprehensive sky image validation
    """
    # Basic sky detection
    has_sky = is_sky_image_improved(pil_img)
    
    # Additional checks
    has_objects = detect_indoor_or_objects(pil_img)
    has_horizon = has_horizon_line(pil_img)
    
    # Combine criteria
    is_valid_sky = has_sky and not has_objects and not has_horizon
    
    return {
        'has_sufficient_sky': is_valid_sky,
        'sky_detected': has_sky,
        'objects_detected': has_objects,
        'horizon_detected': has_horizon
    }

# Test function to help debug
def debug_sky_detection(pil_img):
    """
    Debug function to understand why an image passes/fails sky detection
    """
    img = pil_img.resize((256, 256)).convert("RGB")
    img_np = np.array(img)
    
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Calculate each criteria
    blue_sky = ((h >= 95) & (h <= 135)) & (s > 40) & (v > 100)
    light_blue = ((h >= 85) & (h <= 105)) & (s > 25) & (v > 120)
    gray_sky = (s < 20) & (v > 140) & (v < 200)
    bright_clouds = (v > 180) & (s < 40) & (v < 245)
    
    print(f"Blue sky pixels: {np.sum(blue_sky) / blue_sky.size:.3f}")
    print(f"Light blue pixels: {np.sum(light_blue) / light_blue.size:.3f}")
    print(f"Gray sky pixels: {np.sum(gray_sky) / gray_sky.size:.3f}")
    print(f"Bright cloud pixels: {np.sum(bright_clouds) / bright_clouds.size:.3f}")
    
    # Overall sky detection
    sky_mask = blue_sky | light_blue | gray_sky | bright_clouds
    print(f"Total sky pixels: {np.sum(sky_mask) / sky_mask.size:.3f}")
    
    # Check color distribution
    print(f"Average hue: {np.mean(h):.1f}")
    print(f"Average saturation: {np.mean(s):.1f}")
    print(f"Average brightness: {np.mean(v):.1f}")
    
    return {
        'blue_sky_ratio': np.sum(blue_sky) / blue_sky.size,
        'light_blue_ratio': np.sum(light_blue) / light_blue.size,
        'gray_sky_ratio': np.sum(gray_sky) / gray_sky.size,
        'bright_cloud_ratio': np.sum(bright_clouds) / bright_clouds.size,
        'total_sky_ratio': np.sum(sky_mask) / sky_mask.size
    }

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

def validate_pm25_prediction(prediction_value):
    """Ensure PM2.5 prediction is within reasonable bounds"""
    return max(0.0, min(1000.0, float(prediction_value)))

# ------------------ Load Model ------------------ #
@st.cache_resource
def load_pm25_model():
    try:
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
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.error("Please ensure the model file 'LIME_20240506.best.hdf5' is in the correct directory.")
        st.stop()

model = load_pm25_model()

# ------------------ UI ------------------ #
st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ PM2.5 Level Predictor")
st.write("Upload a **sky image** to predict PM2.5 air quality level. No indoor or obstructed photos.")

uploaded_file = st.file_uploader("Choose a sky image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Validate file size
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File too large. Please upload an image smaller than 10MB.")
        st.stop()
    
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Validate image dimensions
        if image.size[0] < 100 or image.size[1] < 100:
            st.error("Image too small. Please upload an image at least 100x100 pixels.")
            st.stop()
            
    except Exception as e:
        st.error(f"Invalid image file: {str(e)}")
        st.stop()
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform comprehensive quality checks
    with st.spinner("Analyzing image quality..."):
        quality_checks = comprehensive_image_check(image)
    
    failed_checks = []
    if quality_checks['is_blurry']: 
        failed_checks.append("Image is blurry - try taking a clearer photo.")
    if quality_checks['is_poorly_exposed']: 
        failed_checks.append("Image is too dark or bright - adjust exposure.")
    if not quality_checks['has_sufficient_sky']: 
        failed_checks.append("Not enough sky visible - point camera upward.")
    
    cloudy_looking = quality_checks['is_cloudy']
    
    if failed_checks:
        st.error("⚠️ Image Quality Issues:")
        for i, issue in enumerate(failed_checks, 1):
            st.write(f"{i}. {issue}")
        st.info("💡 **Tips:** Take photos outdoors pointing upward at the sky, ensure good lighting and focus.")
        st.stop()
    
    # Prepare image for prediction
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Cloud handling with weather API
    use_weather_override = False
    if cloudy_looking:
        cloud_percent = get_weather_cloud_info(CITY)
        if cloud_percent and cloud_percent > 75:
            use_weather_override = True
    
    # Make prediction
    with st.spinner("Predicting PM2.5 level..."):
        if use_weather_override:
            st.info("☁️ Heavy cloud cover detected. Using minimum baseline value.")
            pm25_value = MIN_PM25_VALUE
        else:
            prediction = model.predict(img_array)
            pm25_value = validate_pm25_prediction(prediction[0][0])
    
    category = categorize_pm25(pm25_value)
    
    # Display results with improved UI
    st.subheader("📊 Prediction Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PM2.5 Level", f"{pm25_value:.1f} µg/m³")
    with col2:
        # Color-coded category display
        colors = {
            "Good": "🟢", 
            "Satisfactory": "🟡", 
            "Moderately Polluted": "🟠", 
            "Poor": "🔴", 
            "Very Poor": "🟣", 
            "Severe": "🔴"
        }
        st.metric("Air Quality", f"{colors.get(category, '⚪')} {category}")
    
    # Additional information
    if cloudy_looking and not use_weather_override:
        st.info("ℹ️ Cloudy conditions detected in image. Prediction accuracy may be affected.")
    
    # Health recommendations based on category
    health_advice = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities!",
        "Satisfactory": "Air quality is acceptable for most people.",
        "Moderately Polluted": "Sensitive individuals should limit prolonged outdoor activities.",
        "Poor": "Everyone should reduce prolonged outdoor activities.",
        "Very Poor": "Avoid outdoor activities. Keep windows closed.",
        "Severe": "Stay indoors. Avoid all outdoor activities."
    }
    
    if category in health_advice:
        st.info(f"💡 **Health Advice:** {health_advice[category]}")
