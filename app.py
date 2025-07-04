import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, LeakyReLU, Dropout
import os
from ultralytics import YOLO



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
USE_UNCERTAINTY = True

# ------------------ Image Quality Checks ------------------ #
def blur_score(pil_img, resize_to=(224, 224)):
    gray_img = pil_img.convert("L")
    if resize_to:
        gray_img = gray_img.resize(resize_to)
    img_np = np.array(gray_img)
    lap_var = cv2.Laplacian(img_np, cv2.CV_64F).var()
    return lap_var

def assess_blur(pil_img, is_sky_detected=False):
    score = blur_score(pil_img)
    threshold = 2.0 if is_sky_detected else 100.0

    if score < threshold:
        msg = f"⚠️ Image is too blurry (sharpness score: {score:.2f})."
        if is_sky_detected:
            st.error(msg + " Even plain skies should have more detail.")
        else:
            st.error(msg + " Please upload a clearer image.")
        st.stop()
    else:
        st.success(f"✅ Image clarity looks good (sharpness score: {score:.2f})")

    return score

def is_overexposed_or_underexposed(pil_img, low=35, high=220):
    mean_val = np.mean(np.array(pil_img.convert("L")))
    return mean_val < low or mean_val > high

def is_mostly_white_or_black(pil_img, white_thresh=235, black_thresh=25, percent=0.75):
    img = np.array(pil_img)
    white = np.sum(np.all(img > white_thresh, axis=2))
    black = np.sum(np.all(img < black_thresh, axis=2))
    return (white + black) / (img.shape[0] * img.shape[1]) > percent

def is_sky_image(pil_img, base_thresh=0.40, min_sky_region=0.20):
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
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    h_hsv, s, v = cv2.split(hsv)
    l_lab, a_lab, b_lab = cv2.split(lab)

    blue_sky = ((h_hsv >= 60) & (h_hsv <= 130)) & (s > 30) & (s < 200) & (v > 60) & (v < 240)
    light_sky = ((h_hsv >= 75) & (h_hsv <= 150)) & (s > 10) & (s < 80) & (v > 100) & (v < 250) & (b_lab < 140)
    gray_white_clouds = (s < 40) & (v > 130) & (v < 245) & (np.abs(a_lab - 128) < 15) & (np.abs(b_lab - 128) < 20)
    warm_sky = (((h_hsv >= 0) & (h_hsv <= 30)) | ((h_hsv >= 130) & (h_hsv <= 180))) & (s > 20) & (s < 180) & (v > 80) & (v < 240)
    cloudy_sky = (s < 50) & (v > 100) & (v < 200)

    dense_clouds = (s < 30) & (v > 150) & (v < 220)
    scattered_clouds = (s < 60) & (v > 120) & (v < 200)
    stormy_clouds = (s > 30) & (v < 100)

    sky_mask = blue_sky | light_sky | gray_white_clouds | warm_sky | cloudy_sky

    y_coords = np.arange(h).reshape(-1, 1)
    weight_mask = 2.0 * np.exp(-2.5 * y_coords / h) + 0.3
    weight_mask = cv2.GaussianBlur(weight_mask, (5, 5), 1.0)
    weighted_sky_pixels = np.sum(sky_mask * weight_mask)
    total_weight = np.sum(weight_mask)
    weighted_ratio = weighted_sky_pixels / total_weight

    upper_third = sky_mask[:h // 3, :]
    upper_sky_ratio = np.sum(upper_third) / (h // 3 * w)
    avg_brightness = np.mean(v)
    brightness_std = np.std(v)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_penalty = min(laplacian_var / 1000.0, 0.2)

    # ✅ Initialize adaptive_thresh BEFORE use
    adaptive_thresh = base_thresh
    if avg_brightness > 200:
        adaptive_thresh += 0.08
    elif avg_brightness < 80:
        adaptive_thresh -= 0.08
    if brightness_std < 30:
        adaptive_thresh -= 0.05

    cloud_ratio = np.sum(dense_clouds | scattered_clouds | stormy_clouds) / (h * w)
    if cloud_ratio > 0.3:
        final_threshold = max(adaptive_thresh - 0.1, 0.05)
    else:
        final_threshold = max(adaptive_thresh - texture_penalty, 0.1)

    return (
        (weighted_ratio > final_threshold) and
        (upper_sky_ratio > min_sky_region) and
        (weighted_ratio > 0.1)
    )


def visualize_sky_mask(pil_img):
    img = pil_img.resize((384, 384), Image.Resampling.LANCZOS).convert("RGB")
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    h, s, v = cv2.split(hsv)
    _, a, b = cv2.split(lab)

    blue_sky = ((h >= 60) & (h <= 130)) & (s > 30) & (s < 200) & (v > 60) & (v < 240)
    light_sky = ((h >= 75) & (h <= 150)) & (s > 10) & (s < 80) & (v > 100) & (v < 250) & (b < 140)
    gray_white_clouds = (s < 40) & (v > 130) & (v < 245) & (np.abs(a - 128) < 15) & (np.abs(b - 128) < 20)
    warm_sky = (((h >= 0) & (h <= 30)) | ((h >= 130) & (h <= 180))) & (s > 20) & (s < 180) & (v > 80) & (v < 240)

    sky_mask = blue_sky | light_sky | gray_white_clouds | warm_sky

    vis = img_np.copy()
    vis[sky_mask] = [0, 255, 0]
    return vis





def detect_cloud_types(pil_img):
    """Classify cloud types that affect PM2.5 visibility"""
    img_np = np.array(pil_img.resize((256, 256)))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # Cloud type masks
    cirrus = (hsv[...,1] < 30) & (hsv[...,2] > 200)
    cumulus = (hsv[...,1] > 40) & (hsv[...,2] > 150)
    stratus = (hsv[...,1] < 50) & (hsv[...,2] < 150)
    
    return {
        'cirrus': np.sum(cirrus),
        'cumulus': np.sum(cumulus),
        'stratus': np.sum(stratus)
    }




def is_valid_cloud_formation(pil_img):
    cloud_data = detect_cloud_types(pil_img)
    total_pixels = 256 * 256
    return (
        (cloud_data['cirrus']/total_pixels < 0.7) and  # Thin clouds OK
        (cloud_data['stratus']/total_pixels < 0.5)     # Thick clouds limited
    )

def contains_non_sky_objects(pil_img, allowed_classes=('cloud', 'sky'), conf_thresh=0.4):
    img_np = np.array(pil_img)

    # Mask bottom half — only keep upper half
    h = img_np.shape[0]
    upper_half = img_np[:h // 2, :, :]

    results = yolo_model(upper_half)[0]
    names = results.names
    classes = results.boxes.cls if hasattr(results, "boxes") else []
    confidences = results.boxes.conf if hasattr(results, "boxes") else []

    detected = []
    for cls_id, conf in zip(classes, confidences):
        if conf >= conf_thresh:
            obj_name = names[int(cls_id)]
            detected.append(obj_name)

    foreign = [obj for obj in detected if obj.lower() not in allowed_classes]
    return len(foreign) > 0, detected




    foreign = [cls for cls in detected_classes if cls.lower() not in allowed_classes]
    return len(foreign) > 0, detected_classes

    # Check for any object not in allowed list
    for obj in detected:
        if obj.lower() not in allowed_classes:
            return True, detected
    return False, detected



# ------------------ Model Loader ------------------ #
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



@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")  # or yolov5s.pt if using YOLOv5

yolo_model = load_yolo_model()


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
st.title("🌫️Air Quality Estimator using PM2.5 value")
input_mode = st.radio("Select input method:", ["📁 Upload Image", "📷 Use Webcam"])
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
    is_sky = is_sky_image(image)
    _ = assess_blur(image, is_sky_detected=is_sky)

    if is_overexposed_or_underexposed(image):
        st.error("Image is over/under exposed.")
        st.stop()
    if is_mostly_white_or_black(image):
        st.error("Image is mostly white or black.")
        st.stop()
    is_foreign, objects = contains_non_sky_objects(image)

                                                   
    if is_foreign:
        st.error(f"❌ Image contains non-sky elements: {set(objects)}. Please upload a clear sky image.")
        st.stop()
        

    if not is_sky:
        st.warning("⚠️ Image may not be a clear sky.")





if image:
    if is_mostly_white_or_black(image):
        st.error("Unable to process: Image lacks contrast")
        
    elif not is_valid_cloud_formation(image):
        st.warning("⚠️ Cloud formation may obscure accurate measurement")





    
    
    pm25_val, pm25_std = predict_pm25(image)
    
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
        # In your prediction display:
    if st.checkbox("🌥️ Show Weather Analysis"):
        cloud_data = detect_cloud_types(image)
        st.write("### Cloud Composition:")
        st.write(f"- Cirrus (thin): {cloud_data['cirrus']/(256*256):.1%}")
        st.write(f"- Cumulus (fluffy): {cloud_data['cumulus']/(256*256):.1%}")
        st.write(f"- Stratus (thick): {cloud_data['stratus']/(256*256):.1%}")

        if cloud_data['stratus'] > cloud_data['cumulus']:
            st.warning("Heavy cloud cover detected. Results may be less accurate.")

