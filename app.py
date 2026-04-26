import streamlit as st
import numpy as np
import cv2
import time
import json
from datetime import datetime
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model.h5",
        custom_objects={
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'iou_score': iou_score
        }
    )

st.set_page_config(page_title="Face Segmentation App", layout="wide")
st.title("Face Segmentation App")
st.write("Upload a movie screenshot to detect and segment faces.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Mask threshold", 0.1, 0.9, 0.5, 0.05)
    alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.4, 0.05)

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_np = np.array(image.convert("RGB"))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    # Preprocess
    img = np.array(image.convert("RGB"), dtype=np.float32)
    img = cv2.resize(img, (224, 224)) / 255.0
    inp = np.expand_dims(img, axis=0)

    # Inference
    start = time.time()
    pred = model.predict(inp, verbose=0)
    ms = (time.time() - start) * 1000

    # Mask
    mask = (pred[0, :, :, 0] > threshold).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]))

    with col2:
        st.subheader("Mask")
        st.image(mask_resized, use_container_width=True, clamp=True)

    with col3:
        st.subheader("Overlay")
        overlay = original_np.copy()
        overlay[:, :, 0] = np.where(mask_resized > 0, 200, overlay[:, :, 0])
        st.image(overlay, use_container_width=True)

    # Metrics
    st.markdown("---")
    st.subheader("Performance Dashboard")
    m1, m2, m3 = st.columns(3)
    m1.metric("Inference Time", f"{ms:.1f} ms",
              delta="Fast" if ms < 100 else "Slow")
    m2.metric("Face Coverage",
              f"{(mask_resized > 0).sum() / mask_resized.size * 100:.1f}%")
    m3.metric("Image Size",
              f"{original_np.shape[1]}x{original_np.shape[0]}")

    # Download log
    log = {
        "timestamp": datetime.now().isoformat(),
        "filename": uploaded_file.name,
        "inference_ms": round(ms, 2),
        "threshold": threshold
    }
    st.download_button("Download Log", 
                       json.dumps(log, indent=2),
                       "log.json", "application/json")