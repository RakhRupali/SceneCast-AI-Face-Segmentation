import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
import json
from PIL import Image
import time

# --- KERAS VERSION COMPATIBILITY LAYER ---
class CompatInputLayer(InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop('batch_shape', None)
        super().__init__(**kwargs)

# --- STANDARD SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="RecycleVision Classifier",
    page_icon="♻️",
    layout="wide"
)

# ---  BACKGROUND  ---

st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.92)), 
                          url('https://images.unsplash.com/photo-1611284446314-60a58ac0deb9?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INTERFACE HEADER ---
st.title("♻️ RecycleVision Module")
st.text("Automated Waste Material Inspection & Deep Learning Pipeline")
st.markdown("---")


# --- MODEL LOADING UTILITY WITH CACHING ---
@st.cache_resource
def load_nn_pipeline():
    trained_model = load_model(
        "models/best_model.h5", 
        compile=False, 
        custom_objects={'InputLayer': CompatInputLayer}
    )
    with open("class_labels.json", "r") as f:
        labels_map = json.load(f)
    return trained_model, labels_map

try:
    model, class_labels = load_nn_pipeline()
except Exception as err:
    st.error(f"IO Error: System failed to initialize model components. Info: {err}")

# Static dictionary for standard disposal suggestions
recycling_tips = {
    "cardboard": "Flatten the boxes. Keep away from moisture and oil stains before disposal.",
    "glass": "Wash the glass container. Always remove metallic or plastic caps.",
    "metal": "Rinse empty cans properly. Crush them flat to reduce space in bin.",
    "paper": "Keep papers dry and untangled. Avoid mixing heavily soiled or glossy sheets.",
    "plastic": "Check the resin code number. Rinse all fluid residues completely.",
    "trash": "Non-recyclable material. Put into standard landfill dumpsters safely."
}

# --- SCREEN WORKSPACE SPLIT ---
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📁 Image Acquisition")
    uploaded_file = st.file_uploader(
        "Upload local waste item sample image:", 
        type=["jpg", "jpeg", "png"],
        help="Accepts common standard RGB image arrays"
    )
    
    if uploaded_file:
        raw_image = Image.open(uploaded_file)
        # Fixed parameter for older Streamlit environments
        st.image(raw_image, caption="Current Input Buffer Stream", use_column_width=True)

with col2:
    st.subheader("📉 Inference & Analytics Metrics")
    
    if uploaded_file:
        with st.status("Processing input matrix...", expanded=True) as state_monitor:
            time.sleep(0.4)
            
            resized_img = raw_image.resize((224, 224))
            img_tensor = np.array(resized_img) / 255.0
            batched_tensor = np.expand_dims(img_tensor, axis=0)
            
            state_monitor.update(label="Computing soft probabilities over network...", state="running")
            network_output = model.predict(batched_tensor)[0]
            
            time.sleep(0.2)
            state_monitor.update(label="Inference run completed.", state="complete", expanded=False)

        ranked_indices = network_output.argsort()[::-1]
        top_index = ranked_indices[0]
        final_label = class_labels[str(top_index)]
        confidence_percent = network_output[top_index] * 100

        st.metric(
            label="Identified Material Class", 
            value=final_label.upper(), 
            delta=f"{confidence_percent:.2f}% System Match Score"
        )
        
        st.markdown("---")
        st.write("**Top Class Probability Breakdown Matrix:**")
        
        for rank in range(3):
            curr_idx = ranked_indices[rank]
            item_name = class_labels[str(curr_idx)]
            probability_val = network_output[curr_idx]
            
            st.text(f"{item_name.capitalize()} (Probability: {probability_val:.4f})")
            st.progress(float(probability_val))

        st.markdown("---")
        matched_tip = recycling_tips.get(final_label, "Sort the item into the assigned waste management container safely.")
        st.info(f"**Disposal Guideline Notice:** {matched_tip}")
        
    else:
        st.info("System idle. Awaiting file input from container pipeline to trigger classification.")

st.markdown("---")
st.caption("RecycleVision Core Subsystem Engine | Local Deployment Environment Diagnostics")
