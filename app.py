# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import os
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Skin Cancer Classifier", page_icon="🩺", layout="centered")

st.title("Skin Cancer Image Classifier")
st.write("Upload a skin image (jpg/png). The app will predict whether it's benign or malignant.")

# ---------------------------
# Helpers
# ---------------------------
def load_labels_from_file(path):
    """
    Read label lines and return cleaned list of labels.
    Handles lines like: "0 BENIGN CANCER SKIN" by returning "BENIGN CANCER SKIN".
    """
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                labels.append(parts[1])
            else:
                labels.append(parts[0])
    return labels

def preprocess_pil_image(pil_img, size=(224, 224)):
    """
    Take a PIL.Image (RGB) and return a numpy array shaped (1,224,224,3) normalized to (-1,1).
    """
    # Ensure RGB
    img = pil_img.convert("RGB")
    # Use Image.Resampling if available (Pillow >= 9.x), fallback to Image.LANCZOS
    resample = getattr(Image, "Resampling", None)
    if resample:
        resample_filter = Image.Resampling.LANCZOS
    else:
        resample_filter = Image.LANCZOS
    img = ImageOps.fit(img, size, resample_filter)
    arr = np.asarray(img).astype(np.float32)
    normalized = (arr / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)

# ---------------------------
# Model + labels loading logic
# ---------------------------
st.sidebar.header("Model / Labels")

use_local_files = False
local_model_path = "keras_Model.h5"
local_labels_path = "labels.txt"

# Show whether local model/labels exist
model_exists = os.path.exists(local_model_path)
labels_exist = os.path.exists(local_labels_path)

st.sidebar.write(f"Local model found: {'✅' if model_exists else '❌'} (`{local_model_path}`)")
st.sidebar.write(f"Local labels found: {'✅' if labels_exist else '❌'} (`{local_labels_path}`)")

# Optionally allow user to upload model and labels instead
uploaded_model = st.sidebar.file_uploader("Upload `keras_Model.h5` (optional)", type=["h5", "keras", "hdf5"])
uploaded_labels = st.sidebar.file_uploader("Upload `labels.txt` (optional)", type=["txt"])

# Try to load labels (uploaded takes precedence)
labels = None
if uploaded_labels is not None:
    try:
        labels_bytes = uploaded_labels.read().decode("utf-8").splitlines()
        # clean same way as load_labels_from_file
        labels = []
        for line in labels_bytes:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            labels.append(parts[1] if len(parts) > 1 else parts[0])
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded labels: {e}")

elif labels_exist:
    try:
        labels = load_labels_from_file(local_labels_path)
    except Exception as e:
        st.sidebar.error(f"Failed to read local labels.txt: {e}")

if labels:
    st.sidebar.success(f"Loaded {len(labels)} labels")
else:
    st.sidebar.info("No labels loaded yet. Upload labels.txt or place it next to the app.")

# Load model (uploaded takes precedence)
model = None
model_load_error = None
if uploaded_model is not None:
    # Save uploaded model to a temp file then load
    try:
        tfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tfile.write(uploaded_model.read())
        tfile.flush()
        tfile.close()
        model = load_model(tfile.name, compile=False)
        st.sidebar.success("Loaded uploaded model")
        # remove temp file on exit? leave for now
    except Exception as e:
        model_load_error = f"Failed to load uploaded model: {e}"
elif model_exists:
    try:
        model = load_model(local_model_path, compile=False)
        st.sidebar.success("Loaded local model")
    except Exception as e:
        model_load_error = f"Failed to load local model: {e}"
else:
    st.sidebar.info("No model loaded. Upload a `keras_Model.h5` or place it next to the app.")

if model_load_error:
    st.sidebar.error(model_load_error)

# ---------------------------
# Main UI — image upload and predict
# ---------------------------
st.markdown("---")
uploaded_image = st.file_uploader("Upload skin image", type=["jpg", "jpeg", "png"])

if uploaded_image is None:
    st.info("Upload an image to get a prediction.")
else:
    # Display uploaded image
    try:
        image = Image.open(uploaded_image)
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)
    st.write("")

    # Predict button
    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded. Upload or provide `keras_Model.h5` in the app folder.")
        elif labels is None or len(labels) == 0:
            st.error("Labels not loaded. Upload `labels.txt` or provide it in the app folder.")
        else:
            with st.spinner("Preprocessing image..."):
                try:
                    data = preprocess_pil_image(image, size=(224, 224))
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")
                    st.stop()

            with st.spinner("Running model inference..."):
                try:
                    preds = model.predict(data)
                except Exception as e:
                    st.error(f"Model inference failed: {e}")
                    st.stop()

            # Interpret predictions
            if preds.ndim == 2:
                pred_vector = preds[0]
            elif preds.ndim == 1:
                pred_vector = preds
            else:
                st.error("Unexpected model output shape: " + str(preds.shape))
                st.stop()

            idx = int(np.argmax(pred_vector))
            confidence = float(pred_vector[idx])
            label = labels[idx] if idx < len(labels) else f"Class {idx}"

            # show results
            st.success(f"Prediction: **{label}**  —  Confidence: **{confidence:.3f}** ({confidence*100:.1f}%)")

            # Nice colored banner for malignant vs benign (simple heuristic)
            low_label = label.lower()
            if "maligna" in low_label or "malignant" in low_label:
                st.markdown(
                    f"<div style='padding:10px;border-radius:6px;background:#ffdddd;color:#8b0000'>"
                    f"<strong>⚠️ Likely MALIGNANT</strong> — consult a healthcare professional.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='padding:10px;border-radius:6px;background:#ddffdd;color:#006400'>"
                    f"<strong>✅ Likely BENIGN</strong> — still consult a professional if unsure.</div>",
                    unsafe_allow_html=True,
                )

            # show raw model output optionally
            if st.checkbox("Show raw model output (probabilities)"):
                prob_str = {i: float(p) for i, p in enumerate(pred_vector)}
                st.write(prob_str)

st.markdown("---")
st.caption("Preprocessing: resize/crop to 224×224, scale to [-1, 1] (same as original).")
st.caption("Run with: `streamlit run app.py`")