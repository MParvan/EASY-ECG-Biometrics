import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.bandpass import preprocess
from segmentation.rcentered import RCentered
from tensorflow.keras.models import load_model
import os

st.title("🧠 ECG Biometric Recognition")

st.sidebar.header("Upload ECG Signal File")
uploaded_file = st.sidebar.file_uploader("Choose a .txt ECG file", type=["txt"])

# Configuration
FS = 250
SAMPLE_LENGTH = 3000

# Load your trained model
@st.cache_resource
def load_deepecg_model():
    model_path = "models/deepecg_model.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.warning("Model file not found!")
        return None

model = load_deepecg_model()

if uploaded_file is not None:
    signal = np.loadtxt(uploaded_file)[:SAMPLE_LENGTH]

    st.subheader("Raw Signal")
    time = np.arange(len(signal)) / FS
    fig, ax = plt.subplots()
    ax.plot(time, signal)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Preprocessing
    pp = preprocess(fs=FS)
    filtered = pp.preprocess(signal)

    # Segmentation
    ss = RCentered()
    segments = ss.segment(filtered)

    if segments.any() and model:
        st.subheader("Model Predictions")
        segments = np.array(segments)[..., np.newaxis]
        predictions = model.predict(segments)
        predicted_labels = np.argmax(predictions, axis=1)
        st.write("Predicted identities:", predicted_labels)
