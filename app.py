import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("model/fire_model.h5")

# Streamlit page config
st.set_page_config(
    page_title="AI Fire Detection",
    page_icon="🔥",
    layout="wide"
)

# Title
st.markdown(
    "<h1 style='text-align: center; color: red;'>🔥 AI Fire Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center;'>Real-Time Fire Detection using Deep Learning & Computer Vision</h4>",
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
This AI system detects fire in real time using:
- TensorFlow
- OpenCV
- CNN Model
- Streamlit
- Computer Vision
""")

# Start Camera
start = st.checkbox("▶️ Start Camera")

# Camera
camera = cv2.VideoCapture(0)

frame_placeholder = st.empty()

status_placeholder = st.empty()

confidence_placeholder = st.empty()

while start:

    success, frame = camera.read()

    if not success:
        st.error("❌ Camera not working")
        break

    # Resize image
    resized = cv2.resize(frame, (128, 128))

    # Normalize
    normalized = resized / 255.0

    # Reshape
    reshaped = np.reshape(normalized, (1, 128, 128, 3))

    # Prediction
    prediction = model.predict(reshaped, verbose=0)

    confidence = float(prediction[0][0])

    # Convert frame for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FIRE DETECTION
    if confidence > 0.7:

        label = "🔥 FIRE DETECTED"

        color = (255, 0, 0)

        status_placeholder.error(
            f"🔥 FIRE DETECTED | Confidence: {round(confidence * 100,2)}%"
        )

        confidence_placeholder.progress(int(confidence * 100))

    else:

        label = "✅ NO FIRE"

        color = (0, 255, 0)

        status_placeholder.success(
            f"✅ NO FIRE | Confidence: {round((1-confidence)*100,2)}%"
        )

        confidence_placeholder.progress(int((1-confidence) * 100))

    # Add text on frame
    cv2.putText(
        frame,
        label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    # Show frame
    frame_placeholder.image(frame, channels="RGB")

camera.release()
