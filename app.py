import streamlit as st
import cv2
import random


st.set_page_config(
    page_title="AI Fire Detection",
    page_icon="🔥",
    layout="centered"
)

# Title
st.title("🔥 AI Fire Detection System")

st.markdown("Real-time Fire Detection using Computer Vision")

# Sidebar
st.sidebar.header("Project Info")

st.sidebar.write("""
AI-based Fire Detection System using:
- OpenCV
- Streamlit
- Computer Vision
- TensorFlow Lite (Upcoming)
""")

# Camera Start
start = st.checkbox("Start Camera")

frame_window = st.image([])

camera = cv2.VideoCapture(0)

while start:
    success, frame = camera.read()

    if not success:
        st.error("Camera not working")
        break

    # Convert frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dummy AI prediction
    prediction = random.random()

    confidence = round(prediction * 100, 2)

    if prediction > 0.5:
        label = f"🔥 FIRE DETECTED ({confidence}%)"
    else:
        label = f"✅ NO FIRE ({confidence}%)"

    # Put label
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    frame_window.image(frame)

camera.release()