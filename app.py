import streamlit as st
import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime
from deepface import DeepFace
import time
import threading

# Disable TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Constants
ATTENDANCE_FILE = "attendance.csv"
EMPLOYEE_FOLDER = "employees"

# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# Load known faces from multiple images per person
def load_known_faces():
    known_faces = {}
    if os.path.exists(EMPLOYEE_FOLDER):
        for person in os.listdir(EMPLOYEE_FOLDER):
            person_path = os.path.join(EMPLOYEE_FOLDER, person)
            if os.path.isdir(person_path):
                known_faces[person] = [os.path.join(person_path, img) for img in os.listdir(person_path) if img.endswith(('jpg', 'png'))]
    return known_faces

known_faces = load_known_faces()

# Initialize session state
if "attendance_running" not in st.session_state:
    st.session_state.attendance_running = False

def mark_attendance(name):
    """Marks attendance only once per session."""
    df = pd.read_csv(ATTENDANCE_FILE)
    now = datetime.now()
    date_today = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    if not ((df["Name"] == name) & (df["Date"] == date_today)).any():
        new_entry = pd.DataFrame([[name, date_today, time_now]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        st.success(f"✅ {name} marked at {time_now}")

def process_frame(frame):
    """Runs face recognition in a separate thread."""
    temp_image = "temp.jpg"
    cv2.imwrite(temp_image, frame)
    for name, img_paths in known_faces.items():
        for img_path in img_paths:
            try:
                result = DeepFace.verify(temp_image, img_path, model_name="VGG-Face", distance_metric="cosine", enforce_detection=False)
                if result["verified"]:
                    mark_attendance(name)
                    return  # Exit once a match is found
            except:
                pass

def face_recognition():
    """Runs face recognition with real-time video feed in Streamlit."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        st.error("❌ Camera not found! Please check your webcam.")
        return

    frame_holder = st.empty()
    last_capture_time = time.time()

    while st.session_state.attendance_running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera disconnected!")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame_rgb, channels="RGB", use_container_width=True)

        if time.time() - last_capture_time >= 1:
            threading.Thread(target=process_frame, args=(frame,)).start()
            last_capture_time = time.time()

    cap.release()
    st.session_state.attendance_running = False
    st.warning("⏹ Attendance system stopped")

# Streamlit UI
st.title("📸 Face Recognition Attendance System")

# Upload user images
st.header("Upload Your Images")
uploaded_files = st.file_uploader("Upload 3-10 images of yourself", type=["jpg", "png"], accept_multiple_files=True)
user_name = st.text_input("Enter your name")

if st.button("Save Images"):
    if user_name and len(uploaded_files) >= 3:
        user_folder = os.path.join(EMPLOYEE_FOLDER, user_name)
        os.makedirs(user_folder, exist_ok=True)
        for i, file in enumerate(uploaded_files):
            file_path = os.path.join(user_folder, f"{i+1}.jpg")
            with open(file_path, "wb") as f:
                f.write(file.read())
        st.success("✅ Images saved successfully!")
        known_faces = load_known_faces()
    else:
        st.warning("⚠️ Please upload at least 3 images and enter your name.")

# Start & Stop Buttons Always Visible
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Attendance"):
        if not st.session_state.attendance_running:
            st.session_state.attendance_running = True
            face_recognition()
        else:
            st.warning("⚠️ Attendance is already running!")

with col2:
    if st.button("⏹ Stop Attendance"):
        st.session_state.attendance_running = False
        st.warning("⏹ Attendance system stopped")

# Show stored attendance data
if st.button("📜 Show Attendance Record"):
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    else:
        st.warning("⚠️ No attendance records found!")