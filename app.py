import streamlit as st
import cv2
import os
import pandas as pd
import threading
from datetime import datetime
from deepface import DeepFace

# Disable oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Attendance file setup
ATTENDANCE_FILE = "attendance.xlsx"
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_excel(ATTENDANCE_FILE, index=False)

# Load known faces
employee_folder = "employees"
known_faces = {name.split(".")[0]: os.path.join(employee_folder, name) for name in os.listdir(employee_folder)}

# Track attendance
marked_attendance = set()

# Store attendance system state
if "attendance_running" not in st.session_state:
    st.session_state.attendance_running = False
if "frame" not in st.session_state:
    st.session_state.frame = None

def mark_attendance(name):
    """Marks attendance only once per session."""
    if name not in marked_attendance:
        now = datetime.now()
        date_today = now.strftime("%Y-%m-%d")
        time_now = now.strftime("%H:%M:%S")

        # Load existing Excel file or create a new one
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_excel(ATTENDANCE_FILE)
        else:
            df = pd.DataFrame(columns=["Name", "Date", "Time"])

        # Append new data
        new_entry = pd.DataFrame([[name, date_today, time_now]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)

        # Save to Excel
        df.to_excel(ATTENDANCE_FILE, index=False)

        st.success(f"✅ {name} marked at {time_now} on {date_today}")
        marked_attendance.add(name)

def face_recognition():
    """Runs face recognition in a separate thread."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Works for most Windows users

    while st.session_state.attendance_running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera not found! Please check your webcam.")
            break

        # Store the latest frame
        st.session_state.frame = frame

        temp_image = "temp.jpg"
        cv2.imwrite(temp_image, frame)

        for name, img_path in known_faces.items():
            try:
                result = DeepFace.verify(temp_image, img_path, model_name="Facenet", distance_metric="cosine", enforce_detection=False)
                if result["verified"]:
                    mark_attendance(name)
                    break  # Stop checking once recognized
            except:
                pass

    cap.release()
    st.session_state.attendance_running = False  # Stop session state
    st.warning("⏹ Attendance system stopped")

# Streamlit UI
st.title("Face Recognition Attendance System")

# Start Attendance Button
if st.button("Start Attendance"):
    if not st.session_state.attendance_running:
        st.session_state.attendance_running = True
        threading.Thread(target=face_recognition, daemon=True).start()
    else:
        st.warning("⚠️ Attendance is already running!")

# Stop Attendance Button
if st.button("Stop Attendance"):
    st.session_state.attendance_running = False
    st.warning("⏹ Attendance system stopped")

# Show the latest webcam frame
if st.session_state.frame is not None:
    frame_rgb = cv2.cvtColor(st.session_state.frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB")

# Show stored attendance data
if st.button("Show Attendance Record"):
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        st.dataframe(df)
    else:
        st.warning("⚠️ No attendance records found!")
