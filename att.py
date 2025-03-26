import streamlit as st
import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# Load known faces
employee_folder = "employees"
known_faces = {name.split(".")[0]: os.path.join(employee_folder, name) for name in os.listdir(employee_folder)}

# Attendance file
ATTENDANCE_FILE = "attendance.csv"
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

def mark_attendance(name):
    """Mark attendance once per session."""
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(ATTENDANCE_FILE)
    df.loc[len(df)] = [name, time_string]
    df.to_csv(ATTENDANCE_FILE, index=False)
    st.success(f"✅ {name} marked at {time_string}")

# Streamlit UI
st.title("Face Recognition Attendance")
start = st.button("Start Camera")

if start:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        temp_image = "temp.jpg"
        cv2.imwrite(temp_image, frame)

        for name, img_path in known_faces.items():
            try:
                result = DeepFace.verify(temp_image, img_path, model_name="Facenet", distance_metric="cosine", enforce_detection=False)
                if result["verified"]:
                    mark_attendance(name)
                    st.image(frame, caption=name)
                    break
            except:
                pass

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
