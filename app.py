import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import subprocess

# Path to folder containing student images
IMAGE_PATH = "students/"
ATTENDANCE_FILE = "attendance.csv"
SIMILARITY_THRESHOLD = 0.40  # Adjusted threshold for 60% similarity
FRAME_RESIZE = (320, 240)  # Resize frame for performance
video_capture = None
running = False

# Ensure the students directory exists
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)
    print(f"Created missing directory: {IMAGE_PATH}")

def mark_attendance(name):
    now = datetime.now()
    time_string = now.strftime("%H:%M:%S")
    date_string = now.strftime("%Y-%m-%d")
    
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        df = pd.read_csv(ATTENDANCE_FILE)
    
    # Check if the person is already marked for today
    if not ((df["Name"] == name) & (df["Date"] == date_string)).any():
        new_entry = pd.DataFrame([[name, date_string, time_string]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"{name} marked present at {time_string}")

def recognize_faces():
    global video_capture, running
    running = True
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FPS, 15)
    session_attendance = set()
    
    while running:
        success, frame = video_capture.read()
        if not success:
            break
        frame = cv2.resize(frame, FRAME_RESIZE)
        try:
            result = DeepFace.find(frame, db_path=IMAGE_PATH, model_name="Facenet", enforce_detection=False)
            if result and isinstance(result, list) and len(result) > 0:
                df = result[0]
                if not df.empty:
                    min_distance = df["distance"].iloc[0]
                    similarity_score = (1 - min_distance) * 100
                    matched_name = os.path.splitext(os.path.basename(df["identity"].iloc[0]))[0]
                    
                    cv2.putText(frame, f"{matched_name} ({similarity_score:.2f}%)", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if similarity_score >= 60 and matched_name not in session_attendance:
                        mark_attendance(matched_name)
                        session_attendance.add(matched_name)
        except Exception as e:
            print(f"Error verifying face: {e}")
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        camera_label.config(image=img)
        camera_label.image = img
    
    video_capture.release()
    cv2.destroyAllWindows()

def start_camera():
    threading.Thread(target=recognize_faces, daemon=True).start()

def stop_camera():
    global running
    running = False
    if video_capture:
        video_capture.release()
    cv2.destroyAllWindows()

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
    if not file_path:
        return
    name = simpledialog.askstring("Input", "Enter Name:")
    if name:
        new_path = os.path.join(IMAGE_PATH, f"{name}.jpg")
        os.rename(file_path, new_path)
        messagebox.showinfo("Success", "Image uploaded successfully")

def show_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        subprocess.Popen(["notepad.exe", ATTENDANCE_FILE])
    else:
        messagebox.showerror("Error", "Attendance file not found")

# GUI Setup
root = tk.Tk()
root.title("Face Recognition Attendance")

camera_label = tk.Label(root)
camera_label.pack()

btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload.pack()

btn_start = tk.Button(root, text="Start Camera", command=start_camera)
btn_start.pack()

btn_stop = tk.Button(root, text="Stop Camera", command=stop_camera)
btn_stop.pack()

btn_show_attendance = tk.Button(root, text="Show Attendance", command=show_attendance)
btn_show_attendance.pack()

root.mainloop()
