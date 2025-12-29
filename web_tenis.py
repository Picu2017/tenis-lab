import streamlit as st
import cv2
import tempfile
import time
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Tenis Lab", layout="centered")

# --- INICIO ---
# Usamos la configuración estándar que sabemos que funciona con la versión 0.10.9
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

st.title("🎾 Tenis Lab: Análisis")

uploaded_file = st.file_uploader("Subí tu video", type=['mp4', 'mov', 'avi'])
run = st.checkbox('Procesar', value=True)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    
    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar es clave para evitar el congelamiento
        frame = cv2.resize(frame, (640, 360))
        
        # Procesar
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Dibujar
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

        stframe.image(frame, channels="BGR", use_container_width=True)
        time.sleep(0.04) # Pausa anti-congelamiento

    cap.release()
