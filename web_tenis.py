import streamlit as st
import cv2
import tempfile
import time
import mediapipe as mp
import numpy as np

# Configuración
st.set_page_config(page_title="Tenis Lab", layout="centered")
st.title("🎾 Tenis Lab: Análisis")

# Carga directa de MediaPipe (como al principio)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

        # Redimensionamos para que no pese tanto en el navegador
        frame = cv2.resize(frame, (640, 360))
        
        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar
        results = pose.process(frame_rgb)

        # Dibujar
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

        # Mostrar
        stframe.image(frame, channels="BGR", use_container_width=True)
        
        # PEQUEÑA PAUSA (Esto evita que se congele como te pasaba antes)
        time.sleep(0.04)

    cap.release()
