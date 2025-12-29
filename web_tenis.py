import streamlit as st
import cv2
import tempfile
import time
import mediapipe as mp
import numpy as np

# Configuración básica
st.set_page_config(page_title="Tenis Lab", layout="centered")
st.title("🎾 Tenis Lab: Análisis")

# Inicializamos MediaPipe (Configuración estándar)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Subida de archivo
uploaded_file = st.file_uploader("Subí tu video", type=['mp4', 'mov', 'avi'])
run = st.checkbox('Procesar Video', value=True)

if uploaded_file is not None:
    # Crear archivo temporal
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() # El cuadro donde se verá el video
    
    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            break # Fin del video

        # 1. Redimensionar (Vital para que no se trabe en la web)
        frame = cv2.resize(frame, (640, 480))
        
        # 2. Procesar con MediaPipe
        # Convertir a RGB porque OpenCV usa BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # 3. Dibujar el esqueleto (Si detecta algo)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

        # 4. Mostrar en Streamlit
        stframe.image(frame, channels="BGR", use_container_width=True)
        
        # --- EL ARREGLO DEL CONGELAMIENTO ---
        # Esta pequeña pausa permite que Streamlit respire y no sature la memoria
        time.sleep(0.05)

    cap.release()
