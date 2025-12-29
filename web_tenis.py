import streamlit as st
import cv2
import tempfile
import time
import os
import mediapipe as mp

# Configuración simple
st.set_page_config(page_title="Tenis Lab", layout="centered")
st.title("🎾 Tenis Lab: Análisis")

# --- SOLUCIÓN AL ERROR DE IMPORTACIÓN ---
# En lugar de importar 'solutions' arriba, lo traemos directo de mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Inicializamos el modelo de forma simple (sin try-except complejos)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Interfaz
uploaded_file = st.file_uploader("Subí tu video", type=['mp4', 'mov', 'avi'])
run = st.checkbox('Procesar Video', value=True)

if uploaded_file is not None:
    # Guardar archivo temporal
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() # Placeholder para el video
    
    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            # Si termina el video, paramos para no consumir memoria infinita
            # O puedes usar cap.set(cv2.CAP_PROP_POS_FRAMES, 0) para loop
            break 

        # Redimensionar para que no pese tanto (evita que se congele)
        frame = cv2.resize(frame, (640, 480))
        
        # Procesar
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Dibujar
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # Mostrar
        stframe.image(frame, channels="BGR", use_column_width=True)
        
        # --- EL PARCHE ANTI-CONGELAMIENTO ---
        # Esta linea libera la CPU para que Streamlit pueda respirar
        time.sleep(0.03) 

    cap.release()
    tfile.close()
