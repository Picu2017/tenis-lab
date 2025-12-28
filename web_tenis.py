import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import os

# Configuración de página
st.set_page_config(page_title="Tenis Lab Pro", layout="centered")
st.title("🎾 Tenis Lab: Análisis Biomecánico")

# --- INICIALIZACIÓN DEL MOTOR IA (Sin descargas ni escrituras) ---
@st.cache_resource
def get_pose_instance():
    # Usamos la clase base para evitar que busque archivos .tflite en carpetas protegidas
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

try:
    pose_engine = get_pose_instance()
except Exception as e:
    st.error(f"Error técnico: {e}")
    st.info("Por favor, haz clic en 'Manage App' -> 'Reboot App' en el panel de Streamlit.")
    st.stop()

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Subí tu video aquí", type=['mp4', 'mov', 'avi'])

st.sidebar.title("Configuración")
mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])
run = st.sidebar.checkbox('Reproducir Análisis', value=True)

# Esqueleto de 13 puntos clave
CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

if uploaded_file is not None:
    # Guardamos el video en la carpeta /tmp (única con permisos)
    tfile = tempfile.NamedTemporaryFile(delete=False, dir='/tmp', suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close() # Cerramos para que OpenCV pueda leerlo
    
    cap = cv2.VideoCapture(tfile.name)
    frame_window = st.empty() 

    # Índices de muñeca y cadera según lateralidad
    idx_m = 16 if mano_dominante == "Derecha" else 15
    idx_c = 24 if mano_dominante == "Derecha" else 23

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Redimensionar para que el video sea fluido en el celular
        frame = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        h, w = frame.shape[:2]

        # Procesar con IA
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_engine.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Dibujar esqueleto negro
            for s, e in CONEXIONES:
                p1, p2 = lm[s], lm[e]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 0), 2)
            
            # Puntos rojos
            for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, (0, 0, 255), -1)

            # Eje vertical y distancia de impacto
            cx = int(lm[idx_c].x * w)
            mx = int(lm[idx_m].x * w)
            cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)
            dist = int(mx - cx if mano_dominante == "Derecha" else cx - mx)
            cv2.putText(frame, f"Eje: {dist}px", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostrar frame
        frame_window.image(frame, channels="BGR", use_container_width=True)
        time.sleep(0.01)

    cap.release()
    os.remove(tfile.name)
else:
    st.info("Subí un video para analizar la biomecánica del golpe.")
