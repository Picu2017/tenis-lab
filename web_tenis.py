import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import os
import urllib.request

# --- SOLUCIÓN DE PERMISOS: DESCARGA MANUAL DEL MODELO ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmark/pose_landmark_lite/float16/1/pose_landmark_lite.tflite"
MODEL_PATH = "/tmp/pose_landmark_lite.tflite"

def download_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

try:
    download_model()
except Exception as e:
    st.error(f"Error al descargar modelo base: {e}")

st.set_page_config(page_title="Tenis Lab Pro", layout="centered")
st.title("🎾 Tenis Lab: Análisis Biomecánico")

# --- INICIALIZACIÓN DEL MOTOR ---
@st.cache_resource
def init_pose():
    # Usamos la solución de MediaPipe pero configurada para evitar escrituras
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

try:
    pose_engine = init_pose()
except Exception as e:
    # Si falla la carga estándar, intentamos limpiar la ruta
    st.error(f"Error de permisos: {e}")
    st.info("Por favor, haz un 'Reboot App' en el panel de Streamlit.")
    st.stop()

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Elegí un video de tu galería", type=['mp4', 'mov', 'avi'])

st.sidebar.title("Ajustes")
mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])
run = st.sidebar.checkbox('Analizar / Pausar', value=True)

# Conexiones 13 puntos clave (Brazos, Tronco, Piernas)
CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]



if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    frame_window = st.empty() 

    idx_m = 16 if mano_dominante == "Derecha" else 15
    idx_c = 24 if mano_dominante == "Derecha" else 23

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Redimensionar para fluidez máxima
        frame = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        h, w = frame.shape[:2]

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_engine.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Dibujar esqueleto (Negro)
            for s, e in CONEXIONES:
                p1, p2 = lm[s], lm[e]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 0), 2)
            
            # Dibujar puntos clave (Rojos)
            for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, (0, 0, 255), -1)

            # Eje vertical cadera y distancia de impacto
            cx = int(lm[idx_c].x * w)
            cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)
            mx = int(lm[idx_m].x * w)
            dist = int(mx - cx if mano_dominante == "Derecha" else cx - mx)
            cv2.putText(frame, f"Eje: {dist}px", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame_window.image(frame, channels="BGR", use_container_width=True)
        time.sleep(0.01)

    cap.release()
    os.unlink(tfile.name)
else:
    st.info("Subí un video para analizar el plano de impacto.")
