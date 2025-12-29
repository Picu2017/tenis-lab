import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import shutil
import time

# --- ARQUITECTURA DE PERMISOS (SOLUCIÓN FINAL) ---
# Definimos rutas
MP_PATH = os.path.dirname(mp.__file__)
MODEL_DIR = os.path.join(MP_PATH, "modules/pose_landmark")
TMP_MODEL_DIR = "/tmp/mediapipe/modules/pose_landmark"

# Hack: Creamos la estructura de carpetas en /tmp y copiamos los modelos
# Esto evita que MediaPipe intente escribir en la carpeta de site-packages
if not os.path.exists(TMP_MODEL_DIR):
    os.makedirs(TMP_MODEL_DIR, exist_ok=True)
    # Copiamos solo los archivos .tflite necesarios si existen
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            if file.endswith(".tflite"):
                shutil.copy2(os.path.join(MODEL_DIR, file), os.path.join(TMP_MODEL_DIR, file))

# Obligamos al sistema a mirar en /tmp
os.environ['MEDIAPIPE_MODEL_PATH'] = '/tmp/mediapipe'

# Ahora sí, cargamos el motor
from mediapipe.python.solutions import pose as mp_pose

st.set_page_config(page_title="Tenis Lab Pro", layout="centered")
st.title("🎾 Tenis Lab: Análisis Biomecánico")

@st.cache_resource
def load_pose_model():
    # model_complexity=0 es vital para usar el modelo 'lite' que movimos a /tmp
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

try:
    pose_engine = load_pose_model()
except Exception as e:
    st.error(f"Error de inicialización: {e}")
    st.stop()

# --- INTERFAZ Y PROCESAMIENTO ---
uploaded_file = st.file_uploader("Subí tu video aquí", type=['mp4', 'mov', 'avi'])
mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])
run = st.sidebar.checkbox('Reproducir Análisis', value=True)

CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]



if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, dir='/tmp', suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    frame_window = st.empty() 

    idx_m = 16 if mano_dominante == "Derecha" else 15
    idx_c = 24 if mano_dominante == "Derecha" else 23

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        h, w = frame.shape[:2]

        results = pose_engine.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            for s, e in CONEXIONES:
                p1, p2 = lm[s], lm[e]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 0), 2)
            
            for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, (0, 0, 255), -1)

            cx = int(lm[idx_c].x * w)
            mx = int(lm[idx_m].x * w)
            cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)
            dist = int(mx - cx if mano_dominante == "Derecha" else cx - mx)
            cv2.putText(frame, f"Eje: {dist}px", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame_window.image(frame, channels="BGR", use_container_width=True)
        time.sleep(0.01)

    cap.release()
    os.unlink(tfile.name)
