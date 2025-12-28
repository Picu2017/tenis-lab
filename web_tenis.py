import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import shutil
import sys

# --- CONFIGURACIÓN DE PERMISOS (EL TRUCO) ---
def inicializar_entorno():
    import mediapipe as mp
    # Ubicación original y ubicación con permisos
    path_original = os.path.dirname(mp.__file__)
    path_temporal = "/tmp/mediapipe"
    
    # Copiamos la librería entera a una zona con permisos de escritura
    if not os.path.exists(path_temporal):
        shutil.copytree(path_original, path_temporal, dirs_exist_ok=True)
    
    # Forzamos a Python a leer mediapipe desde /tmp
    if "/tmp" not in sys.path:
        sys.path.insert(0, "/tmp")
    
    from mediapipe.python.solutions import pose as mp_pose
    return mp_pose

# Ejecutamos el parche
try:
    mp_pose = inicializar_entorno()
except Exception as e:
    st.error(f"Error preparando entorno: {e}")
    st.stop()

st.set_page_config(page_title="Tenis Lab Pro", layout="centered")
st.title("🎾 Tenis Lab: Análisis Biomecánico")

# --- INICIALIZACIÓN DE IA ---
@st.cache_resource
def load_pose_engine():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = load_pose_engine()

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Subí tu video aquí", type=['mp4', 'mov', 'avi'])
mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])
run = st.sidebar.checkbox('Reproducir Análisis', value=True)

# Esqueleto simplificado para tenis
CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]



if uploaded_file is not None:
    # Guardamos video en zona segura
    tfile = tempfile.NamedTemporaryFile(delete=False, dir='/tmp', suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    frame_window = st.empty() 

    # Marcadores de análisis
    idx_m = 16 if mano_dominante == "Derecha" else 15
    idx_c = 24 if mano_dominante == "Derecha" else 23

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        h, w = frame.shape[:2]

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Dibujar esqueleto negro
            for s, e in CONEXIONES:
                p1, p2 = lm[s], lm[e]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 0), 2)
            
            # Dibujar puntos rojos
            for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, (0, 0, 255), -1)

            # Eje vertical y distancia
            cx = int(lm[idx_c].x * w)
            mx = int(lm[idx_m].x * w)
            cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)
            dist = int(mx - cx if mano_dominante == "Derecha" else cx - mx)
            cv2.putText(frame, f"Eje: {dist}px", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame_window.image(frame, channels="BGR", use_container_width=True)
        time.sleep(0.01)

    cap.release()
    os.unlink(tfile.name)
