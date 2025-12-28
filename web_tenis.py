import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

st.set_page_config(page_title="Tenis Lab Pro", layout="wide")
st.title("🎾 Tenis Lab: Analisis de Tecnica: Marcelo")

# --- BARRA LATERAL ---
st.sidebar.title("Configuración")
uploaded_file = st.sidebar.file_uploader("Sube tu video", type=['mp4', 'mov', 'avi'])
run = st.sidebar.checkbox('Ejecutar Video (Play/Pause)', value=True)
mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])

# Definición de conexiones para el esqueleto de 13 puntos
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Tren superior
    (11, 23), (12, 24), (23, 24),                   # Tronco
    (23, 25), (25, 27), (24, 26), (26, 28)          # Tren inferior
]

PUNTOS_CONTROL = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

if mano_dominante == "Derecha":
    IDX_MUÑECA, IDX_CADERA = 16, 24
else: 
    IDX_MUÑECA, IDX_CADERA = 15, 23

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(tfile.name)

    st_video = st.empty() 

    while st.session_state.cap.isOpened():
        if not run:
            st.stop() 

        ret, frame = st.session_state.cap.read()
        if not ret:
            st.session_state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Proporción automática
        h_orig, w_orig = frame.shape[:2]
        ancho_web = 850
        alto_web = int((h_orig / w_orig) * ancho_web)
        frame = cv2.resize(frame, (ancho_web, alto_web))
        h, w = alto_web, ancho_web

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # 1. DIBUJAR LÍNEAS DEL ESQUELETO EN NEGRO
            for connection in CONEXIONES_TENIS:
                p1_idx, p2_idx = connection
                p1 = lm[p1_idx]
                p2 = lm[p2_idx]
                
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 1) 

            # 2. DIBUJAR PUNTOS EN ROJO (Radio 2)
            for idx in PUNTOS_CONTROL:
                punto = lm[idx]
                if punto.visibility > 0.5:
                    px, py = int(punto.x * w), int(punto.y * h)
                    # Rojo: (0, 0, 255). La cabeza (0) se mantiene en Blanco (255, 255, 255)
                    color = (255, 255, 255) if idx == 0 else (0, 0, 255)
                    cv2.circle(frame, (px, py), 2, color, -1)

            # 3. PLANO DEL CUERPO Y DISTANCIA
            cadera_x = int(lm[IDX_CADERA].x * w)
            muñeca_x = int(lm[IDX_MUÑECA].x * w)
            cv2.line(frame, (cadera_x, 0), (cadera_x, h), (255, 255, 255), 1)
            
            dist_px = int(muñeca_x - cadera_x if mano_dominante == "Derecha" else cadera_x - muñeca_x)
            cv2.putText(frame, f"Dist. Plano: {dist_px}px", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        st_video.image(frame, channels="BGR", width="stretch")
        time.sleep(0.01)
else:
    st.info("Sube tu video para iniciar el análisis.")