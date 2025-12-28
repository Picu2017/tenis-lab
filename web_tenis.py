import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

st.set_page_config(page_title="Tenis Lab Pro", layout="centered")
st.title("🎾 Tenis Lab: Análisis en tiempo real")

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_pose():
    # Usamos model_complexity 0 para que sea ultra rápido y no se trabe
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = load_pose()

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Subí tu video aquí", type=['mp4', 'mov', 'avi'])
mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])
run = st.sidebar.checkbox('Reproducir / Pausar', value=True)

# Conexiones 13 puntos
CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    # Creamos un contenedor vacío donde irá el video
    frame_placeholder = st.empty()

    idx_m = 16 if mano_dominante == "Derecha" else 15
    idx_c = 24 if mano_dominante == "Derecha" else 23

    while cap.isOpened() and run:
        ret, frame = cap.read()
        
        if not ret:
            # Si termina el video, vuelve a empezar (loop)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Redimensionamos a un tamaño pequeño para que viaje rápido por internet
        frame = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        h, w, _ = frame.shape

        # Procesar con IA
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Dibujar esqueleto (Líneas negras)
            for s, e in CONEXIONES:
                p1, p2 = lm[s], lm[e]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 0), 2)
            
            # Dibujar articulaciones (Puntos rojos)
            for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*w), int(p.y*h)), 4, (0, 0, 255), -1)

            # Eje vertical cadera y distancia al plano
            cx = int(lm[idx_c].x * w)
            cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)
            mx = int(lm[idx_m].x * w)
            dist = int(mx - cx if mano_dominante == "Derecha" else cx - mx)
            cv2.putText(frame, f"Eje: {dist}px", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # MOSTRAR EL FRAME EN EL CONTENEDOR
        # Usamos channels="BGR" porque OpenCV lee así, y st.image lo muestra
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        # EL PASO MÁGICO: Un pequeño respiro de 0.01 segundos
        # Esto permite que Streamlit envíe la imagen al navegador antes de procesar la siguiente
        time.sleep(0.01)

    cap.release()
else:
    st.info("Subí un video para analizar el plano de golpe.")
