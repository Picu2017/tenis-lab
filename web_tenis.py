import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

st.set_page_config(page_title="Tenis Lab Pro", layout="centered")
st.title("🎾 Tenis Lab")

uploaded_file = st.file_uploader("Elegí un video de tu galería", type=['mp4', 'mov', 'avi'])

mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])
run = st.sidebar.checkbox('Reproducir Análisis', value=True)

# Conexiones del esqueleto
CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

if uploaded_file is not None:
    # --- SOLUCIÓN AL ERROR DE PERMISOS ---
    try:
        # Usamos model_complexity=1 para evitar el archivo 'lite' bloqueado
        pose_engine = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        st.error(f"Error de permisos en el servidor: {e}")
        st.info("Intentando recargar el motor de IA...")
        st.stop()

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    idx_m = 16 if mano_dominante == "Derecha" else 15
    idx_c = 24 if mano_dominante == "Derecha" else 23

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Redimensionar para fluidez
        frame = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        h, w, _ = frame.shape

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_engine.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            for start, end in CONEXIONES:
                p1, p2 = lm[start], lm[end]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 0), 2)
            
            for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, (0, 0, 255), -1)

            cx = int(lm[idx_c].x * w)
            cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)
            mx = int(lm[idx_m].x * w)
            dist = int(mx - cx if mano_dominante == "Derecha" else cx - mx)
            cv2.putText(frame, f"Eje: {dist}px", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        st_frame.image(frame, channels="BGR", use_container_width=True)
        time.sleep(0.01)

    cap.release()
