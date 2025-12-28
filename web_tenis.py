import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# Configuración de página
st.set_page_config(page_title="Tenis Lab Pro", layout="centered")
st.title("🎾 Tenis Lab")

# Selector de archivos
uploaded_file = st.file_uploader("Elegí un video de tu galería", type=['mp4', 'mov', 'avi'])

# Configuración lateral
mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])
run = st.sidebar.checkbox('Reproducir Análisis', value=True)

# Conexiones del esqueleto (13 puntos)
CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

if uploaded_file is not None:
    # --- IMPORTACIÓN ESTÁNDAR PERO PROTEGIDA ---
    try:
        mp_pose = mp.solutions.pose
        pose_engine = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, # Versión rápida para móviles
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        st.error(f"Error al iniciar la IA: {e}")
        st.stop()

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    # Índices según mano (Muñeca y Cadera)
    idx_m = 16 if mano_dominante == "Derecha" else 15
    idx_c = 24 if mano_dominante == "Derecha" else 23

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Redimensionar para fluidez en la nube
        h_orig, w_orig = frame.shape[:2]
        ancho_celu = 480
        alto_celu = int((h_orig / w_orig) * ancho_celu)
        frame = cv2.resize(frame, (ancho_celu, alto_celu))
        
        # Procesar IA
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_engine.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Dibujar esqueleto (Líneas negras)
            for start, end in CONEXIONES:
                p1, p2 = lm[start], lm[end]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(frame, (int(p1.x*ancho_celu), int(p1.y*alto_celu)), 
                             (int(p2.x*ancho_celu), int(p2.y*alto_celu)), (0, 0, 0), 2)
            
            # Dibujar puntos clave (Rojos)
            puntos_id = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            for i in puntos_id:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*ancho_celu), int(p.y*alto_celu)), 3, (0, 0, 255), -1)

            # Plano vertical y distancia
            cx = int(lm[idx_c].x * ancho_celu)
            mx = int(lm[idx_m].x * ancho_celu)
            cv2.line(frame, (cx, 0), (cx, alto_celu), (255, 255, 255), 1)
            dist = int(mx - cx if mano_dominante == "Derecha" else cx - mx)
            cv2.putText(frame, f"Eje: {dist}px", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        st_frame.image(frame, channels="BGR", use_container_width=True)
        time.sleep(0.01)

    cap.release()
