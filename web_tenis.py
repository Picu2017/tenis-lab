import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import time

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Pro", layout="centered")

# Estilos para limpiar la interfaz
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tenis Lab: Análisis de Golpe")

# --- MOTOR IA (Cacheado para que no recargue lento) ---
@st.cache_resource
def cargar_modelo():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = cargar_modelo()

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Carga tu video", type=['mp4', 'mov', 'avi'])
run = st.checkbox('Analizar', value=True)

# Conexiones Tenis (Brazos, Tronco, Piernas)
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]

if uploaded_file is not None:
    # 1. GUARDAR VIDEO TEMPORAL
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    tfile.close() # <--- ¡ESTA ES LA LÍNEA CLAVE QUE FALTABA!
    
    # 2. ABRIR VIDEO
    cap = cv2.VideoCapture(tfile.name)
    
    # Verificamos si abrió bien
    if not cap.isOpened():
        st.error("Error al leer el video. Intenta con otro formato.")
    
    st_frame = st.empty() # Contenedor de imagen
    
    # Datos de pantalla
    w_screen = 640
    h_screen = 360

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            # Si termina el video, salimos del bucle
            break

        # Redimensionar (Clave para velocidad en nube)
        frame = cv2.resize(frame, (w_screen, h_screen))
        
        # Procesar IA
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Dibujar (Manual para mejor estética)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Dibujar líneas blancas
            for p_start, p_end in CONEXIONES_TENIS:
                if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                    pt1 = (int(lm[p_start].x * w_screen), int(lm[p_start].y * h_screen))
                    pt2 = (int(lm[p_end].x * w_screen), int(lm[p_end].y * h_screen))
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 1)

            # Dibujar puntos rojos (Solo los necesarios)
            for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cx, cy = int(p.x * w_screen), int(p.y * h_screen)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Mostrar frame
        st_frame.image(frame, channels="BGR", use_container_width=True)
        
        # Control de velocidad (FPS)
        # Si va muy lento, baja este número (ej: 0.01)
        # Si va muy rápido, súbelo (ej: 0.1)
        time.sleep(0.02)

    cap.release()
