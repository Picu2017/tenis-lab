import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import time

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Pro", layout="centered")

# Ocultar elementos molestos
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tenis Lab: Análisis Biomecánico")

# --- MOTOR IA ---
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

uploaded_file = st.file_uploader("Carga tu video", type=['mp4', 'mov', 'avi'])

CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]

if uploaded_file is not None:
    # 1. Guardar y cerrar archivo (Vital)
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    
    # 2. CALCULAR TAMAÑO PERFECTO (Para que no se vea cortado)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Fijamos un ancho seguro para internet (600px) y calculamos la altura proporcional
    new_width = 600
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)
    
    st_frame = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # --- ESTRATEGIA DE FLUIDEZ ---
        # No saltamos frames (para que veas todo), pero bajamos la resolución
        # para que viaje rápido por internet.
        
        # Redimensionar respetando la proporción original
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Procesar
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Dibujar
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Dibujamos sobre el frame
            for p_start, p_end in CONEXIONES_TENIS:
                if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                    pt1 = (int(lm[p_start].x * new_width), int(lm[p_start].y * new_height))
                    pt2 = (int(lm[p_end].x * new_width), int(lm[p_end].y * new_height))
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 1)

            for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*new_width), int(p.y*new_height)), 4, (0, 0, 255), -1)

        # Mostrar frame a frame
        st_frame.image(frame, channels="BGR", use_container_width=True)
        
        # --- CONTROL DE VELOCIDAD ---
        # Pequeña pausa para que la imagen llegue al navegador antes de enviar la siguiente
        time.sleep(0.03)

    cap.release()
