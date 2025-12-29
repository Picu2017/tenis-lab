import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import time

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Pro", layout="centered")

st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tenis Lab: Análisis Paso a Paso")

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

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Carga tu video", type=['mp4', 'mov', 'avi'])
run = st.checkbox('Analizar', value=True)

CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    tfile.close() # Importante
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    # Reducimos resolución para ayudar a la transmisión
    w_screen = 480 
    h_screen = 270
    
    frame_counter = 0

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Redimensionar (Más chico = Más fluido en la nube)
        frame = cv2.resize(frame, (w_screen, h_screen))
        
        # Procesar
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Dibujar
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            for p_start, p_end in CONEXIONES_TENIS:
                if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                    pt1 = (int(lm[p_start].x * w_screen), int(lm[p_start].y * h_screen))
                    pt2 = (int(lm[p_end].x * w_screen), int(lm[p_end].y * h_screen))
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 1)

            for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*w_screen), int(p.y*h_screen)), 3, (0, 0, 255), -1)

        # Contador visual para verificar que avanza
        cv2.putText(frame, f"Frame: {frame_counter}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Mostrar
        st_frame.image(frame, channels="BGR", use_container_width=True)
        
        # --- FRENO DE MANO ---
        # 0.1 segundos es lento, pero garantiza que el navegador reciba la imagen
        time.sleep(0.1)

    cap.release()
