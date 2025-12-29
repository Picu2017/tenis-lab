import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab", layout="centered")

# Estilos CSS (Ocultar elementos innecesarios)
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        /* Ajuste para separar el slider */
        .stSlider {margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tenis Lab: Análisis Frame a Frame")

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

# --- CARGA DE ARCHIVO ---
uploaded_file = st.file_uploader("Carga tu video", type=['mp4', 'mov', 'avi'])

# Conexiones Tenis (Esqueleto simplificado)
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]

# Puntos a dibujar (Articulaciones clave)
PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

if uploaded_file is not None:
    # 1. Guardar video en carpeta temporal del sistema (Lo más seguro para Streamlit)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close() # Cerramos bien para que se guarde
    
    # 2. Leer video
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        # --- CONTROL DESLIZANTE ---
        # Permite moverte por el video frame a frame
        frame_index = st.slider("Desliza para buscar el golpe", 0, total_frames - 1, 0)
        
        # Ir al frame seleccionado
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if ret:
            # Redimensionar (Ancho 600px para buena calidad en cel)
            h_orig, w_orig = frame.shape[:2]
            aspect = h_orig / w_orig
            new_w = 600 
            new_h = int(new_w * aspect)
            frame = cv2.resize(frame, (new_w, new_h))
            
            # Procesar IA
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            # --- DIBUJO ESTÉTICO ---
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # 1. LÍNEAS BLANCAS (Primero las líneas para que queden abajo)
                for p_start, p_end in CONEXIONES_TENIS:
                    if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                        pt1 = (int(lm[p_start].x * new_w), int(lm[p_start].y * new_h))
                        pt2 = (int(lm[p_end].x * new_w), int(lm[p_end].y * new_h))
                        # cv2.LINE_AA es la clave: hace la línea suave y no pixelada
                        cv2.line(frame, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

                # 2. PUNTOS ROJOS (Círculos pequeños y perfectos)
                for i in PUNTOS_CLAVE:
                    p = lm[i]
                    if p.visibility > 0.5:
                        center = (int(p.x*new_w), int(p.y*new_h))
                        # Radio 4, Rojo (0,0,255), Relleno (-1), Suavizado (LINE_AA)
                        cv2.circle(frame, center, 4, (0, 0, 255), -1, cv2.LINE_AA)

            # Mostrar imagen
            st.image(frame, channels="BGR", use_container_width=True)
            
        else:
            st.error("Error al leer el cuadro.")
            
    cap.release()
    # No borramos tfile para que el slider siga funcionando rápido
