import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import gc
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tenis Lab Móvil", layout="wide")

# CSS para optimizar interfaz táctil
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        .block-container {padding: 1rem 0.5rem !important;}
        .stButton button {
            width: 100%;
            height: 60px; /* Botones más altos para el dedo */
            font-weight: bold;
            font-size: 24px;
        }
    </style>
""", unsafe_allow_html=True)

st.write("### 🎾 Tenis Lab: Análisis Móvil")

# --- MOTOR IA (Instancia única por sesión) ---
if 'pose' not in st.session_state:
    mp_pose = mp.solutions.pose
    
    # INTENTO DE CARGA SEGURA (Usa el archivo local para evitar errores de nube)
    try:
        # Primero intentamos carga normal
        st.session_state.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, # 0 = Lite (Rápido para celular)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception:
        # Si falla (tu error actual), forzamos el archivo local que subiste
        # El archivo debe llamarse EXACTAMENTE 'pose_landmark_lite.tflite'
        path_modelo = 'pose_landmark_lite.tflite'
        if os.path.exists(path_modelo):
            with open(path_modelo, 'rb') as f:
                model_content = f.read()
            
            # Forzamos la carga del archivo local
            st.session_state.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_asset_path=path_modelo # <--- CLAVE: Carga manual
            )
        else:
            st.error("⚠️ No encuentro el archivo 'pose_landmark_lite.tflite' en GitHub.")

# Asignamos a variable local
if 'pose' in st.session_state:
    pose = st.session_state.pose
else:
    st.stop() # Detiene la app si no hay modelo

# --- MEMORIA DE NAVEGACIÓN ---
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

# --- CARGA DE VIDEO ---
uploaded_file = st.file_uploader("Toca para elegir video", type=['mp4', 'mov', 'avi'])

# CONEXIONES DEL ESQUELETO
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]
PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

if uploaded_file is not None:
    # Guardado temporal
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        
        # --- CONTROLES GRANDES ---
        col_video, col_controls = st.columns([1, 100]) # Truco visual
        
        # NAVEGACIÓN
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("◀"):
                st.session_state.frame_index = max(0, st.session_state.frame_index - 1)
        with c3:
            if st.button("▶"):
                st.session_state.frame_index = min(total_frames - 1, st.session_state.frame_index + 1)
        with c2:
            st.slider("Timeline", 0, total_frames - 1, key='frame_index', label_visibility="collapsed")
            st.markdown(f"<p style='text-align: center;'>Cuadro: {st.session_state.frame_index}</p>", unsafe_allow_html=True)

        # --- PROCESAMIENTO ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
        ret, frame = cap.read()
        
        if ret:
            # --- MAGIA DE OPTIMIZACIÓN (ACHICAR VIDEO 4K) ---
            h, w = frame.shape[:2]
            ANCHO_MAXIMO = 640 
            
            if w > ANCHO_MAXIMO:
                factor = ANCHO_MAXIMO / w
