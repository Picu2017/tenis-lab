import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import gc
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tenis Lab Móvil", layout="wide")

st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        .block-container {padding: 1rem 0.5rem !important;}
        .stButton button {
            width: 100%;
            height: 60px;
            font-weight: bold;
            font-size: 24px;
        }
    </style>
""", unsafe_allow_html=True)

st.write("### 🎾 Tenis Lab: Análisis Móvil")

# --- MOTOR IA (NUEVA VERSIÓN: TASKS API) ---
# Esta versión sí permite cargar el archivo local sin errores de permiso
if 'landmarker' not in st.session_state:
    try:
        # Buscamos el archivo que subiste a GitHub
        model_path = 'pose_landmark_lite.tflite'
        
        # Si por alguna razón no está, usamos un nombre genérico para que no explote
        if not os.path.exists(model_path):
            st.warning(f"⚠️ No encuentro '{model_path}'. Asegúrate de haberlo subido a GitHub.")
            model_path = None # Esto forzará error controlado abajo

        if model_path:
            # Configuración para cargar el archivo local
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            # Creamos el detector
            st.session_state.landmarker = vision.PoseLandmarker.create_from_options(options)
        else:
            st.session_state.landmarker = None

    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        st.session_state.landmarker = None

# --- MEMORIA DE NAVEGACIÓN ---
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

# --- CARGA DE VIDEO ---
uploaded_file = st.file_uploader("Toca para elegir video", type=['mp4', 'mov', 'avi'])

# CONEXIONES (Igual que antes)
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]
PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

if uploaded_file is not None and st.session_state.landmarker is not None:
    # Guardado temporal
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        
        # --- CONTROLES ---
        col_video, col_controls = st.columns([1, 100])
        
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
            # Achicar video si es 4K (Optimización Celular)
            h, w = frame.shape[:2]
            ANCHO_MAXIMO = 640 
            if w > ANCHO_MAXIMO:
                factor = ANCHO_MAXIMO / w
                nuevo_alto = int(h * factor)
                frame = cv2.resize(frame, (ANCHO_MAXIMO, nuevo_alto))
                h, w = frame.shape[:2]

            # Convertir a formato MediaPipe Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # DETECCIÓN CON EL NUEVO MOTOR
            detection_result = st.session_state.landmarker.detect(mp_image)
            
            # DIBUJADO (Adaptado al nuevo formato de resultado)
            if detection_result.pose_landmarks:
                # El nuevo API devuelve una lista de listas. Tomamos la primera (primer persona).
                lm = detection_result.pose_landmarks[0]
                
                for p_start, p_end in CONEXIONES_TENIS:
                    # Chequeamos visibilidad (ahora es .visibility)
                    if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                        pt1 = (int(lm[p_start].x * w), int(lm[p_start].y * h))
                        pt2 = (int(lm[p_end].x * w), int(lm[p_end].y * h))
                        cv2.line(frame, pt1, pt2, (0, 0, 0), 2, cv2.LINE_AA)

                for i in PUNTOS_CLAVE:
                    p = lm[i]
                    if p.visibility > 0.5:
                        center = (int(p.x*w), int(p.y*h))
                        cv2.circle(frame, center, 4, (0, 0, 255), -1, cv2.LINE_AA)

            st.image(frame, channels="BGR", use_container_width=True)
            
        else:
            st.warning("Error leyendo el cuadro.")
            
    cap.release()
    gc.collect()

elif st.session_state.landmarker is None:
    st.error("❌ El motor de IA no se pudo iniciar. Verifica que 'pose_landmark_lite.tflite' esté en GitHub.")
else:
    st.info("👆 Sube un video para empezar")
