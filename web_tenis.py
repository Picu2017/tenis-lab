import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os
import gc

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Móvil", layout="centered")

st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        .stSlider {margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tenis Lab: Análisis Fino")

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
uploaded_file = st.file_uploader("Carga video", type=['mp4', 'mov', 'avi'])

CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]

PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

if uploaded_file is not None:
    # 1. Guardar video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    gc.collect() # Limpiar memoria antes de procesar

    # 2. Leer video
    cap = cv2.VideoCapture(tfile.name)
    
    # Intento de corrección para videos de celular
    # A veces esto ayuda a leer videos con metadatos complejos
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1) 

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        # --- SLIDER ---
        st.write(f"🎞️ Frames detectados: {total_frames}")
        frame_index = st.slider("Desliza para buscar el golpe", 0, total_frames - 1, 0)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if ret:
            # Redimensionar (Mantener ligero)
            h_orig, w_orig = frame.shape[:2]
            aspect = h_orig / w_orig
            new_w = 500 
            new_h = int(new_w * aspect)
            frame = cv2.resize(frame, (new_w, new_h))
            
            # Procesar IA
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            # --- DIBUJO ULTRA FINO ---
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # 1. LÍNEAS NEGRAS (Grosor 1 = La mitad de fino)
                for p_start, p_end in CONEXIONES_TENIS:
                    if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                        pt1 = (int(lm[p_start].x * new_w), int(lm[p_start].y * new_h))
                        pt2 = (int(lm[p_end].x * new_w), int(lm[p_end].y * new_h))
                        # thickness=1 (Mínimo posible)
                        cv2.line(frame, pt1, pt2, (0, 0, 0), 1, cv2.LINE_AA)

                # 2. PUNTOS ROJOS (Radio 2 = Muy pequeños)
                for i in PUNTOS_CLAVE:
                    p = lm[i]
                    if p.visibility > 0.5:
                        center = (int(p.x*new_w), int(p.y*new_h))
                        cv2.circle(frame, center, 2, (0, 0, 255), -1, cv2.LINE_AA)

            st.image(frame, channels="BGR", use_container_width=True)
            
        else:
            st.warning("No se pudo mostrar este cuadro. Intenta mover el slider un poco.")
    else:
        st.error("El video parece estar dañado o tiene un formato (codec) que el servidor no reconoce.")
    
    cap.release()
