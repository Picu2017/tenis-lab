import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os
import gc

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Móvil", layout="wide") # 'wide' ayuda en modo horizontal

# CSS para ganar espacio y ajustar márgenes
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Reducimos los márgenes gigantes de Streamlit */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Título pequeño
st.markdown("##### 🎾 Tenis Lab")

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
    # Guardar temporal
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    gc.collect()

    cap = cv2.VideoCapture(tfile.name)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1) # Corrección de rotación
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        
        # --- DISEÑO RESPONSIVO (LA SOLUCIÓN) ---
        # Creamos dos columnas.
        # En CELULAR VERTICAL: Streamlit las apila (Col 1 arriba, Col 2 abajo).
        # En CELULAR HORIZONTAL: Streamlit las pone lado a lado.
        # [85, 15] significa: La columna del video ocupa el 85%, la del slider el 15% (en PC/Horizontal)
        col_video, col_slider = st.columns([85, 15])

        # Definimos el Slider PRIMERO (para tener el valor), pero lo dibujamos en la Col 2
        with col_slider:
            # Ponemos el slider. En horizontal quedará a la derecha.
            frame_index = st.slider("Frame", 0, total_frames - 1, 0)

        # Ahora procesamos y mostramos el video en la Col 1
        with col_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if ret:
                # Redimensionar (Mantener calidad decente pero ligero)
                h, w = frame.shape[:2]
                
                # Procesar IA
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                # Dibujo Fino (Negro y Rojo pequeño)
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # Líneas Negras Finas (Grosor 1)
                    for p_start, p_end in CONEXIONES_TENIS:
                        if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                            pt1 = (int(lm[p_start].x * w), int(lm[p_start].y * h))
                            pt2 = (int(lm[p_end].x * w), int(lm[p_end].y * h))
                            cv2.line(frame, pt1, pt2, (0, 0, 0), 1, cv2.LINE_AA)

                    # Puntos Rojos Pequeños (Radio 2)
                    for i in PUNTOS_CLAVE:
                        p = lm[i]
                        if p.visibility > 0.5:
                            center = (int(p.x*w), int(p.y*h))
                            cv2.circle(frame, center, 2, (0, 0, 255), -1, cv2.LINE_AA)

                # Mostrar Imagen Directa (Sin placeholders raros = Cero parpadeo)
                st.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning("No se pudo leer el frame.")

    cap.release()
