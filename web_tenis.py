import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os
import gc

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Móvil", layout="centered")

# CSS para maximizar espacio
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 1rem !important;
        }
        .stSlider {
            padding-top: 1rem;
            padding-bottom: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 🎾 Tenis Lab")

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

# --- GESTIÓN DE MEMORIA VISUAL (Anti-Parpadeo) ---
if 'last_frame_bgr' not in st.session_state:
    st.session_state['last_frame_bgr'] = None

# --- CARGA DE ARCHIVO ---
uploaded_file = st.file_uploader("Carga video", type=['mp4', 'mov', 'avi'])

CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]
PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    gc.collect()

    cap = cv2.VideoCapture(tfile.name)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        # 1. Creamos el espacio del video ARRIBA
        video_placeholder = st.empty()
        
        # TRUCO: Si ya tenemos una imagen en memoria, la mostramos YA MISMO.
        # Esto evita que la pantalla se quede en blanco mientras calculamos el nuevo frame.
        if st.session_state['last_frame_bgr'] is not None:
            video_placeholder.image(st.session_state['last_frame_bgr'], channels="BGR", use_container_width=True)

        # 2. Ponemos el slider ABAJO
        frame_index = st.slider("Buscar golpe", 0, total_frames - 1, 0)
        
        # 3. Calculamos el NUEVO frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if ret:
            # Redimensionar
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

            # 4. ACTUALIZAMOS EL VIDEO y guardamos en memoria
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
            st.session_state['last_frame_bgr'] = frame # Guardar en memoria para la próxima vuelta
            
            st.caption(f"Frame: {frame_index}")
            
        else:
            st.warning("Mueve el slider.")
    
    cap.release()
