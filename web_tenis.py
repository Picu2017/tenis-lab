import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os
import gc

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Tenis Lab Móvil", layout="wide")

# --- ESTILOS CSS (Para que se vea bien en celular) ---
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        /* Botones más grandes para facilitar el toque con el dedo */
        .stButton button {
            width: 100%;
            font-weight: bold;
            font-size: 20px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("##### 🎾 Tenis Lab: Análisis Técnico")

# --- MOTOR IA (CORREGIDO PARA MULTI-USUARIO) ---
# En lugar de usar @st.cache_resource (que comparte el modelo con todos),
# usamos session_state para dar un modelo propio a cada usuario nuevo.

if 'pose' not in st.session_state:
    mp_pose = mp.solutions.pose
    st.session_state.pose = mp_pose.Pose(
        static_image_mode=False,       # False es mejor para video
        model_complexity=1,            # 1 es buen balance velocidad/precisión
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# Asignamos el modelo de la sesión actual a una variable local para usarlo fácil
pose = st.session_state.pose

# --- MEMORIA DE POSICIÓN ---
# Controlamos en qué frame (cuadro) del video está cada usuario
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

# --- CARGA DE ARCHIVO ---
uploaded_file = st.file_uploader("Carga video de tenis", type=['mp4', 'mov', 'avi'])

# --- MAPAS DE CONEXIÓN DEL CUERPO (Esqueleto) ---
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Brazos
    (11, 23), (12, 24), (23, 24),                     # Torso
    (23, 25), (24, 26), (25, 27), (26, 28)            # Piernas
]
PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# --- LÓGICA PRINCIPAL ---
if uploaded_file is not None:
    # Guardamos el video temporalmente en el servidor
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    # Abrimos el video
    cap = cv2.VideoCapture(tfile.name)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1) # Auto-rotar si se grabó vertical
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        
        # --- FUNCIONES DE NAVEGACIÓN ---
        def siguiente_frame():
            if st.session_state.frame_index < total_frames - 1:
                st.session_state.frame_index += 1

        def anterior_frame():
            if st.session_state.frame_index > 0:
                st.session_state.frame_index -= 1

        # --- DISEÑO (Video a la izquierda, Controles a la derecha) ---
        col_video, col_controls = st.columns([80, 20])

        with col_controls:
            # 1. Botones de Navegación Frame a Frame
            c_prev, c_next = st.columns(2)
            with c_prev:
                st.button("◀", on_click=anterior_frame)
            with c_next:
                st.button("▶", on_click=siguiente_frame)
            
            # 2. Barra de Progreso (Slider)
            st.slider("Línea de tiempo", 0, total_frames - 1, key='frame_index', label_visibility="collapsed")
            
            # Info del frame actual
            st.write(f"Cuadro: {st.session_state.frame_index}/{total_frames}")

        with col_video:
            # Vamos al frame específico que el usuario seleccionó
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
            ret, frame = cap.read()
            
            if ret:
                h, w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- PROCESAMIENTO IA ---
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # Dibujar líneas del esqueleto (Negras)
                    for p_start, p_end in CONEXIONES_TENIS:
                        # Solo dibujamos si la IA está segura de ver esos puntos
                        if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                            pt1 = (int(lm[p_start].x * w), int(lm[p_start].y * h))
                            pt2 = (int(lm[p_end].x * w), int(lm[p_end].y * h))
                            cv2.line(frame, pt1, pt2, (0, 0, 0), 2, cv2.LINE_AA)

                    # Dibujar articulaciones (Puntos Rojos)
                    for i in PUNTOS_CLAVE:
                        p = lm[i]
                        if p.visibility > 0.5:
                            center = (int(p.x*w), int(p.y*h))
                            cv2.circle(frame, center, 5, (0, 0, 255), -1, cv2.LINE_AA)

                # Mostrar el resultado final
                st.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning("No se pudo leer el cuadro del video.")

    cap.release()
    # Limpieza de memoria manual para evitar sobrecarga en servidores compartidos
    gc.collect()

else:
    # Mensaje de bienvenida si no hay video cargado
    st.info("👆 Carga un video para comenzar el análisis biomecánico.")
