import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import gc

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
    st.session_state.pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,            # CAMBIO CLAVE: 0 es más rápido para móviles (Lite)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = st.session_state.pose

# --- MEMORIA DE NAVEGACIÓN ---
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

# --- CARGA DE VIDEO ---
# accept_multiple_files=False ayuda en móviles
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
        # Usamos columnas para poner los botones abajo o arriba del video según prefieras
        col_video, col_controls = st.columns([1, 100]) # Truco para centrar en móvil
        
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
            # --- MAGIA DE OPTIMIZACIÓN ---
            # Si el video es gigante (ej: 4K del celular), lo achicamos
            h, w = frame.shape[:2]
            ANCHO_MAXIMO = 640 # Resolución segura para la nube
            
            if w > ANCHO_MAXIMO:
                factor = ANCHO_MAXIMO / w
                nuevo_alto = int(h * factor)
                frame = cv2.resize(frame, (ANCHO_MAXIMO, nuevo_alto))
                h, w = frame.shape[:2] # Actualizamos dimensiones

            # Procesar IA
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # Dibujar esqueleto
                for p_start, p_end in CONEXIONES_TENIS:
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

else:
    st.info("👆 Sube un video para empezar")
