import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os
import gc

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Móvil", layout="wide")

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
        .stButton button {
            width: 100%;
            font-weight: bold;
            font-size: 24px;
            height: 60px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("##### 🎾 Tenis Lab: Análisis Vectorial")

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

# --- MEMORIA ---
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]
PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Índices Mano Derecha
WRIST_IDX = 16
PINKY_IDX = 18 # Usamos Meñique como reemplazo del Anular (la IA no tiene anular)
INDEX_IDX = 20

# --- CÁLCULO VECTORIAL ---
def estimar_vector_raqueta(frame, landmarks, w, h):
    try:
        # 1. Coordenadas del Triángulo de la Mano
        p_wrist = np.array([landmarks[WRIST_IDX].x * w, landmarks[WRIST_IDX].y * h])
        p_index = np.array([landmarks[INDEX_IDX].x * w, landmarks[INDEX_IDX].y * h])
        p_pinky = np.array([landmarks[PINKY_IDX].x * w, landmarks[PINKY_IDX].y * h])
        
        # Validar visibilidad
        if (landmarks[WRIST_IDX].visibility < 0.5 or 
            landmarks[INDEX_IDX].visibility < 0.5 or 
            landmarks[PINKY_IDX].visibility < 0.5):
            return frame

        # 2. Calcular el Eje del Mango (Centro de la Raqueta)
        # Promedio de los nudillos para encontrar el centro de la mano
        knuckles_center = (p_index + p_pinky) / 2.0
        
        # Vector Dirección: Desde Muñeca hacia Nudillos
        handle_vector = knuckles_center - p_wrist
        handle_len = np.linalg.norm(handle_vector)
        
        if handle_len < 1e-6: return frame
            
        handle_dir = handle_vector / handle_len
        
        # Estimamos el centro de la raqueta (aprox 3.5 veces la distancia muñeca-nudillo)
        racket_len_px = handle_len * 3.5
        racket_center = p_wrist + (handle_dir * racket_len_px)

        # 3. Calcular la Normal (Cara de la Raqueta)
        # Vector que une los nudillos (define el plano ancho de la mano)
        knuckle_vector = p_index - p_pinky
        
        # Vector Perpendicular (Normal) a los nudillos
        # Esto nos dice hacia dónde mira la palma
        normal_vec = np.array([-knuckle_vector[1], knuckle_vector[0]])
        
        # Normalizar
        normal_len = np.linalg.norm(normal_vec)
        if normal_len < 1e-6: return frame
        normal_dir = normal_vec / normal_len
        
        # 4. Extender la Flecha (El doble de largo que antes)
        # Antes era racket_len_px / 2.0, ahora usamos racket_len_px * 1.0 (Largo completo)
        arrow_length = racket_len_px 
        arrow_end = racket_center + (normal_dir * arrow_length)

        # --- DIBUJADO ---
        # Centro (Amarillo)
        cv2.circle(frame, (int(racket_center[0]), int(racket_center[1])), 8, (0, 255, 255), -1, cv2.LINE_AA)
        
        # Flecha Vector (Verde Lima Brillante)
        cv2.arrowedLine(frame, 
                        (int(racket_center[0]), int(racket_center[1])), 
                        (int(arrow_end[0]), int(arrow_end[1])), 
                        (50, 255, 50), # Verde Lima
                        4,             # Grosor extra
                        cv2.LINE_AA, 
                        tipLength=0.2)
                        
    except:
        pass
    return frame

# --- CARGA ---
uploaded_file = st.file_uploader("Carga video", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    gc.collect()

    cap = cv2.VideoCapture(tfile.name)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        
        def siguiente_frame():
            if st.session_state.frame_index < total_frames - 1:
                st.session_state.frame_index += 1

        def anterior_frame():
            if st.session_state.frame_index > 0:
                st.session_state.frame_index -= 1

        # Diseño
        col_video, col_controls = st.columns([80, 20])

        with col_controls:
            c_prev, c_next = st.columns(2)
            with c_prev: st.button("◀", on_click=anterior_frame)
            with c_next: st.button("▶", on_click=siguiente_frame)
            
            st.slider("Timeline", 0, total_frames - 1, key='frame_index', label_visibility="collapsed")
            st.write(f"Frame: {st.session_state.frame_index}/{total_frames}")

        with col_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
            ret, frame = cap.read()
            
            if ret:
                h, w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # Esqueleto Fuerte
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
                    
                    # Vector Raqueta
                    frame = estimar_vector_raqueta(frame, lm, w, h)

                st.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning("Fin del video.")

    cap.release()
