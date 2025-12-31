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

st.markdown("##### 🎾 Tenis Lab: Proyecciones")

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

if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]
PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Índices (Brazo Derecho)
ELBOW_IDX = 14
WRIST_IDX = 16
INDEX_IDX = 20

# --- FUNCION DE PROYECCIONES ---
def dibujar_proyecciones(frame, landmarks, w, h):
    try:
        # Coordenadas
        p_elbow = np.array([landmarks[ELBOW_IDX].x * w, landmarks[ELBOW_IDX].y * h])
        p_wrist = np.array([landmarks[WRIST_IDX].x * w, landmarks[WRIST_IDX].y * h])
        p_index = np.array([landmarks[INDEX_IDX].x * w, landmarks[INDEX_IDX].y * h])
        
        if (landmarks[ELBOW_IDX].visibility < 0.5 or 
            landmarks[WRIST_IDX].visibility < 0.5 or 
            landmarks[INDEX_IDX].visibility < 0.5):
            return frame

        # Referencia de largo: Usamos la distancia del antebrazo
        forearm_dist = np.linalg.norm(p_wrist - p_elbow)
        extension_len = forearm_dist * 1.2 # Un poco más largo que el antebrazo

        # --- 1. PROYECCIÓN DEL ANTEBRAZO (AZUL) ---
        # Dirección: Codo -> Muñeca
        vec_forearm = p_wrist - p_elbow
        norm_forearm = np.linalg.norm(vec_forearm)
        
        if norm_forearm > 1e-6:
            dir_forearm = vec_forearm / norm_forearm
            # Comienza en la Muñeca, sigue la línea del antebrazo
            end_forearm = p_wrist + (dir_forearm * extension_len)
            
            # Dibujamos línea punteada azul (simulada con círculos o línea delgada)
            cv2.arrowedLine(frame, 
                            tuple(p_wrist.astype(int)), 
                            tuple(end_forearm.astype(int)), 
                            (255, 200, 0), # Azul Cielo
                            2, cv2.LINE_AA, tipLength=0.1)

        # --- 2. PROYECCIÓN DE LA MANO (VERDE) ---
        # Dirección: Muñeca -> Índice
        vec_hand = p_index - p_wrist
        norm_hand = np.linalg.norm(vec_hand)
        
        if norm_hand > 1e-6:
            dir_hand = vec_hand / norm_hand
            # Comienza en el Índice, sigue la línea de la mano
            end_hand = p_index + (dir_hand * extension_len)
            
            # Dibujamos flecha verde gruesa (La dirección real del golpe)
            cv2.arrowedLine(frame, 
                            tuple(p_index.astype(int)), 
                            tuple(end_hand.astype(int)), 
                            (0, 255, 0), # Verde Lima
                            4, cv2.LINE_AA, tipLength=0.2)

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
                    
                    # Esqueleto Base
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

                    # PROYECCIONES
                    frame = dibujar_proyecciones(frame, lm, w, h)

                st.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning("Fin del video.")
    cap.release()
