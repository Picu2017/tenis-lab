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

st.markdown("##### 🎾 Tenis Lab: Puntos Completos")

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

# --- CONEXIONES (Líneas del Esqueleto) ---
CONEXIONES_TENIS = [
    # Torso
    (11, 12), (23, 24), (11, 23), (12, 24),
    # Brazos
    (11, 13), (13, 15), (12, 14), (14, 16), 
    # Piernas
    (23, 25), (25, 27), (24, 26), (26, 28),
    # Pies
    (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32)
]

# --- ÍNDICES PARA LA FLECHA DE FUERZA ---
WRIST_IDX = 16
INDEX_IDX = 20

# --- FUNCIÓN VECTOR MAESTRO (Flecha Verde) ---
def dibujar_vector_maestro(frame, landmarks, w, h):
    try:
        if landmarks[WRIST_IDX].visibility < 0.5 or landmarks[INDEX_IDX].visibility < 0.5:
            return frame

        p_wrist = np.array([landmarks[WRIST_IDX].x * w, landmarks[WRIST_IDX].y * h])
        p_index = np.array([landmarks[INDEX_IDX].x * w, landmarks[INDEX_IDX].y * h])
        
        vec_hand = p_index - p_wrist
        dist_hand = np.linalg.norm(vec_hand)
        
        if dist_hand < 1e-6: return frame
            
        dir_hand = vec_hand / dist_hand
        arrow_length = dist_hand * 3.0 
        
        arrow_start = p_index 
        arrow_end = arrow_start + (dir_hand * arrow_length)
        
        start_pt = tuple(arrow_start.astype(int))
        end_pt = tuple(arrow_end.astype(int))

        # Círculo Base y Flecha
        cv2.circle(frame, start_pt, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.arrowedLine(frame, start_pt, end_pt, (50, 255, 50), 4, cv2.LINE_AA, tipLength=0.2)
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
                    
                    # 1. DIBUJAR LÍNEAS DEL ESQUELETO (Negras finas)
                    for p_start, p_end in CONEXIONES_TENIS:
                        if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                            pt1 = (int(lm[p_start].x * w), int(lm[p_start].y * h))
                            pt2 = (int(lm[p_end].x * w), int(lm[p_end].y * h))
                            cv2.line(frame, pt1, pt2, (0, 0, 0), 2, cv2.LINE_AA)
                    
                    # 2. DIBUJAR TODOS LOS PUNTOS ROJOS (0 al 32)
                    # MediaPipe tiene 33 puntos en total (0-32)
                    for i in range(33):
                        p = lm[i]
                        # Umbral de visibilidad para no dibujar puntos fantasma
                        if p.visibility > 0.5:
                            center = (int(p.x*w), int(p.y*h))
                            # ROJO (BGR: 0, 0, 255), Radio Pequeño (3)
                            cv2.circle(frame, center, 3, (0, 0, 255), -1, cv2.LINE_AA)

                    # 3. DIBUJAR FLECHA VERDE (Encima de todo)
                    frame = dibujar_vector_maestro(frame, lm, w, h)

                st.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning("Fin del video.")
    cap.release()
