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

st.markdown("##### 🎾 Tenis Lab: Mapa de Puntos")

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

# --- PUNTOS DISPONIBLES EN EL BRAZO DERECHO ---
# Estos son los únicos que ve la IA
PUNTOS_BRAZO = {
    12: "Hombro",
    14: "Codo",
    16: "Muñeca",
    18: "Meñique",
    20: "Indice",
    22: "Pulgar"
}

def dibujar_diagnostico_mano(frame, landmarks, w, h):
    overlay = frame.copy()
    
    # 1. DIBUJAR TODOS LOS PUNTOS DISPONIBLES
    for idx, nombre in PUNTOS_BRAZO.items():
        lm = landmarks[idx]
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            
            # Dibujar punto
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1, cv2.LINE_AA) # Azul
            
            # Escribir nombre del punto
            cv2.putText(frame, str(idx), (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 2. DIBUJAR EL PLANO DE LA PALMA (TRIÁNGULO)
    # Usamos Muñeca(16), Meñique(18), Índice(20)
    try:
        if (landmarks[16].visibility > 0.5 and 
            landmarks[18].visibility > 0.5 and 
            landmarks[20].visibility > 0.5):
            
            p16 = np.array([landmarks[16].x * w, landmarks[16].y * h]).astype(int)
            p18 = np.array([landmarks[18].x * w, landmarks[18].y * h]).astype(int)
            p20 = np.array([landmarks[20].x * w, landmarks[20].y * h]).astype(int)
            
            # Dibujar Triángulo Relleno (Semi-transparente)
            triangle_cnt = np.array([p16, p20, p18])
            cv2.drawContours(overlay, [triangle_cnt], 0, (0, 255, 255), -1) # Amarillo
            
            # Mezclar para transparencia
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Dibujar bordes del triángulo
            cv2.line(frame, tuple(p16), tuple(p20), (0, 255, 0), 2)
            cv2.line(frame, tuple(p20), tuple(p18), (0, 255, 0), 2)
            cv2.line(frame, tuple(p18), tuple(p16), (0, 255, 0), 2)
            
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
                    # Solo dibujamos el diagnóstico de puntos
                    frame = dibujar_diagnostico_mano(frame, lm, w, h)

                st.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning("Fin del video.")
    cap.release()
