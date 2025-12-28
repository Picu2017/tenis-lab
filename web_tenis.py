import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tenis Lab Pro", layout="wide")
st.title("🎾 Tenis Lab: Análisis Biomecánico")

# --- BARRA LATERAL ---
st.sidebar.title("Configuración")
uploaded_file = st.sidebar.file_uploader("Sube tu video", type=['mp4', 'mov', 'avi'])
run = st.sidebar.checkbox('Ejecutar Video (Play/Pause)', value=True)
mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])

# Definición de conexiones (13 puntos)
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]
PUNTOS_CONTROL = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

if mano_dominante == "Derecha":
    IDX_MUÑECA, IDX_CADERA = 16, 24
else: 
    IDX_MUÑECA, IDX_CADERA = 15, 23

# --- CARGA DEL MODELO ---
@st.cache_resource
def get_pose():
    # Usamos la carga estándar pero dentro de una función con caché
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = get_pose()

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_video = st.empty() 

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h_orig, w_orig = frame.shape[:2]
        frame = cv2.resize(frame, (640, int(h_orig * 640 / w_orig)))
        h, w = frame.shape[:2]

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            for connection in CONEXIONES_TENIS:
                p1, p2 = lm[connection[0]], lm[connection[1]]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 0), 2)

            for idx in PUNTOS_CONTROL:
                punto = lm[idx]
                if punto.visibility > 0.5:
                    color = (255, 255, 255) if idx == 0 else (0, 0, 255)
                    cv2.circle(frame, (int(punto.x*w), int(punto.y*h)), 3, color, -1)

            cadera_x = int(lm[IDX_CADERA].x * w)
            muñeca_x = int(lm[IDX_MUÑECA].x * w)
            cv2.line(frame, (cadera_x, 0), (cadera_x, h), (255, 255, 255), 1)
            dist_px = int(muñeca_x - cadera_x if mano_dominante == "Derecha" else cadera_x - muñeca_x)
            cv2.putText(frame, f"Dist. Plano: {dist_px}px", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        st_video.image(frame, channels="BGR", use_container_width=True)
    cap.release()
else:
    st.info("Sube tu video para comenzar el análisis profesional.")
