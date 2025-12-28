import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# Configuración de la página para móviles y escritorio
st.set_page_config(page_title="Tenis Lab Pro", layout="wide")
st.title("🎾 Tenis Lab: Análisis Biomecánico")

# --- BARRA LATERAL ---
st.sidebar.title("Configuración")
uploaded_file = st.sidebar.file_uploader("Sube tu video de tenis", type=['mp4', 'mov', 'avi'])
run = st.sidebar.checkbox('Ejecutar Video (Play/Pause)', value=True)
mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])

# Definición de conexiones para el esqueleto de 13 puntos
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Brazos y hombros
    (11, 23), (12, 24), (23, 24),                   # Tronco
    (23, 25), (25, 27), (24, 26), (26, 28)          # Piernas
]

PUNTOS_CONTROL = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

if mano_dominante == "Derecha":
    IDX_MUÑECA, IDX_CADERA = 16, 24
else: 
    IDX_MUÑECA, IDX_CADERA = 15, 23

# Inicializar MediaPipe Pose con manejo de errores para la nube
@st.cache_resource
def load_pose_model():
    return mp.solutions.pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = load_pose_model()

if uploaded_file is not None:
    # Guardar video temporalmente
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_video = st.empty() 

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Ajustar tamaño para que el procesamiento sea rápido en la nube
        h_orig, w_orig = frame.shape[:2]
        ancho_proc = 640
        alto_proc = int((h_orig / w_orig) * ancho_proc)
        frame = cv2.resize(frame, (ancho_proc, alto_proc))
        h, w = alto_proc, ancho_proc

        # Convertir a RGB para MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # 1. DIBUJAR LÍNEAS NEGRAS
            for connection in CONEXIONES_TENIS:
                p1_idx, p2_idx = connection
                p1, p2 = lm[p1_idx], lm[p2_idx]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2) 

            # 2. DIBUJAR PUNTOS ROJOS (Radio 3 para que se vea en el celu)
            for idx in PUNTOS_CONTROL:
                punto = lm[idx]
                if punto.visibility > 0.5:
                    px, py = int(punto.x * w), int(punto.y * h)
                    color = (255, 255, 255) if idx == 0 else (0, 0, 255)
                    cv2.circle(frame, (px, py), 3, color, -1)

            # 3. PLANO DEL CUERPO
            cadera_x = int(lm[IDX_CADERA].x * w)
            muñeca_x = int(lm[IDX_MUÑECA].x * w)
            cv2.line(frame, (cadera_x, 0), (cadera_x, h), (255, 255, 255), 1)
            
            dist_px = int(muñeca_x - cadera_x if mano_dominante == "Derecha" else cadera_x - muñeca_x)
            cv2.putText(frame, f"Dist. Plano: {dist_px}px", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostrar en Streamlit
        st_video.image(frame, channels="BGR", use_container_width=True)

    cap.release()
else:
    st.info("Sube un video desde tu galería para empezar el análisis.")
