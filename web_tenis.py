import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

st.set_page_config(page_title="Tenis Lab Pro", layout="wide")

# Título principal
st.title("🎾 Tenis Lab: Análisis Biomecánico")

# --- INTERFAZ PRINCIPAL (VISIBLE EN EL CELULAR) ---
st.write("### 1. Sube tu video aquí")
uploaded_file = st.file_uploader("Elegí un video de tu galería", type=['mp4', 'mov', 'avi'])

# Controles en la barra lateral
run = st.sidebar.checkbox('Ejecutar Video (Play/Pause)', value=True)
mano_dominante = st.sidebar.radio("Mano Dominante", ["Derecha", "Izquierda"])

# Conexiones esqueleto
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

if mano_dominante == "Derecha":
    IDX_MUÑECA, IDX_CADERA = 16, 24
else: 
    IDX_MUÑECA, IDX_CADERA = 15, 23

# Inicializar IA
@st.cache_resource
def load_pose():
    return mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

pose = load_pose()

if uploaded_file is not None:
    # Crear archivo temporal
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() # Espacio reservado para el video

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            # Al terminar el video, reinicia automáticamente
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Redimensionar para que sea fluido en el celular (640px de ancho)
        frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        h, w = frame.shape[:2]

        # Procesar IA
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Dibujar líneas negras
            for conn in CONEXIONES_TENIS:
                p1, p2 = lm[conn[0]], lm[conn[1]]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 0), 2)

            # Puntos rojos pequeños (radio 3)
            for idx in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[idx]
                if p.visibility > 0.5:
                    color = (255, 255, 255) if idx == 0 else (0, 0, 255)
                    cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, color, -1)

            # Plano vertical (blanco)
            cadera_x = int(lm[IDX_CADERA].x * w)
            cv2.line(frame, (cadera_x, 0), (cadera_x, h), (255, 255, 255), 1)
            
            dist_px = int(int(lm[IDX_MUÑECA].x * w) - cadera_x if mano_dominante == "Derecha" else cadera_x - int(lm[IDX_MUÑECA].x * w))
            cv2.putText(frame, f"Dist. Plano: {dist_px}px", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # MOSTRAR FRAME
        st_frame.image(frame, channels="BGR", use_container_width=True)
        
        # Pequeña pausa para que el servidor de la nube "respire" y mande el siguiente cuadro
        time.sleep(0.01)

    cap.release()
else:
    st.info("Subí un video para empezar el análisis.")
