import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Pro", layout="centered")

# Estilos CSS para ocultar elementos molestos
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tenis Lab: Análisis de Golpe")

# --- MOTOR IA ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Carga tu video de tenis", type=['mp4', 'mov', 'avi'])
col1, col2 = st.columns(2)
with col1:
    mano = st.radio("Mano hábil", ["Derecha", "Izquierda"])
with col2:
    run = st.checkbox('Analizar', value=True)

# Puntos específicos para tenis (evitamos dibujar todo el cuerpo si no es necesario)
# Conexiones personalizadas
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Brazos y hombros
    (11, 23), (12, 24), (23, 24),                     # Tronco
    (23, 25), (24, 26), (25, 27), (26, 28)            # Piernas
]

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() # El contenedor del video
    
    # Índices para calcular impacto
    idx_muneca = 16 if mano == "Derecha" else 15
    idx_cadera = 24 if mano == "Derecha" else 23

    frame_count = 0

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # --- TRUCO DE FLUIDEZ: FRAME SKIPPING ---
        # Solo procesamos 1 de cada 3 frames. 
        # Esto hace que el video se vea fluido en la web.
        if frame_count % 3 != 0:
            continue

        # 1. Redimensionar (Mantiene la app rápida)
        frame = cv2.resize(frame, (640, 360))
        h, w = frame.shape[:2]
        
        # 2. IA
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # 3. Dibujo Personalizado (Puntos chicos como pediste antes)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Dibujar conexiones (Líneas blancas finas)
            for p_start, p_end in CONEXIONES_TENIS:
                if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                    pt1 = (int(lm[p_start].x * w), int(lm[p_start].y * h))
                    pt2 = (int(lm[p_end].x * w), int(lm[p_end].y * h))
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 1)

            # Dibujar Articulaciones (Puntos rojos pequeños)
            for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cx, cy = int(p.x * w), int(p.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # 4. Lógica de Tenis (Línea de Impacto)
            try:
                # Coordenada X de la cadera (referencia)
                cx_cadera = int(lm[idx_cadera].x * w)
                cx_muneca = int(lm[idx_muneca].x * w)

                # Línea vertical en la cadera (Referencia de impacto)
                cv2.line(frame, (cx_cadera, 0), (cx_cadera, h), (0, 255, 255), 1)
                
                # Texto de distancia
                dist = cx_muneca - cx_cadera if mano == "Derecha" else cx_cadera - cx_muneca
                color_texto = (0, 255, 0) if dist > 0 else (0, 165, 255) # Verde o Naranja
                cv2.putText(frame, f"Impacto: {dist}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)
            except:
                pass

        # Mostrar frame final
        st_frame.image(frame, channels="BGR", use_container_width=True)

    cap.release()
