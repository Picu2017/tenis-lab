import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import mediapipe as mp

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tenis Lab Pro", layout="centered")

# --- MOTOR IA (INICIALIZACIÓN ROBUSTA) ---
# Usamos @st.cache_resource para cargar el modelo una sola vez y no saturar la memoria
@st.cache_resource
def load_pose_engine():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,       # 1 es más preciso que 0 (Lite), ideal para deportes
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

try:
    pose_engine = load_pose_engine()
except Exception as e:
    st.error(f"Error fatal al cargar el modelo: {e}")
    st.stop()

# --- INTERFAZ ---
st.title("🎾 Tenis Lab: Análisis Biomecánico")
st.markdown("---")

uploaded_file = st.file_uploader("Subí tu video aquí", type=['mp4', 'mov', 'avi'])
col1, col2 = st.columns(2)
with col1:
    mano_dominante = st.radio("Mano Dominante", ["Derecha", "Izquierda"])
with col2:
    run = st.checkbox('Analizar / Pausar', value=True)

# Conexiones anatómicas relevantes para tenis
CONEXIONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), 
    (24, 26), (26, 28)
]

# Puntos clave a dibujar
PUNTOS_CLAVE = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

if uploaded_file is not None:
    # Gestión de archivo temporal compatible con Linux/Cloud
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    frame_window = st.empty()
    
    # Índices según mano dominante
    idx_m = 16 if mano_dominante == "Derecha" else 15 # Muñeca
    idx_c = 24 if mano_dominante == "Derecha" else 23 # Cadera

    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            # Reiniciar video en loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. Redimensionar manteniendo relación de aspecto
        max_height = 640 # Altura fija para consistencia
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (int(max_height * aspect_ratio), max_height))
        h, w = frame.shape[:2]

        # 2. Procesamiento IA
        # Convertir a RGB para MediaPipe (MediaPipe usa RGB, OpenCV usa BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_engine.process(frame_rgb)

        # 3. Dibujado (Sobre el frame original BGR)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Dibujar conexiones (Esqueleto)
            for s, e in CONEXIONES:
                p1, p2 = lm[s], lm[e]
                # Umbral de visibilidad para evitar ruido
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(frame, 
                             (int(p1.x*w), int(p1.y*h)), 
                             (int(p2.x*w), int(p2.y*h)), 
                             (200, 200, 200), 2) # Color gris claro, grosor 2
            
            # Dibujar puntos (Articulaciones) - MÁS CHICOS (Radio 3)
            for i in PUNTOS_CLAVE:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, (0, 0, 255), -1)

            # 4. Lógica de Tenis (Plano de impacto)
            try:
                # Coordenadas X de cadera y muñeca
                cx = int(lm[idx_c].x * w)
                mx = int(lm[idx_m].x * w)
                
                # Línea vertical de referencia (Cadera)
                cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 1)
                
                # Cálculo de distancia (Impacto adelante o atrás)
                # Si es derecha: Positivo = impacto adelante de la cadera
                dist = mx - cx if mano_dominante == "Derecha" else cx - mx
                
                color_texto = (0, 255, 0) if dist > 0 else (0, 0, 255) # Verde si es bueno, Rojo si es malo
                cv2.putText(frame, f"Impacto: {int(dist)}px", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)
            except IndexError:
                pass

        # Mostrar en Streamlit
        frame_window.image(frame, channels="BGR", use_container_width=True)
        
        # Pequeña pausa para no saturar CPU
        time.sleep(0.01)

    cap.release()
    # Limpieza del archivo temporal
    try:
        os.unlink(tfile.name)
    except:
        pass
