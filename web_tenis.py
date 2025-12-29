import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Pro", layout="centered")

st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tenis Lab: Análisis Completo")
st.info("ℹ️ Esta versión procesa el video primero para garantizar fluidez total.")

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

uploaded_file = st.file_uploader("Carga tu video", type=['mp4', 'mov', 'avi'])

# Conexiones Tenis
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]

if uploaded_file is not None:
    # 1. Guardar video de entrada
    tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
    tfile_in.write(uploaded_file.read())
    tfile_in.close()
    
    # 2. Configurar lectura y escritura
    cap = cv2.VideoCapture(tfile_in.name)
    
    # Obtenemos propiedades del video original
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Archivo de salida temporal
    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile_out.close() # Cerramos para que ffmpeg/opencv pueda escribir
    
    # Codec 'mp4v' es el más compatible para generar archivos simples
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tfile_out.name, fourcc, fps, (width, height))

    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0

    # --- BUCLE DE PROCESAMIENTO (SIN MOSTRAR IMAGEN) ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Actualizar barra cada 10 frames para no frenar el proceso
        if frame_count % 10 == 0:
            progreso = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progreso)
            status_text.text(f"Procesando Frame {frame_count} de {total_frames}...")

        # Procesar IA
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Dibujar
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Líneas
            for p_start, p_end in CONEXIONES_TENIS:
                if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                    pt1 = (int(lm[p_start].x * width), int(lm[p_start].y * height))
                    pt2 = (int(lm[p_end].x * width), int(lm[p_end].y * height))
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
            # Puntos
            for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                p = lm[i]
                if p.visibility > 0.5:
                    cv2.circle(frame, (int(p.x*width), int(p.y*height)), 5, (0, 0, 255), -1)

        # Escribir frame en el video de salida
        out.write(frame)

    # Cerrar todo
    cap.release()
    out.release()
    
    progress_bar.progress(1.0)
    status_text.success("¡Análisis completado!")

    # --- MOSTRAR VIDEO FINAL ---
    # Leemos el archivo generado y lo mostramos con el player nativo
    st.video(tfile_out.name)
    
    # Limpieza (opcional)
    try:
        os.unlink(tfile_in.name)
        # No borramos tfile_out inmediatamente para que Streamlit pueda servirlo
    except:
        pass
