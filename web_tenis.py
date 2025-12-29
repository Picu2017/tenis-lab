import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Pro", layout="centered")

# Estilos CSS para ocultar elementos innecesarios
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tenis Lab: Analizador de Golpe")
st.info("💡 Mueve la barra deslizante para analizar el movimiento cuadro por cuadro.")

# --- MOTOR IA ---
@st.cache_resource
def cargar_modelo():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=False, # False es mejor para video
        model_complexity=1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = cargar_modelo()

# --- CARGA DE ARCHIVO ---
uploaded_file = st.file_uploader("Carga tu video", type=['mp4', 'mov', 'avi'])

CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]

if uploaded_file is not None:
    # 1. Guardar video en disco (Usamos os.getcwd para asegurar persistencia)
    temp_path = os.path.join(os.getcwd(), "temp_video_input.mp4")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # 2. Abrir video para leer propiedades
    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        # --- INTERFAZ DE CONTROL ---
        # Slider para elegir el frame exacto
        frame_index = st.slider("Mueve la barra para ver el golpe", 0, total_frames - 1, 0)
        
        # Ir al frame seleccionado
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if ret:
            # --- PROCESAMIENTO DEL FRAME ELEGIDO ---
            
            # Redimensionar para que entre bien en pantalla
            # Mantenemos proporción pero limitamos ancho a 700px
            h_orig, w_orig = frame.shape[:2]
            aspect = h_orig / w_orig
            new_w = 700
            new_h = int(new_w * aspect)
            frame = cv2.resize(frame, (new_w, new_h))
            
            # Procesar IA
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            # --- DIBUJADO ---
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # Líneas blancas
                for p_start, p_end in CONEXIONES_TENIS:
                    if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                        pt1 = (int(lm[p_start].x * new_w), int(lm[p_start].y * new_h))
                        pt2 = (int(lm[p_end].x * new_w), int(lm[p_end].y * new_h))
                        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

                # Puntos rojos
                for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                    p = lm[i]
                    if p.visibility > 0.5:
                        cv2.circle(frame, (int(p.x*new_w), int(p.y*new_h)), 5, (0, 0, 255), -1)

                # --- LÓGICA DE TENIS (Opcional: Ver impacto) ---
                # Si quieres medir distancias en el frame congelado
                # idx_cadera = 24 # Derecha
                # idx_muneca = 16
                # ... lógica de distancia aquí ...

            # Mostrar imagen final estática
            st.image(frame, channels="BGR", use_container_width=True)
            st.caption(f"Analizando Frame: {frame_index}")
            
        else:
            st.error("No se pudo leer el frame seleccionado.")
            
    cap.release()
