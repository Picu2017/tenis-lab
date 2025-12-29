import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Móvil", layout="centered")

# Estilos para que se vea bien en celular (oculta menús molestos)
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        /* Ajuste para que el slider sea más fácil de tocar en el cel */
        .stSlider {padding-top: 2rem; padding-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tenis Lab Móvil")
st.warning("📱 **Tip para celular:** Si se desconecta, prueba subir videos más cortos o grabados en menor calidad (HD en vez de 4K).")

# --- MOTOR IA (Carga ultra rápida) ---
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

# --- CARGA DE ARCHIVO ---
uploaded_file = st.file_uploader("Elige tu video", type=['mp4', 'mov', 'avi'])

CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]

if uploaded_file is not None:
    # Barra de progreso falsa para dar feedback visual inmediato
    with st.spinner('Cargando video...'):
        
        # 1. Guardar video de forma segura
        temp_path = os.path.join(os.getcwd(), "temp_mobile.mp4")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # 2. Leer video
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            st.success(f"Video cargado: {total_frames} cuadros.")
            
            # --- CONTROL DESLIZANTE ---
            frame_index = st.slider("Desliza para buscar el impacto", 0, total_frames - 1, 0)
            
            # Ir al frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if ret:
                # Redimensionar dinámico para pantalla de celular
                h_orig, w_orig = frame.shape[:2]
                aspect = h_orig / w_orig
                # 400px de ancho es ideal para celulares verticales
                new_w = 400 
                new_h = int(new_w * aspect)
                frame = cv2.resize(frame, (new_w, new_h))
                
                # Procesar
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                # Dibujar
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # Líneas blancas más finas para que no tapen
                    for p_start, p_end in CONEXIONES_TENIS:
                        if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                            pt1 = (int(lm[p_start].x * new_w), int(lm[p_start].y * new_h))
                            pt2 = (int(lm[p_end].x * new_w), int(lm[p_end].y * new_h))
                            cv2.line(frame, pt1, pt2, (255, 255, 255), 1)

                    # Puntos rojos pequeños
                    for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                        p = lm[i]
                        if p.visibility > 0.5:
                            cv2.circle(frame, (int(p.x*new_w), int(p.y*new_h)), 3, (0, 0, 255), -1)

                # Mostrar
                st.image(frame, channels="BGR", use_container_width=True)
                
            else:
                st.error("Error al leer el cuadro.")
        
        cap.release()
