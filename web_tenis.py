import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import gc
import os
import urllib.request

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tenis Lab Móvil", layout="wide")

st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        .block-container {padding: 1rem 0.5rem !important;}
        .stButton button {
            width: 100%;
            height: 60px;
            font-weight: bold;
            font-size: 24px;
        }
    </style>
""", unsafe_allow_html=True)

st.write("### 🎾 Tenis Lab: Análisis Móvil")

# --- FUNCIÓN "SALVAVIDAS" PARA EL MODELO ---
def conseguir_modelo():
    """Busca el archivo localmente, y si no está, lo baja a temporales."""
    nombre_archivo = 'pose_landmark_lite.tflite'
    
    # 1. Intentar ruta directa (si está en GitHub)
    if os.path.exists(nombre_archivo):
        return nombre_archivo
    
    # 2. Intentar ruta absoluta (a veces necesario en la nube)
    ruta_absoluta = os.path.join(os.path.dirname(__file__), nombre_archivo)
    if os.path.exists(ruta_absoluta):
        return ruta_absoluta

    # 3. PLAN B: Descargar a carpeta temporal (Donde SI tenemos permiso)
    url_modelo = "https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite"
    path_temp = os.path.join(tempfile.gettempdir(), nombre_archivo)
    
    if not os.path.exists(path_temp):
        try:
            # st.info("Descargando modelo IA por primera vez...") # Descomentar para debug
            urllib.request.urlretrieve(url_modelo, path_temp)
        except Exception as e:
            st.error(f"Fallo al descargar el modelo: {e}")
            return None
            
    return path_temp

# --- INICIALIZAR MOTOR IA ---
if 'landmarker' not in st.session_state:
    try:
        model_path = conseguir_modelo()
        
        if model_path:
            with open(model_path, 'rb') as f:
                model_data = f.read()

            base_options = python.BaseOptions(model_asset_buffer=model_data)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            st.session_state.landmarker = vision.PoseLandmarker.create_from_options(options)
        else:
            st.error("❌ No se pudo encontrar ni descargar el modelo de IA.")
            st.session_state.landmarker = None

    except Exception as e:
        st.error(f"Error crítico iniciando IA: {e}")
        st.session_state.landmarker = None

# --- MEMORIA DE NAVEGACIÓN ---
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

# --- CARGA DE VIDEO ---
uploaded_file = st.file_uploader("Toca para elegir video", type=['mp4', 'mov', 'avi'])

CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]
PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

if uploaded_file is not None and st.session_state.landmarker:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        col_video, col_controls = st.columns([1, 100])
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("◀"):
                st.session_state.frame_index = max(0, st.session_state.frame_index - 1)
        with c3:
            if st.button("▶"):
                st.session_state.frame_index = min(total_frames - 1, st.session_state.frame_index + 1)
        with c2:
            st.slider("Timeline", 0, total_frames - 1, key='frame_index', label_visibility="collapsed")
            st.markdown(f"<p style='text-align: center;'>Cuadro: {st.session_state.frame_index}</p>", unsafe_allow_html=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
        ret, frame = cap.read()
        
        if ret:
            h, w = frame.shape[:2]
            ANCHO_MAXIMO = 640 
            if w > ANCHO_MAXIMO:
                factor = ANCHO_MAXIMO / w
                nuevo_alto = int(h * factor)
                frame = cv2.resize(frame, (ANCHO_MAXIMO, nuevo_alto))
                h, w = frame.shape[:2]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            detection_result = st.session_state.landmarker.detect(mp_image)
            
            if detection_result.pose_landmarks:
                lm = detection_result.pose_landmarks[0]
                for p_start, p_end in CONEXIONES_TENIS:
                    if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                        pt1 = (int(lm[p_start].x * w), int(lm[p_start].y * h))
                        pt2 = (int(lm[p_end].x * w), int(lm[p_end].y * h))
                        cv2.line(frame, pt1, pt2, (0, 0, 0), 2, cv2.LINE_AA)

                for i in PUNTOS_CLAVE:
                    p = lm[i]
                    if p.visibility > 0.5:
                        center = (int(p.x*w), int(p.y*h))
                        cv2.circle(frame, center, 4, (0, 0, 255), -1, cv2.LINE_AA)

            st.image(frame, channels="BGR", use_container_width=True)
            
        else:
            st.warning("Error leyendo el cuadro.")
    cap.release()
    gc.collect()
elif uploaded_file is None:
    st.info("👆 Sube un video para empezar")
