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

st.markdown("##### 🎾 Tenis Lab: Análisis Vectorial")

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

# --- MEMORIA ---
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]
PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Índices Necesarios (Brazo Derecho)
ELBOW_IDX = 14
WRIST_IDX = 16
INDEX_IDX = 20

# --- CÁLCULO VECTORIAL (NUEVO ENFOQUE) ---
def dibujar_vector_indice(frame, landmarks, w, h):
    try:
        # 1. Obtener coordenadas Codo, Muñeca, Índice
        p_elbow = np.array([landmarks[ELBOW_IDX].x * w, landmarks[ELBOW_IDX].y * h])
        p_wrist = np.array([landmarks[WRIST_IDX].x * w, landmarks[WRIST_IDX].y * h])
        p_index = np.array([landmarks[INDEX_IDX].x * w, landmarks[INDEX_IDX].y * h])
        
        # Validar visibilidad de los tres puntos
        if (landmarks[ELBOW_IDX].visibility < 0.5 or 
            landmarks[WRIST_IDX].visibility < 0.5 or 
            landmarks[INDEX_IDX].visibility < 0.5):
            return frame

        # 2. DIBUJAR LÍNEA DE ALINEACIÓN (Codo -> Muñeca -> Índice)
        # Color: Violeta azulado (BGR)
        line_color = (255, 100, 50) 
        thickness_line = 3
        
        # Convertir a tuplas de enteros para OpenCV
        pt_elbow = tuple(p_elbow.astype(int))
        pt_wrist = tuple(p_wrist.astype(int))
        pt_index = tuple(p_index.astype(int))
        
        cv2.line(frame, pt_elbow, pt_wrist, line_color, thickness_line, cv2.LINE_AA)
        cv2.line(frame, pt_wrist, pt_index, line_color, thickness_line, cv2.LINE_AA)

        # 3. CALCULAR Y DIBUJAR FLECHA DE EXTENSIÓN
        # La dirección la marca el segmento Muñeca -> Índice
        direction_vec = p_index - p_wrist
        current_len = np.linalg.norm(direction_vec)

        if current_len < 1e-6: return frame
        direction_norm = direction_vec / current_len
        
        # Largo de la flecha: Usamos el largo del antebrazo como referencia de escala
        forearm_len = np.linalg.norm(p_wrist - p_elbow)
        extension_len = forearm_len * 1.5 # 1.5 veces el largo del antebrazo

        # Inicio y Fin de la flecha
        arrow_start = pt_index
        arrow_end_np = p_index + (direction_norm * extension_len)
        arrow_end = tuple(arrow_end_np.astype(int))

        # Flecha Vector (Verde Lima) que sale del Índice
        cv2.arrowedLine(frame, 
                        arrow_start, 
                        arrow_end, 
                        (50, 255, 50), # Verde Lima
                        4,             # Grosor
                        cv2.LINE_AA, 
                        tipLength=0.2)
                        
    except Exception as e:
        print(f"Error en cálculo vectorial: {e}")
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

        # Diseño
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
                    
                    # 1. Esqueleto Base (Negro)
                    for p_start, p_end in CONEXIONES_TENIS:
                        if lm[p_start].visibility > 0.5 and lm[p_end].visibility > 0.5:
                            pt1 = (int(lm[p_start].x * w), int(lm[p_start].y * h))
                            pt2 = (int(lm[p_end].x * w), int(lm[p_end].y * h))
                            cv2.line(frame, pt1, pt2, (0, 0, 0), 2, cv2.LINE_AA)

                    # 2. Vector de Alineación e Índice (Nuevo)
                    frame = dibujar_vector_indice(frame, lm, w, h)
                    
                    # 3. Puntos Rojos (Encima de todo)
                    for i in PUNTOS_CLAVE:
                        p = lm[i]
                        if p.visibility > 0.5:
                            center = (int(p.x*w), int(p.y*h))
                            cv2.circle(frame, center, 4, (0, 0, 255), -1, cv2.LINE_AA)

                st.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning("Fin del video.")

    cap.release()
