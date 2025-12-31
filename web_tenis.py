import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os
import gc

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tenis Lab Pro", layout="wide")

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
    # Usamos modelo complejo 2 para mejor precisión en manos si es posible
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2, 
        min_detection_confidence=0.6, # Subimos un poco la confianza
        min_tracking_confidence=0.6
    )

pose = cargar_modelo()

# --- MEMORIA ---
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

# --- DATOS DEL ESQUELETO ---
CONEXIONES_TENIS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
]
PUNTOS_CLAVE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Índices de la mano para calcular el vector
# Usaremos muñeca (16), índice (20) y meñique (18) para mano DERECHA
# Para izquierda sería 15, 19, 17. (Por ahora lo hacemos para derecha por defecto)
WRIST_IDX = 16
INDEX_IDX = 20
PINKY_IDX = 18

# --- NUEVA FUNCIÓN: ESTIMACIÓN DEL VECTOR DE RAQUETA ---
def estimar_vector_raqueta(frame, landmarks, w, h):
    """
    Usa la geometría de la mano para estimar dónde está el centro
    de la raqueta y hacia dónde apunta su cara (vector normal).
    """
    # 1. Obtener coordenadas clave de la mano (en pixeles)
    try:
        p_wrist = np.array([landmarks[WRIST_IDX].x * w, landmarks[WRIST_IDX].y * h])
        p_index = np.array([landmarks[INDEX_IDX].x * w, landmarks[INDEX_IDX].y * h])
        p_pinky = np.array([landmarks[PINKY_IDX].x * w, landmarks[PINKY_IDX].y * h])
    except:
        # Si la IA no detecta bien la mano, no hacemos nada
        return frame

    # Verificar visibilidad para no dibujar cosas locas si la mano está oculta
    if (landmarks[WRIST_IDX].visibility < 0.6 or 
        landmarks[INDEX_IDX].visibility < 0.6 or 
        landmarks[PINKY_IDX].visibility < 0.6):
        return frame

    # --- A. CALCULAR EL CENTRO ESTIMADO DE LA RAQUETA ---
    
    # Vector dirección de la mano (promedio entre índice y meñique desde la muñeca)
    hand_center = (p_index + p_pinky) / 2.0
    hand_direction_vec = hand_center - p_wrist
    
    # Normalizamos el vector de dirección (longitud 1)
    hand_dir_norm = hand_direction_vec / (np.linalg.norm(hand_direction_vec) + 1e-6)
    
    # Estimamos el largo de la raqueta relativo al tamaño de la mano en pantalla.
    # Un factor de 3.5 veces la distancia muñeca-nudillos suele funcionar.
    dist_wrist_knuckles = np.linalg.norm(hand_direction_vec)
    racket_length_pixels = dist_wrist_knuckles * 3.5
    
    # El centro de la raqueta está siguiendo esa dirección
    racket_center = p_wrist + (hand_dir_norm * racket_length_pixels)


    # --- B. CALCULAR EL VECTOR NORMAL (LA FLECHA) ---

    # Usamos el vector que va del meñique al índice para definir el plano ancho de la mano
    palm_width_vec = p_index - p_pinky
    
    # En 2D, el vector perpendicular a [dx, dy] es [-dy, dx].
    # Esto nos da un vector que "sale" perpendicularmente de la línea de los nudillos.
    normal_vec_2d = np.array([-palm_width_vec[1], palm_width_vec[0]])
    
    # Normalizamos (longitud 1)
    normal_vec_norm = normal_vec_2d / (np.linalg.norm(normal_vec_2d) + 1e-6)
    
    # Longitud de la flecha: La mitad del tamaño estimado de la raqueta
    arrow_length = racket_length_pixels / 2.0
    
    # Punto final de la flecha
    arrow_end = racket_center + (normal_vec_norm * arrow_length)

    # --- DIBUJAR ---
    
    # 1. Centro de la raqueta (Círculo Amarillo)
    cv2.circle(frame, (int(racket_center[0]), int(racket_center[1])), 8, (0, 255, 255), -1, cv2.LINE_AA)
    
    # 2. La Flecha Vector (Color Cian/Celeste)
    # Usamos arrowedLine para que dibuje la punta automáticamente
    cv2.arrowedLine(frame, 
                    (int(racket_center[0]), int(racket_center[1])), 
                    (int(arrow_end[0]), int(arrow_end[1])), 
                    (255, 255, 0), # Color Cian (BGR)
                    3,             # Grosor
                    cv2.LINE_AA,
                    tipLength=0.3) # Tamaño de la punta

    return frame

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Carga video (Jugador Diestro preferible)", type=['mp4', 'mov', 'avi'])

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
                    
                    # 1. Dibujo Esqueleto (Negro grueso y Rojo grande)
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
                            
                    # 2. NUEVO: Dibujo del Vector de Raqueta
                    # (Asume jugador diestro por los índices usados WRIST_IDX=16)
                    frame = estimar_vector_raqueta(frame, lm, w, h)

                st.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning("Fin del video.")

    cap.release()
