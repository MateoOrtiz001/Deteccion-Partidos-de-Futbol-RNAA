import cv2 

def read_video(video_path):
    """
    Lee un archivo de video y extrae todos sus frames.
    
    Esta función utiliza OpenCV para cargar un video completo en memoria
    como una lista de frames individuales. Útil para procesamiento offline
    donde se necesita acceso aleatorio a cualquier frame.
    
    Args:
        video_path (str): Ruta al archivo de video a procesar
        
    Returns:
        list: Lista de frames como arrays numpy (BGR format)
        
    Ejemplo:
        frames = read_video("input_videos/partido.mp4")
        print(f"Video cargado con {len(frames)} frames")
    """
    # Inicializar capturador de video desde archivo
    video_capture = cv2.VideoCapture(video_path)
    frames_list = []
    
    # Extraer frames uno por uno hasta el final del video
    while True:
        # Leer siguiente frame del video
        frame_exists, current_frame = video_capture.read()
        
        # Si no hay más frames, terminar el bucle
        if not frame_exists:
            break
            
        # Agregar frame válido a la lista
        frames_list.append(current_frame)
    
    
    return frames_list

def save_video(output_video_frames, output_video_path):
    """
    Guarda una lista de frames como un archivo de video.
    
    Toma una secuencia de frames y los codifica en un archivo de video
    usando el códec XVID. Mantiene las dimensiones originales de los frames
    con una velocidad de reproducción fija de 24 FPS.
    
    Args:
        output_video_frames (list): Lista de frames como arrays numpy
        output_video_path (str): Ruta donde guardar el video resultante
        
    Ejemplo:
        processed_frames = process_video(original_frames)
        save_video(processed_frames, "output_videos/resultado.mp4")
    """
    # Definir códec de compresión XVID
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Crear objeto escritor de video con dimensiones del primer frame
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    # Escribir cada frame al archivo de video
    for frame in output_video_frames:
        out.write(frame)
    
    # Finalizar escritura y liberar recursos
    out.release()