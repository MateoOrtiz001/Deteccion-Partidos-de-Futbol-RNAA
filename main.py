# Importar módulos de utilidades personalizadas
from utils import read_video, save_video
from trackers import Tracker
import cv2  


def main():
    """
    Función principal del sistema de detección y seguimiento de fútbol.
    
    Procesa un video de fútbol completo detectando y rastreando jugadores,
    árbitros y la pelota. Genera un video de salida con anotaciones visuales
    y guarda una imagen recortada de un jugador como ejemplo.
    """
    # ===== CARGA DEL VIDEO DE ENTRADA =====
    print("Cargando video de entrada...")
    video_frames = read_video("input_videos/08fd33_4.mp4")
    print(f"Video cargado exitosamente: {len(video_frames)} frames")

    # ===== INICIALIZACIÓN DEL SISTEMA DE TRACKING =====
    print("Inicializando detector YOLO y tracker...")
    tracker = Tracker("model/best.pt")
    
    # Ejecutar detección y seguimiento con cache para optimizar
    print("Ejecutando detección y seguimiento de objetos...")
    tracks = tracker.object_tracks(
        video_frames, 
        read_from_stub=True,           # Usar cache si está disponible
        stub_path="stubs/track_stubs.pkl"  # Archivo de cache
    )
    print("Tracking completado exitosamente")

    # ===== GENERACIÓN DE IMAGEN RECORTADA =====
    print("Generando imagen recortada de jugador...")
    save_cropped_player_image(video_frames, tracks)

    # ===== GENERACIÓN DEL VIDEO ANOTADO =====
    print("Dibujando anotaciones en los frames...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # ===== GUARDADO DEL VIDEO FINAL =====
    print("Guardando video procesado...")
    save_video(output_video_frames, "output_videos/08fd33_4.mp4")
    print("Procesamiento completado exitosamente!")


def save_cropped_player_image(video_frames, tracks):
    """
    Extrae y guarda una imagen recortada del primer jugador detectado.
    
    Toma el bounding box del primer jugador en el primer frame
    y recorta esa región para guardarla como imagen independiente.
    Útil para análisis individual de jugadores o entrenamiento de modelos.
    
    Args:
        video_frames (list): Lista de frames del video
        tracks (dict): Datos de tracking con información de bounding boxes
    """
    try:
        # Obtener datos del primer frame
        first_frame_players = tracks["players"][0]
        
        # Verificar si hay jugadores detectados
        if not first_frame_players:
            print("No se detectaron jugadores en el primer frame")
            return
            
        # Tomar el primer jugador disponible
        track_id, player_data = next(iter(first_frame_players.items()))
        player_bbox = player_data["bbox"]
        first_frame = video_frames[0]

        # Extraer coordenadas del bounding box
        x1, y1, x2, y2 = map(int, player_bbox)
        
        # Recortar región del jugador desde el frame
        cropped_player_image = first_frame[y1:y2, x1:x2]
        
        # Guardar imagen recortada
        output_path = "output_images/cropped_image.jpg"
        cv2.imwrite(output_path, cropped_player_image)
        
        print(f"Imagen del jugador guardada en: {output_path}")
        print(f"Track ID del jugador: {track_id}")
        print(f"Dimensiones de la imagen: {cropped_player_image.shape}")
        
    except Exception as error:
        print(f"Error al generar imagen recortada: {error}")


if __name__ == "__main__":
    """
    Punto de entrada del programa.
    
    Se ejecuta solo cuando el script se llama directamente,
    no cuando se importa como módulo.
    """
    main()