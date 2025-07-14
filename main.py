
# Importar librerías necesarias para el procesamiento de video y asignación de equipos
from utils import read_video, save_video
from trackers import Tracker
import cv2  
import numpy as np
from team_assigner import TeamAssigner


def main():
    """
    Función principal del sistema de análisis de fútbol.
    
    Procesa un video detectando jugadores, asignando equipos por color de camiseta
    y generando un video de salida con anotaciones visuales.
    """
    # ===== CARGA Y PROCESAMIENTO DEL VIDEO =====
    # Leer video de entrada y cargar todos los frames en memoria
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Inicializar el tracker con el modelo YOLO entrenado
    tracker = Tracker("model/best.pt")
    
    # Ejecutar detección y seguimiento de objetos en el video
    # read_from_stub=True permite usar cache si existe para acelerar el proceso
    tracks = tracker.object_tracks(
        video_frames, 
        read_from_stub=True, 
        stub_path="stubs/track_stubs.pkl"
    )

    # ===== ASIGNACIÓN DE EQUIPOS POR COLOR =====
    # Inicializar el asignador de equipos basado en colores de camisetas
    team_assigner = TeamAssigner()
    
    # Analizar el primer frame para determinar los colores de los dos equipos
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # Asignar equipo a cada jugador en todos los frames
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Determinar el equipo del jugador basándose en el color de su camiseta
            team = team_assigner.get_player_team(
                video_frames[frame_num],   
                track['bbox'],
                player_id
            )
            
            # Almacenar información del equipo en los datos de tracking
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # ===== GENERACIÓN DEL VIDEO DE SALIDA =====
    # Dibujar anotaciones (elipses con colores de equipo, IDs, etc.) en todos los frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Guardar el video procesado con todas las anotaciones
    save_video(output_video_frames, "output_videos/08fd33_4.mp4") 


# Punto de entrada del programa
if __name__ == "__main__":
    main()