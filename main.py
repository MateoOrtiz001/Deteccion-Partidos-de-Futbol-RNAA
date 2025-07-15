
# Importar librerías necesarias para el procesamiento de video y asignación de equipos
from utils import read_video, save_video
from trackers import Tracker
import cv2  
import numpy as np
from assigner import TeamAssigner
from asignadorJugador import PlayerBallAssigner
from mov_camera import EstimadorMovimientoCam
from perspective_transformer import ViewTransformer
from info import SpeedAndDistanceEstimator


def main():
    """
    Función principal del sistema de análisis de fútbol.
    
    Procesa un video detectando jugadores, asignando equipos por color de camiseta
    y generando un video de salida con anotaciones visuales.
    """
    # ===== CARGA Y PROCESAMIENTO DEL VIDEO =====
    # Leer video de entrada y cargar todos los frames en memoria
    video_frames = read_video("videos/08fd33_4.mp4")

    # Inicializar el tracker con el modelo YOLO entrenado
    tracker = Tracker("model/best.pt")
    
    # Ejecutar detección y seguimiento de objetos en el video
    # read_from_stub=True permite usar cache si existe para acelerar el proceso
    tracks = tracker.object_tracks(
        video_frames, 
        read_from_stub=True, 
        stub_path="stubs/track_stubs.pkl"
    )
    
    # Obteniendo posiciones de objetos
    tracker.add_possition_to_tracks(tracks)
    
    # Estimación del movimiento de la cámara
    camera_movement_estimator = EstimadorMovimientoCam(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, readFromStub=True,stubPath='stubs/camera_movement_stub.pk1')
    
    camera_movement_estimator.add_adjust_position_to_tracks(tracks,camera_movement_per_frame)
    
    # Transformador de perspectiva
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_2_tracks(tracks)
    
    # Interpolación de la posición de la pelota
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    # Estimador de información (después de tener las posiciones transformadas)
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_2_tracks(tracks)

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
            
    # Asigna el jugador que tiene el balón
    
    player_assigner =  PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                # Decide qué valor inicial poner, por ejemplo 0 o None
                team_ball_control.append(0)
    team_ball_control = np.array(team_ball_control)

    # ===== GENERACIÓN DEL VIDEO DE SALIDA =====
    # Dibujar anotaciones (elipses con colores de equipo, IDs, etc.) en todos los frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    # Dibuja el movimiento de la cámara
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    # Dibuja la información de los jugadores
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Guardar el video procesado con todas las anotaciones
    save_video(output_video_frames, "output_videos/08fd33_4.mp4") 


# Punto de entrada del programa
if __name__ == "__main__":
    main()
