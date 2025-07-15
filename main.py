# Importar librerías necesarias para el procesamiento de video y asignación de equipos
import argparse
import os
import sys
from pathlib import Path
from utils import read_video, save_video
from trackers import Tracker
import cv2  
import numpy as np
from assigner import TeamAssigner
from asignadorJugador import PlayerBallAssigner
from mov_camera import EstimadorMovimientoCam
from perspective_transformer import ViewTransformer
from info import SpeedAndDistanceEstimator


def resolve_input_path(input_path, default_dir="videos"):
    """
    Resuelve la ruta del archivo de entrada.
    Si es solo un nombre de archivo, busca en el directorio por defecto.
    
    Args:
        input_path (str): Ruta o nombre del archivo de entrada
        default_dir (str): Directorio por defecto donde buscar
    
    Returns:
        str: Ruta completa del archivo de entrada
    """
    # Si es una ruta absoluta o contiene carpetas, usarla tal como está
    if os.path.dirname(input_path) or os.path.isabs(input_path):
        return input_path
    
    # Si es solo un nombre de archivo, buscar en el directorio por defecto
    full_path = os.path.join(default_dir, input_path)
    return full_path


def resolve_model_path(model_path, default_dir="model"):
    """
    Resuelve la ruta del modelo.
    Si es solo un nombre de archivo, busca en el directorio por defecto.
    
    Args:
        model_path (str): Ruta o nombre del archivo del modelo
        default_dir (str): Directorio por defecto donde buscar
    
    Returns:
        str: Ruta completa del archivo del modelo
    """
    # Si es una ruta absoluta o contiene carpetas, usarla tal como está
    if os.path.dirname(model_path) or os.path.isabs(model_path):
        return model_path
    
    # Si es solo un nombre de archivo, buscar en el directorio por defecto
    full_path = os.path.join(default_dir, model_path)
    return full_path


def create_output_path(input_path, output_dir="output_videos"):
    """
    Crea la ruta de salida basada en el archivo de entrada.
    
    Args:
        input_path (str): Ruta del archivo de entrada
        output_dir (str): Directorio de salida
    
    Returns:
        str: Ruta completa del archivo de salida
    """
    # Crear el directorio de salida si no existe
    Path(output_dir).mkdir(exist_ok=True)
    
    # Obtener el nombre del archivo sin extensión
    input_file = Path(input_path)
    output_filename = f"{input_file.stem}_analyzed{input_file.suffix}"
    
    return os.path.join(output_dir, output_filename)


def main():
    """
    Función principal del sistema de análisis de fútbol.
    
    Procesa un video detectando jugadores, asignando equipos por color de camiseta
    y generando un video de salida con anotaciones visuales.
    """
    # ===== CONFIGURACIÓN DE ARGUMENTOS =====
    parser = argparse.ArgumentParser(
        description="Sistema de análisis de fútbol con detección de jugadores y asignación de equipos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py -i partido.mp4
  python main.py -i partido.mp4 -m custom_model.pt
  python main.py -i videos/partido.mp4 -m models/custom_model.pt -o resultados/
  python main.py -i partido.mp4 --no-cache --no-interpolation
  python main.py -i partido.mp4 --stub-dir cache_personalizado/
        """
    )
    
    # Argumentos obligatorios
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Video de entrada. Puedes usar solo el nombre (ej: video.mp4) y se buscará en videos/, o la ruta completa"
    )
    
    # Argumentos opcionales
    parser.add_argument(
        "-m", "--model",
        default="best.pt",
        help="Modelo YOLO. Puedes usar solo el nombre (ej: best.pt) y se buscará en model/, o la ruta completa (default: best.pt)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="output_videos",
        help="Directorio de salida para el video procesado (default: output_videos)"
    )
    
    parser.add_argument(
        "--stub-dir",
        default="stubs",
        help="Directorio para archivos de cache/stub (default: stubs)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Desactivar el uso de cache para el tracking"
    )
    
    parser.add_argument(
        "--no-interpolation",
        action="store_true",
        help="Desactivar la interpolación de posiciones de la pelota"
    )
    
    parser.add_argument(
        "--no-camera-movement",
        action="store_true",
        help="Desactivar la estimación de movimiento de cámara"
    )
    
    parser.add_argument(
        "--no-perspective",
        action="store_true",
        help="Desactivar la transformación de perspectiva"
    )
    
    parser.add_argument(
        "--no-speed-distance",
        action="store_true",
        help="Desactivar el cálculo de velocidad y distancia"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostrar información detallada durante el procesamiento"
    )
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # ===== RESOLUCIÓN DE RUTAS =====
    # Resolver ruta del video de entrada
    input_path = resolve_input_path(args.input)
    
    # Resolver ruta del modelo
    model_path = resolve_model_path(args.model)
    
    # ===== VALIDACIÓN DE ARGUMENTOS =====
    if not os.path.exists(input_path):
        # Si no existe en la ruta resuelta, mostrar mensaje de ayuda
        if not os.path.dirname(args.input):
            print(f"Error: No se encontró el video '{args.input}' en la carpeta 'videos/'")
            print(f"Ruta buscada: {input_path}")
            print("Sugerencias:")
            print("  1. Coloca el video en la carpeta 'videos/'")
            print("  2. O usa la ruta completa: python main.py -i ruta/completa/video.mp4")
        else:
            print(f"Error: El archivo de entrada '{input_path}' no existe.")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        # Si no existe en la ruta resuelta, mostrar mensaje de ayuda
        if not os.path.dirname(args.model):
            print(f"Error: No se encontró el modelo '{args.model}' en la carpeta 'model/'")
            print(f"Ruta buscada: {model_path}")
            print("Sugerencias:")
            print("  1. Coloca el modelo en la carpeta 'model/'")
            print("  2. O usa la ruta completa: python main.py -i video.mp4 -m ruta/completa/modelo.pt")
        else:
            print(f"Error: El modelo '{model_path}' no existe.")
        sys.exit(1)
    
    # Crear directorio de stubs si no existe
    Path(args.stub_dir).mkdir(exist_ok=True)
    
    # Crear ruta de salida
    output_path = create_output_path(input_path, args.output_dir)
    
    if args.verbose:
        print(f"Configuración:")
        print(f"  - Video de entrada: {input_path}")
        print(f"  - Modelo: {model_path}")
        print(f"  - Video de salida: {output_path}")
        print(f"  - Directorio de cache: {args.stub_dir}")
        print(f"  - Usar cache: {not args.no_cache}")
        print(f"  - Interpolación: {not args.no_interpolation}")
        print(f"  - Movimiento de cámara: {not args.no_camera_movement}")
        print(f"  - Transformación de perspectiva: {not args.no_perspective}")
        print(f"  - Velocidad y distancia: {not args.no_speed_distance}")
        print()
    
    # ===== CARGA Y PROCESAMIENTO DEL VIDEO =====
    if args.verbose:
        print("Cargando video...")
    
    # Leer video de entrada y cargar todos los frames en memoria
    video_frames = read_video(input_path)
    
    if args.verbose:
        print(f"Video cargado: {len(video_frames)} frames")
    
    # Inicializar el tracker con el modelo YOLO entrenado
    if args.verbose:
        print("Inicializando tracker...")
    
    tracker = Tracker(model_path)
    
    # Ejecutar detección y seguimiento de objetos en el video
    if args.verbose:
        print("Ejecutando detección y seguimiento...")
    
    tracks = tracker.object_tracks(
        video_frames, 
        read_from_stub=not args.no_cache, 
        stub_path=os.path.join(args.stub_dir, "track_stubs.pkl")
    )
    
    # Obteniendo posiciones de objetos
    if args.verbose:
        print("Calculando posiciones...")
    
    tracker.add_possition_to_tracks(tracks)
    
    # Estimación del movimiento de la cámara
    if not args.no_camera_movement:
        if args.verbose:
            print("Estimando movimiento de cámara...")
        
        camera_movement_estimator = EstimadorMovimientoCam(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
            video_frames, 
            readFromStub=not args.no_cache,
            stubPath=os.path.join(args.stub_dir, 'camera_movement_stub.pkl')
        )
        
        camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)
    else:
        camera_movement_per_frame = None
    
    # Transformador de perspectiva
    if not args.no_perspective:
        if args.verbose:
            print("Aplicando transformación de perspectiva...")
        
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_2_tracks(tracks)
    
    # Interpolación de la posición de la pelota
    if not args.no_interpolation:
        if args.verbose:
            print("Interpolando posiciones de la pelota...")
        
        tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    # Estimador de información (después de tener las posiciones transformadas)
    if not args.no_speed_distance:
        if args.verbose:
            print("Calculando velocidad y distancia...")
        
        speed_and_distance_estimator = SpeedAndDistanceEstimator()
        speed_and_distance_estimator.add_speed_and_distance_2_tracks(tracks)

    # ===== ASIGNACIÓN DE EQUIPOS POR COLOR =====
    if args.verbose:
        print("Asignando equipos por color...")
    
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
    if args.verbose:
        print("Asignando posesión del balón...")
    
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
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
    if args.verbose:
        print("Generando anotaciones...")
    
    # Dibujar anotaciones (elipses con colores de equipo, IDs, etc.) en todos los frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    # Dibuja el movimiento de la cámara
    if not args.no_camera_movement and camera_movement_per_frame is not None:
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    # Dibuja la información de los jugadores
    if not args.no_speed_distance:
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Guardar el video procesado con todas las anotaciones
    if args.verbose:
        print(f"Guardando video en: {output_path}")
    
    save_video(output_video_frames, output_path)
    
    if args.verbose:
        print("¡Procesamiento completado!")
    else:
        print(f"Video procesado guardado en: {output_path}")


# Punto de entrada del programa
if __name__ == "__main__":
    main()