import cv2
import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_foot_position

class SpeedAndDistanceEstimator():
   
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24
        self.min_distance_threshold = 0.1  # Reducido para menos procesamiento
        self.max_speed_threshold = 50.0
       
    def add_speed_and_distance_2_tracks(self, tracks):
        total_distance = {}
        
        for object_name, object_tracks in tracks.items():
            if object_name in ['ball', 'referee', 'referees']:
                continue
                
            number_of_frames = len(object_tracks)
            if number_of_frames == 0:
                continue
                
            total_distance[object_name] = {}
            
            # Procesar por ventanas SIN superposición (más eficiente)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
                
                # Solo procesar tracks que existen en ambos frames
                if frame_num >= number_of_frames or last_frame >= number_of_frames:
                    continue
                    
                current_frame_tracks = object_tracks[frame_num]
                last_frame_tracks = object_tracks[last_frame]
                
                # Intersección de track_ids para evitar iteraciones innecesarias
                common_track_ids = set(current_frame_tracks.keys()) & set(last_frame_tracks.keys())
                
                for track_id in common_track_ids:
                    # Inicializar distancia total una sola vez
                    if track_id not in total_distance[object_name]:
                        total_distance[object_name][track_id] = 0
                    
                    # Obtener posiciones directamente
                    start_pos = current_frame_tracks[track_id].get('position_transformed')
                    end_pos = last_frame_tracks[track_id].get('position_transformed')
                    
                    if start_pos is None or end_pos is None:
                        continue
                    
                    # Calcular distancia y velocidad una sola vez
                    distance_covered = measure_distance(start_pos, end_pos)
                    
                    # Filtro de distancia mínima
                    if distance_covered < self.min_distance_threshold:
                        speed_km = 0.0
                        distance_covered = 0.0
                    else:
                        time_elapsed = (last_frame - frame_num) / self.frame_rate
                        if time_elapsed > 0:
                            speed_ms = distance_covered / time_elapsed
                            speed_km = speed_ms * 3.6
                            # Filtro de velocidad máxima
                            if speed_km > self.max_speed_threshold:
                                speed_km = 0.0
                                distance_covered = 0.0
                        else:
                            speed_km = 0.0
                            distance_covered = 0.0
                    
                    # Actualizar distancia total
                    total_distance[object_name][track_id] += distance_covered
                    
                    # Asignar a todos los frames del batch de una vez
                    for batch_frame in range(frame_num, last_frame):
                        if batch_frame < number_of_frames and track_id in object_tracks[batch_frame]:
                            object_tracks[batch_frame][track_id]['speed'] = speed_km
                            object_tracks[batch_frame][track_id]['distance'] = total_distance[object_name][track_id]
    
    def draw_speed_and_distance(self, frames, tracks):
        # Procesar frames in-place para ahorrar memoria
        for frame_num, frame in enumerate(frames):
            if frame is None:
                continue
                
            for object_name, object_tracks in tracks.items():
                if object_name in ['ball', 'referee', 'referees']:
                    continue
                    
                if frame_num >= len(object_tracks):
                    continue
                    
                frame_tracks = object_tracks[frame_num]
                
                for track_id, track_info in frame_tracks.items():
                    vel = track_info.get('speed', 0)
                    dist = track_info.get('distance', 0)
                    
                    # Solo dibujar si hay valores significativos
                    if vel < 1.0 and dist < 1.0:
                        continue
                        
                    bbox = track_info.get('bbox')
                    if bbox is None:
                        continue
                        
                    try:
                        position = get_foot_position(bbox)
                        x, y = int(position[0]), int(position[1]) + 40
                        
                        # Verificar que las coordenadas estén dentro del frame
                        if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
                            continue
                        
                        # Color basado en velocidad
                        if vel < 5:
                            color = (0, 255, 0)  # Verde
                        elif vel < 15:
                            color = (0, 255, 255)  # Amarillo
                        else:
                            color = (0, 0, 255)  # Rojo
                        
                        # Dibujar directamente en el frame (sin crear copias)
                        cv2.putText(frame, f"{vel:.1f}km/h", (x, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Verificar bounds para segunda línea
                        if y + 15 < frame.shape[0]:
                            cv2.putText(frame, f"{dist:.1f}m", (x, y + 15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                    except Exception:
                        # Ignorar errores de dibujo silenciosamente
                        continue
        
        # Retornar la misma lista (modificada in-place)
        return frames