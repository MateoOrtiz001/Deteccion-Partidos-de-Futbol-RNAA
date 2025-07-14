from ultralytics import YOLO
import supervision as sv
import pickle as pk
import os
import sys
import cv2
import numpy as np

# Añadir el directorio padre al path para importaciones
sys.path.append("../")
from utils.bbox_utils import get_center_of_bbox, get_bbox_width


class Tracker:
    """
    Clase principal para detectar y rastrear objetos en videos de fútbol.
    Utiliza YOLO para detección y ByteTrack para seguimiento de objetos.
    """
    
    def __init__(self, model_path):
        """
        Inicializa el tracker con el modelo YOLO y el algoritmo de seguimiento.
        
        Args:
            model_path (str): Ruta al archivo del modelo YOLO entrenado (.pt)
        """
        # Cargar modelo YOLO preentrenado para detección de objetos
        self.model = YOLO(model_path)
        
        # Inicializar tracker ByteTrack para seguimiento multi-objeto
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """
        Ejecuta detección YOLO en lotes de frames para optimizar rendimiento.
        
        Args:
            frames (list): Lista de frames de video como arrays numpy
            
        Returns:
            list: Lista de resultados de detección de YOLO
        """
        # Procesar frames en lotes para mayor eficiencia
        batch_size = 20
        detections = []
        
        for i in range(0, len(frames), batch_size):
            # Predecir objetos en el lote actual con confianza mínima del 10%
            detections_batch = self.model.predict(
                frames[i:i+batch_size], 
                conf=0.1
            )   
            detections += detections_batch
            
            # Solo procesar el primer lote por ahora (para testing)
            break
            
        return detections

    def object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Rastrea objetos a través de todos los frames del video.
        Puede cargar desde cache o procesar desde cero.
        
        Args:
            frames (list): Frames del video
            read_from_stub (bool): Si cargar datos desde cache
            stub_path (str): Ruta del archivo cache
            
        Returns:
            dict: Diccionario con tracks organizados por tipo de objeto y frame
        """
        # Intentar cargar desde cache si está disponible
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pk.load(f)
            return tracks

        # Ejecutar detecciones en todos los frames
        detections = self.detect_frames(frames)       

        # Inicializar estructura de datos para almacenar tracks
        tracks = {
            "players": [],     # Jugadores por frame
            "referees": [],    # Árbitros por frame  
            "ball": []         # Pelota por frame
        }

        for frame_num, detection in enumerate(detections):
            # Obtener mapeo de clases del modelo
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convertir detecciones YOLO al formato de supervision
            detection_supervision = sv.Detections.from_ultralytics(detection)  

            # Normalizar: convertir porteros en jugadores para tracking uniforme
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Aplicar algoritmo de seguimiento para mantener IDs consistentes
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Organizar detecciones por tipo de objeto
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Clasificar cada detección según su tipo
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # Asignar a la categoría correspondiente
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    
                elif cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                # Tratamiento especial para la pelota (solo una por frame)
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Guardar en cache si se especifica ruta
        if stub_path:
            with open(stub_path, 'wb') as f:
                pk.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Dibuja una elipse en la base del objeto detectado para visualización.
        
        Args:
            frame: Frame donde dibujar
            bbox: Coordenadas del bounding box [x1, y1, x2, y2]
            color: Color RGB para la elipse
            track_id: ID del track (opcional, para mostrar número)
            
        Returns:
            frame: Frame modificado con la elipse dibujada
        """
        # Calcular posición y dimensiones de la elipse
        y_bottom = int(bbox[3])  # Parte inferior del bbox
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Dibujar elipse en la base del objeto
        cv2.ellipse(
            frame, 
            center=(x_center, y_bottom), 
            axes=(int(width), int(0.35 * width)), 
            angle=0.0, 
            startAngle=45, 
            endAngle=235, 
            color=color, 
            thickness=2, 
            lineType=cv2.LINE_4
        )

        # Añadir etiqueta con ID del track si se proporciona
        if track_id is not None:
            self._draw_track_label(frame, x_center, y_bottom, track_id, color)
        
        return frame

    def _draw_track_label(self, frame, x_center, y_bottom, track_id, color):
        """
        Método auxiliar para dibujar la etiqueta del ID del track.
        """
        # Dimensiones del rectángulo de la etiqueta
        rectangle_width = 40
        rectangle_height = 20
        
        # Calcular posición del rectángulo
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y_bottom - rectangle_height // 2) + 15
        y2_rect = (y_bottom + rectangle_height // 2) + 15

        # Dibujar rectángulo de fondo
        cv2.rectangle(
            frame, 
            (int(x1_rect), int(y1_rect)), 
            (int(x2_rect), int(y2_rect)), 
            color, 
            cv2.FILLED
        )
        
        # Ajustar posición del texto según número de dígitos
        x_text = x1_rect + 12
        if track_id > 99:
            x_text -= 10
        
        # Dibujar número del track
        cv2.putText(
            frame, 
            f"{track_id}", 
            (int(x_text), int(y1_rect + 15)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 0),  # Texto negro
            2
        )

    def draw_triangle(self, frame, bbox, color):
        """
        Dibuja un triángulo para marcar objetos especiales como la pelota.
        
        Args:
            frame: Frame donde dibujar
            bbox: Coordenadas del bounding box
            color: Color del triángulo
            
        Returns:
            frame: Frame modificado con el triángulo
        """
        # Obtener posición para el triángulo (parte superior del bbox)
        y_top = int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)

        # Definir puntos del triángulo
        triangle_points = np.array([
            [x_center, y_top],           # Punta superior
            [x_center - 10, y_top - 20], # Esquina inferior izquierda
            [x_center + 10, y_top - 20], # Esquina inferior derecha
        ])
        
        # Dibujar triángulo relleno
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        
        # Dibujar borde negro para mejor visibilidad
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        """
        Dibuja todas las anotaciones de tracking en los frames del video.
        
        Args:
            video_frames (list): Lista de frames originales
            tracks (dict): Datos de tracking por tipo de objeto y frame
            
        Returns:
            list: Frames anotados con visualizaciones de tracking
        """
        output_video_frames = []
        
        # Procesar cada frame del video
        for frame_num, frame in enumerate(video_frames):
            # Crear copia para no modificar el original
            annotated_frame = frame.copy()

            # Obtener datos de tracking para el frame actual
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num] 
            ball_dict = tracks["ball"][frame_num]

            # Dibujar jugadores con elipses rojas (o color del equipo)
            for track_id, player in player_dict.items():
                # Usar color del equipo si está disponible, sino rojo por defecto
                color = player.get("team_color", (0, 0, 255))
                annotated_frame = self.draw_ellipse(
                    annotated_frame, 
                    player["bbox"], 
                    color, 
                    track_id
                )

            # Dibujar árbitros con elipses amarillas (sin ID)
            for _, referee in referee_dict.items():
                annotated_frame = self.draw_ellipse(
                    annotated_frame, 
                    referee["bbox"], 
                    (0, 255, 255)  # Color amarillo
                )

            # Dibujar pelota con triángulo verde
            for track_id, ball in ball_dict.items():
                annotated_frame = self.draw_triangle(
                    annotated_frame, 
                    ball["bbox"], 
                    (0, 255, 0)  # Color verde
                )

            # Añadir frame anotado a la lista de salida
            output_video_frames.append(annotated_frame)

        return output_video_frames
                          