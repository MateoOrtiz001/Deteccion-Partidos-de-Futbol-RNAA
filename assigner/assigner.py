from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Clase para asignar automáticamente equipos a jugadores basándose en el color de sus uniformes.
    
    Utiliza algoritmos de clustering (K-means) para analizar los colores dominantes
    en las camisetas de los jugadores y agruparlos en dos equipos diferentes.
    """
    
    def __init__(self):
        """
        Inicializa el asignador de equipos.
        
        Establece las estructuras de datos necesarias para almacenar
        los colores característicos de cada equipo y las asignaciones
        de jugadores individuales.
        """
        # Diccionario para almacenar colores representativos de cada equipo
        # Formato: {team_id: [R, G, B]}
        self.team_colors = {}
        
        # Cache de asignaciones por jugador para mantener consistencia
        # Formato: {player_id: team_id}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        """
        Crea y entrena un modelo de clustering K-means para segmentación de colores.
        
        Args:
            image: Imagen RGB a procesar
            
        Returns:
            KMeans: Modelo entrenado para clasificación de colores
        """
        # Convertir imagen 3D (altura, ancho, canales) a matriz 2D (píxeles, RGB)
        image_2d = image.reshape(-1, 3)

        # Configurar y entrenar modelo K-means con 2 clusters
        # Un cluster para la camiseta del jugador, otro para el fondo
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extrae el color dominante de la camiseta de un jugador específico.
        
        Args:
            frame: Frame completo del video
            bbox: Coordenadas del bounding box [x1, y1, x2, y2]
            
        Returns:
            array: Color RGB dominante de la camiseta
        """
        # Recortar región del jugador usando las coordenadas del bounding box
        player_region = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Enfocarse en la mitad superior donde típicamente está la camiseta
        # Esto evita confusión con pantalones, césped, etc.
        top_half_image = player_region[0:int(player_region.shape[0]/2), :]

        # Aplicar clustering para separar camiseta del fondo
        kmeans = self.get_clustering_model(top_half_image)

        # Obtener etiquetas de cluster para cada píxel
        labels = kmeans.labels_

        # Reorganizar etiquetas para coincidir con forma de imagen
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Identificar cluster del fondo usando las esquinas de la imagen
        # Asumimos que las esquinas generalmente contienen fondo, no jugador
        corner_clusters = [
            clustered_image[0, 0],      # Esquina superior izquierda
            clustered_image[0, -1],     # Esquina superior derecha
            clustered_image[-1, 0],     # Esquina inferior izquierda
            clustered_image[-1, -1]     # Esquina inferior derecha
        ]
        
        # El cluster más común en las esquinas es probablemente el fondo
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        
        # El otro cluster debe ser el jugador
        player_cluster = 1 - non_player_cluster

        # Obtener color representativo del cluster del jugador
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Analiza todos los jugadores detectados y determina los colores de los dos equipos.
        
        Args:
            frame: Frame del video para análisis
            player_detections: Diccionario con detecciones de jugadores
        """
        # Recopilar colores dominantes de todos los jugadores detectados
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # Aplicar clustering global para separar jugadores en dos equipos
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        # Guardar modelo para futuras clasificaciones
        self.kmeans = kmeans

        # Almacenar colores representativos de cada equipo
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determina a qué equipo pertenece un jugador específico.
        
        Args:
            frame: Frame actual del video
            player_bbox: Bounding box del jugador [x1, y1, x2, y2]
            player_id: ID único del jugador para tracking
            
        Returns:
            int: ID del equipo (1 o 2)
        """
        # Verificar cache para mantener consistencia temporal
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Extraer color dominante del jugador actual
        player_color = self.get_player_color(frame, player_bbox)

        # Clasificar color usando el modelo de equipos entrenado
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        
        # Convertir de índice (0,1) a ID de equipo (1,2)
        team_id += 1

        # Caso especial: corrección manual para jugador específico
        # Esto podría ser un portero o caso edge detectado manualmente
        if player_id == 91:
            team_id = 1

        # Guardar en cache para futuras consultas
        self.player_team_dict[player_id] = team_id

        return team_id