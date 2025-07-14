def get_center_of_bbox(bbox):
    """
    Calcula el punto central de un bounding box.
    
    Toma las coordenadas de un rectángulo delimitador y retorna
    el punto central como coordenadas enteras. Útil para posicionar
    anotaciones o calcular distancias entre objetos.
    
    Args:
        bbox (list/tuple): Coordenadas del bounding box en formato [x1, y1, x2, y2]
                          donde (x1,y1) es esquina superior izquierda
                          y (x2,y2) es esquina inferior derecha
                          
    Returns:
        tuple: (center_x, center_y) como números enteros
        
    Ejemplo:
        bbox = [10, 20, 50, 80]  # x1=10, y1=20, x2=50, y2=80
        center = get_center_of_bbox(bbox)  # Resultado: (30, 50)
    """
    # Extraer coordenadas del bounding box
    x1, y1, x2, y2 = bbox
    
    # Calcular punto central y convertir a enteros
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    return center_x, center_y


def get_bbox_width(bbox):
    """
    Calcula el ancho de un bounding box.
    
    Determina la dimensión horizontal del rectángulo delimitador
    restando la coordenada x izquierda de la coordenada x derecha.
    
    Args:
        bbox (list/tuple): Coordenadas del bounding box en formato [x1, y1, x2, y2]
                          
    Returns:
        float/int: Ancho del bounding box en píxeles
        
    Ejemplo:
        bbox = [10, 20, 50, 80]  # Rectángulo de 40 píxeles de ancho
        width = get_bbox_width(bbox)  # Resultado: 40
    """
    # El ancho es la diferencia entre coordenadas x
    width = bbox[2] - bbox[0]  # x2 - x1
    
    return width





    