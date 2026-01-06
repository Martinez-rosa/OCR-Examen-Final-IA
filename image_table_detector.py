"""
Detector de tablas e imágenes en documentos
"""

import cv2
import numpy as np

class ImageTableDetector:
    """Detecta tablas e imágenes dentro de documentos"""
    
    def __init__(self):
        self.min_table_area = 5000  # Píxeles mínimos para considerar tabla
        self.min_image_area = 2000  # Píxeles mínimos para considerar imagen
    
    def detect_tables(self, img):
        """
        Detecta tablas usando detección de líneas
        
        Args:
            img: Imagen en escala de grises
        Returns:
            Lista de diccionarios con tablas detectadas
        """
        tables = []
        
        # Binarizar
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detectar líneas horizontales
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detectar líneas verticales
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combinar líneas
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Dilatar para conectar líneas cercanas
        kernel = np.ones((5, 5), np.uint8)
        table_mask = cv2.dilate(table_mask, kernel, iterations=3)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.min_table_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Verificar que sea rectangular (aspecto tabla)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 10:  # No muy estrecho ni muy ancho
                    # Extraer región de tabla
                    table_img = img[y:y+h, x:x+w]
                    
                    tables.append({
                        'image': table_img,
                        'position': (x, y, w, h),
                        'area': area
                    })
        
        return tables
    
    def detect_images(self, img):
        """
        Detecta imágenes/figuras dentro del documento
        
        Args:
            img: Imagen en escala de grises
        Returns:
            Lista de diccionarios con imágenes detectadas
        """
        images = []
        
        # Binarizar
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detectar regiones densas (imágenes suelen tener más detalle)
        # Aplicar desenfoque para unir regiones cercanas
        blurred = cv2.GaussianBlur(binary, (21, 21), 0)
        _, mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        
        # Morfología para limpiar
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.min_image_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calcular densidad de píxeles negros
                roi = binary[y:y+h, x:x+w]
                density = np.sum(roi > 0) / (w * h)
                
                # Las imágenes suelen tener densidad media-alta
                if 0.1 < density < 0.9:
                    # Extraer región
                    img_region = img[y:y+h, x:x+w]
                    
                    images.append({
                        'image': img_region,
                        'position': (x, y, w, h),
                        'area': area,
                        'density': density
                    })
        
        return images