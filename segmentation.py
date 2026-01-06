"""
Módulo de segmentación de documentos.

Divide la imagen en regiones (Texto, Tablas, Imágenes) y segmenta líneas y caracteres.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import os

class DocumentSegmenter:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.east_model = None
        self.east_path = os.path.join("models", "frozen_east_text_detection.pb")

    def segment_page(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Segmenta la página completa en regiones.
        Retorna diccionario con recortes de tablas, imágenes y líneas de texto.
        """
        # 1. Preprocesamiento básico para segmentación
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        _, binary = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Detectar Tablas y Líneas Horizontales/Verticales
        # (Simplificado: Detectar grandes contornos rectangulares)
        # Para mejorar, usamos kernels morfológicos grandes
        
        # Detectar líneas horizontales y verticales para tablas
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        
        hor_lines = cv2.erode(binary, hor_kernel, iterations=1)
        hor_lines = cv2.dilate(hor_lines, hor_kernel, iterations=1)
        
        ver_lines = cv2.erode(binary, ver_kernel, iterations=1)
        ver_lines = cv2.dilate(ver_lines, ver_kernel, iterations=1)
        
        table_mask = cv2.add(hor_lines, ver_lines)
        
        # Dilatar un poco para conectar líneas desconectadas
        table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        
        # Encontrar contornos de tablas
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        tables_rects = []
        
        text_mask = binary.copy()
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filtro: Debe ser suficientemente grande para ser una tabla
            if w > 100 and h > 50:
                tables.append(img[y:y+h, x:x+w])
                tables_rects.append((x, y, w, h))
                # Borrar tabla de la máscara de texto (poner en negro)
                cv2.rectangle(text_mask, (x, y), (x+w, y+h), 0, -1)

        cnn_boxes = []
        final_text_mask = text_mask.copy()
        if os.path.exists(self.east_path):
            if self.east_model is None:
                try:
                    self.east_model = cv2.dnn.readNet(self.east_path)
                    if self.debug:
                        print(f"[INFO] Modelo EAST cargado desde: {self.east_path}")
                except Exception as e:
                    print(f"[WARN] Error cargando EAST: {e}")
                    self.east_model = None
            if self.east_model is not None:
                cnn_boxes = self._detect_text_cnn(img)
                
                # Máscara para regiones EAST (sólidas)
                east_region_mask = np.zeros_like(binary)
                for (x, y, w, h) in cnn_boxes:
                    cv2.rectangle(east_region_mask, (x, y), (x+w, y+h), 255, -1)
                    # Actualizar máscara de exclusión (sólida)
                    cv2.rectangle(final_text_mask, (x, y), (x+w, y+h), 255, -1)
                
                # Filtrar máscara de píxeles: Conservar solo píxeles dentro de regiones EAST
                text_mask = cv2.bitwise_and(text_mask, east_region_mask)
                
                # Añadir las cajas de tabla de nuevo como negras (por seguridad)
                for (x, y, w, h) in tables_rects:
                    cv2.rectangle(final_text_mask, (x, y), (x+w, y+h), 0, -1)
                    cv2.rectangle(text_mask, (x, y), (x+w, y+h), 0, -1)

        kernel_img = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        img_candidates = cv2.dilate(binary, kernel_img, iterations=2)
        mask_non_text = cv2.bitwise_and(img_candidates, cv2.bitwise_not(final_text_mask))
        contours, _ = cv2.findContours(mask_non_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        images = []
        codes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 3000:
                continue
            roi = binary[y:y+h, x:x+w]
            density = cv2.countNonZero(roi) / max(area, 1)
            sub_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_small = sum(1 for c in sub_contours if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] < 500)
            if density > 0.5 and num_small < 5:
                images.append(img[y:y+h, x:x+w])
                cv2.rectangle(final_text_mask, (x, y), (x+w, y+h), 0, -1)
                cv2.rectangle(text_mask, (x, y), (x+w, y+h), 0, -1)
        
        # 4. Extraer Líneas de Texto
        text_lines = self._extract_text_lines(text_mask)
        
        return {
            "tables": tables,
            "tables_rects": tables_rects,
            "images": images,
            "codes": codes,
            "text_lines": text_lines,
            "binary_image": binary # Útil para debug o extracción fina
        }

    def _extract_text_lines(self, binary_text: np.ndarray) -> List[np.ndarray]:
        """
        Extrae líneas de texto usando proyección horizontal o contornos dilatados.
        """
        # Dilatar horizontalmente para conectar letras en palabras y palabras en líneas
        # Aumentamos el kernel para asegurar que palabras separadas se unan en una sola línea
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        dilated = cv2.dilate(binary_text, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        # Filtrar ruido muy pequeño (puntos aislados)
        # Ajustado: h > 8 es razonable para texto normal (aprox 10px altura mínima)
        bounding_boxes = [b for b in bounding_boxes if b[2] > 5 and b[3] > 8]
        
        # Ordenar por coordenada Y (arriba a abajo)
        bounding_boxes.sort(key=lambda b: b[1])
        
        lines = []
        for x, y, w, h in bounding_boxes:
            # Extraer la región del binario original (sin dilatar)
            # Agregamos un pequeño margen
            y0, y1 = max(0, y-2), min(binary_text.shape[0], y+h+2)
            x0, x1 = max(0, x-2), min(binary_text.shape[1], x+w+2)
            line_img = binary_text[y0:y1, x0:x1]
            lines.append(line_img)
            
        return lines

    def _detect_text_cnn(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = img.shape[:2]
        newW, newH = (640, 640)
        rW = w / float(newW)
        rH = h / float(newH)
        blob = cv2.dnn.blobFromImage(img, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.east_model.setInput(blob)
        scores, geometry = self.east_model.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
        rects = []
        confidences = []
        numRows, numCols = scores.shape[2:4]
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(0, numCols):
                score = scoresData[x]
                if score < 0.5:
                    continue
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h1 = xData0[x] + xData2[x]
                w1 = xData1[x] + xData3[x]
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w1)
                startY = int(endY - h1)
                rects.append((startX, startY, endX, endY))
                confidences.append(float(score))
        indices = cv2.dnn.NMSBoxes(
            [ (x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in rects ],
            confidences, 0.5, 0.4
        )
        boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = rects[i]
                x = max(0, int(x1 * rW))
                y = max(0, int(y1 * rH))
                ww = max(1, int((x2 - x1) * rW))
                hh = max(1, int((y2 - y1) * rH))
                boxes.append((x, y, ww, hh))
        return boxes

    def segment_characters(self, line_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segmenta caracteres individuales de una línea de texto.
        """
        # Dilatar verticalmente un poco para asegurar conectividad vertical del caracter (ej. "i", "j")
        # Usamos (1, 3) o (2, 3) si hay caracteres muy fragmentados. (1, 3) es seguro.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated = cv2.dilate(line_img, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        chars_data = []
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        # Ordenar de izquierda a derecha
        bounding_boxes.sort(key=lambda b: b[0])
        
        debug_dir = "debug_chars"
        
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            # Filtro de ruido: Si es demasiado pequeño, lo ignoramos
            # Ajuste: Bajamos w mínimo a 2 para permitir 'i', 'l', '1' delgados.
            # Ajuste: Subimos h mínimo a 10 para evitar ruido de puntos sueltos.
            if w < 2 or h < 10: 
                continue 
            
            # Recorte del original
            char_img = line_img[y:y+h, x:x+w]
            
            # Padding cuadrado y resize
            char_img_sq = self._make_square(char_img)
            
            if self.debug and os.path.exists(debug_dir):
                import time
                timestamp = int(time.time() * 100000)
                fname = os.path.join(debug_dir, f"char_{timestamp}_{i}.png")
                cv2.imwrite(fname, char_img_sq)

            chars_data.append({
                'img': char_img_sq,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
            
        return chars_data

    def _make_square(self, img: np.ndarray, size: int = 32) -> np.ndarray:
        """
        Redimensiona la imagen a size x size manteniendo aspect ratio (padding).
        Añade un borde extra para evitar que toque los bordes (target 28x28).
        """
        h, w = img.shape
        if h == 0 or w == 0: return np.zeros((size, size), dtype=np.uint8)
        
        # Target inner size (margen de 2px)
        target_size = size - 4
        
        # Calcular escala
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Redimensionar contenido
        # Si escalamos hacia arriba (scale > 1), usar CUBIC. Si es hacia abajo, AREA.
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        # Crear imagen negra cuadrada
        square = np.zeros((size, size), dtype=np.uint8)
        
        # Centrar
        y_offset = (size - new_h) // 2
        x_offset = (size - new_w) // 2
        
        square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return square

    def extract_table_structure(self, table_bin: np.ndarray) -> List[List[np.ndarray]]:
        """
        Intenta segmentar celdas de una tabla binaria.
        Retorna lista de filas, donde cada fila es lista de imágenes de celdas.
        """
        # 1. Detectar filas (proyección horizontal)
        # Usamos un kernel muy ancho para detectar líneas horizontales divisorias
        kernel_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, table_bin.shape[1] // 2), 1))
        lines_hor = cv2.erode(table_bin, kernel_hor, iterations=1)
        lines_hor = cv2.dilate(lines_hor, kernel_hor, iterations=1)
        
        # Encontrar coordenadas Y de las líneas
        contours_hor, _ = cv2.findContours(lines_hor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        y_coords = sorted([cv2.boundingRect(c)[1] for c in contours_hor])
        
        # Si no hay líneas claras, intentamos heurística de espacios vacíos
        if len(y_coords) < 2:
            return [] # No se pudo detectar estructura
            
        rows = []
        for i in range(len(y_coords) - 1):
            y1 = y_coords[i]
            y2 = y_coords[i+1]
            if y2 - y1 < 10: continue # Fila muy fina
            
            row_slice = table_bin[y1:y2, :]
            
            # 2. Detectar columnas en esta fila (proyección vertical)
            # Invertimos lógica: buscamos espacios verticales vacíos o líneas verticales
            # Simplificación: Segmentar por contornos externos en la fila
            # (Asumiendo que las celdas tienen contenido separado)
            
            # Una mejor aproximación para tablas con bordes es usar las líneas verticales detectadas antes
            # Pero aquí haremos segmentación por "bloques de texto" dentro de la fila
            
            cell_contours, _ = cv2.findContours(row_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cell_bboxes = [cv2.boundingRect(c) for c in cell_contours]
            cell_bboxes.sort(key=lambda b: b[0])
            
            row_cells = []
            for cx, cy, cw, ch in cell_bboxes:
                if cw > 5 and ch > 5:
                    row_cells.append(row_slice[cy:cy+ch, cx:cx+cw])
            
            if row_cells:
                rows.append(row_cells)
                
        return rows
