"""
Sistema OCR Unificado.
Clase de alto nivel que integra segmentación y reconocimiento.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from segmentation import DocumentSegmenter
from recognition import TemplateMatcher, CNNRecognizer

class OCRSystem:
    def __init__(self, templates_dir: str = "templates", debug: bool = False):
        self.debug = debug
        self.templates_dir = templates_dir
        
        # Inicializar componentes
        # Aseguramos que existan las plantillas
        if not os.path.exists(templates_dir):
            print(f"[WARN] Directorio de plantillas '{templates_dir}' no encontrado.")
            
        self.segmenter = DocumentSegmenter(debug=debug)
        self.matcher = TemplateMatcher(templates_dir)
        
        # Inicializar modelos especializados
        self.cnn_digital = CNNRecognizer(templates_dir=os.path.join(templates_dir, "digital"), model_path="models/model_digital.h5")
        self.cnn_manuscrito = CNNRecognizer(templates_dir=os.path.join(templates_dir, "manuscrito"), model_path="models/model_manuscrito.h5")
        
    def process_image(self, image_input: Any, text_type: str = "digital") -> Dict[str, Any]:
        img = None
        if isinstance(image_input, str):
            img = self._load_image(image_input)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise ValueError("Input debe ser ruta (str) o imagen (numpy array)")
            
        # Seleccionar modelo
        if text_type == "manuscrito":
            active_cnn = self.cnn_manuscrito
        else:
            active_cnn = self.cnn_digital

        regions = self.segmenter.segment_page(img)

        full_text = ""
        text_lines_content: List[str] = []
        for line_img in regions['text_lines']:
            chars_data = self.segmenter.segment_characters(line_img)
            # Usar CNN si está disponible, sino Matcher
            if getattr(active_cnn, "model", None) is not None:
                line_text = active_cnn.recognize_line(chars_data)
            else:
                line_text = self.matcher.recognize_line(chars_data)
            full_text += line_text + "\n"
            text_lines_content.append(line_text)

        processed_tables: List[List[List[str]]] = []
        if regions["tables"] and regions.get("binary_image") is not None:
            binary_full = regions.get("binary_image")
            for tbl_rect in regions.get("tables_rects", []):
                x, y, w, h = tbl_rect
                tbl_bin = binary_full[y:y+h, x:x+w]
                try:
                    rows_cells = self.segmenter.extract_table_structure(tbl_bin)
                    table_data: List[List[str]] = []
                    for row in rows_cells:
                        row_data: List[str] = []
                        for cell_img in row:
                            chars = self.segmenter.segment_characters(cell_img)
                            if getattr(active_cnn, "model", None) is not None:
                                cell_text = active_cnn.recognize_line(chars)
                            else:
                                cell_text = self.matcher.recognize_line(chars)
                            row_data.append(cell_text)
                        table_data.append(row_data)
                    processed_tables.append(table_data)
                except Exception as e:
                    if self.debug:
                        print(f"Error procesando tabla: {e}")
                    processed_tables.append([])

        return {
            "text": full_text,
            "lines": text_lines_content,
            "tables_data": processed_tables,
            "tables_imgs": regions["tables"],
            "images": regions["images"],
            "codes": regions.get("codes", [])
        }

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        try:
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            return cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        except Exception:
            return None
