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
        # Inicializar matcher específico para manuscrito
        self.matcher_manuscrito = TemplateMatcher(os.path.join(templates_dir, "manuscrito"))
        
        # Inicializar modelos especializados
        self.cnn_digital = CNNRecognizer(templates_dir=os.path.join(templates_dir, "digital"), model_path="models/model_digital.h5")
        self.cnn_manuscrito = CNNRecognizer(templates_dir=os.path.join(templates_dir, "manuscrito"), model_path="models/model_manuscrito.h5")
        
    def process_image(self, image_input: Any, text_type: str = "digital") -> Dict[str, Any]:
        import time
        import datetime
        
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
        active_recognizer = None
        
        if text_type == "manuscrito":
            # El usuario solicitó explícitamente usar MATCHING con templates de manuscrito
            print("[INFO] Usando Template Matching para manuscrito (solicitud de usuario)")
            active_recognizer = self.matcher_manuscrito
        else:
            # Para digital preferimos CNN
            active_recognizer = self.cnn_digital

        regions = self.segmenter.segment_page(img)

        full_text = ""
        text_lines_content: List[str] = []
        for line_img in regions['text_lines']:
            chars_data = self.segmenter.segment_characters(line_img)
            
            line_text = active_recognizer.recognize_line(chars_data)
            
            if line_text.strip(): # Solo añadir si no es vacío
                full_text += line_text + "\n"
                text_lines_content.append(line_text)

        processed_tables: List[List[List[str]]] = []
        # TABLAS DESACTIVADAS POR SOLICITUD DEL USUARIO
        
        # === GUARDAR RESULTADOS EN DISCO ===
        # Crear carpeta outputs si no existe
        output_base = "output"
        if not os.path.exists(output_base):
            os.makedirs(output_base)
            
        # Carpeta timestamp para esta ejecución
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_base, f"run_{timestamp}")
        os.makedirs(run_dir)
        
        # 1. Guardar Texto
        txt_path = os.path.join(run_dir, "result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        # 2. Guardar Imágenes extraídas (si las hay)
        extracted_images_paths = []
        for i, img_roi in enumerate(regions.get("images", [])):
            if img_roi is not None and img_roi.size > 0:
                img_name = f"image_{i}.png"
                img_path = os.path.join(run_dir, img_name)
                cv2.imwrite(img_path, img_roi)
                extracted_images_paths.append(img_path)

        print(f"[INFO] Resultados guardados en: {run_dir}")

        return {
            "text": full_text,
            "lines": text_lines_content,
            "tables_data": processed_tables,
            "tables_imgs": regions["tables"],
            "images": regions["images"],
            "codes": regions.get("codes", []),
            "output_dir": run_dir
        }

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        try:
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            return cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        except Exception:
            return None
