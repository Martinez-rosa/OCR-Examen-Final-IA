"""
Módulo de reconocimiento de caracteres mediante Template Matching (Correlación).

Compara cada carácter segmentado contra una base de datos de plantillas (prototipos)
y devuelve el carácter con mayor similitud. No utiliza Machine Learning ni entrenamiento.
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    tf = None

class TemplateMatcher:
    def __init__(self, templates_dir: str):
        self.templates_dir = templates_dir
        self.templates: Dict[str, List[np.ndarray]] = {}
        self.template_ratios: Dict[str, List[float]] = {}
        self.load_templates()

    def load_templates(self):
        """
        Carga las plantillas recursivamente.
        Usa el nombre de la carpeta contenedora inmediata como la etiqueta del carácter.
        """
        if not os.path.exists(self.templates_dir):
            print(f"[WARN] Directorio de plantillas no encontrado: {self.templates_dir}")
            return

        print("[INFO] Cargando plantillas...", end=" ")
        count = 0
        # Recorrer recursivamente para soportar estructuras anidadas (ej. templates/mayusculas/A)
        for root, dirs, files in os.walk(self.templates_dir):
            # El nombre de la carpeta actual se usa como etiqueta (ej. "A")
            label_name = os.path.basename(root)
            
            # Evitar usar el directorio raíz de templates como etiqueta
            if os.path.abspath(root) == os.path.abspath(self.templates_dir):
                continue
            
            # === FILTRO DE SÍMBOLOS ===
            # Solo permitimos alfanuméricos y símbolos esenciales para email/web
            # Ignoramos comas, puntos y comas, paréntesis, etc. que causan falsos positivos
            allowed_specials = ["at", "underscore", "hyphen"] 
            ignored_labels = ["dot", "colon", "semicolon", "comma", "slash", "backslash", "quote", "doublequote"]

            is_valid = False
            if label_name in ignored_labels:
                is_valid = False
            elif label_name.isalnum():
                is_valid = True
            elif label_name in allowed_specials:
                is_valid = True
            
            # Excepciones: Si es una carpeta contenedora como "mayusculas", la ignoramos
            if label_name in ["mayusculas", "minusculas", "numeros", "simbolos"]:
                is_valid = False

            if not is_valid:
                continue

            # Mapeo de nombres de carpeta a caracteres reales
            if label_name == "at": label = "@"
            elif label_name == "dot": label = "."
            elif label_name == "underscore": label = "_"
            elif label_name == "hyphen": label = "-"
            else: label = label_name
            
            if label not in self.templates:
                self.templates[label] = []

            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, filename)
                    
                    try:
                        # Leer imagen
                        stream = open(img_path, "rb")
                        bytes = bytearray(stream.read())
                        numpyarray = np.asarray(bytes, dtype=np.uint8)
                        img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
                        stream.close()
                    except Exception:
                        continue
                    
                    if img is None: continue
                    
                    # Procesamiento idéntico al input: Binarizar Otsu + Invertir si es necesario
                    # Las plantillas generadas son letras negras sobre blanco (normalmente)
                    # o blancas sobre negro. Debemos estandarizar a blanco sobre negro.
                    
                    # Binarizar
                    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Chequeo de polaridad: Queremos Texto=255 (Blanco), Fondo=0 (Negro)
                    # Contamos píxeles blancos
                    n_white = cv2.countNonZero(bin_img)
                    n_pixels = bin_img.shape[0] * bin_img.shape[1]
                    
                    # Si hay más blanco que negro, es fondo blanco -> Invertir
                    if n_white > n_pixels // 2:
                        bin_img = cv2.bitwise_not(bin_img)
                    
                    # Recortar al contenido (ROI)
                    coords = cv2.findNonZero(bin_img)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        bin_img = bin_img[y:y+h, x:x+w]
                    else:
                        # Si no hay contenido (todo negro), saltar
                        continue
                    
                    # Calcular aspect ratio original
                    ratio = w / h if h > 0 else 1.0

                    # Redimensionar (Forzando ajuste)
                    resized = self._resize_template(bin_img)
                    
                    if label not in self.templates:
                        self.templates[label] = []
                    self.templates[label].append(resized)
                    
                    if label not in self.template_ratios:
                        self.template_ratios[label] = []
                    self.template_ratios[label].append(ratio)

                    count += 1
        
        print(f"OK. {count} plantillas cargadas para {len(self.templates)} clases.")

    def _resize_template(self, img: np.ndarray, size: int = 32) -> np.ndarray:
        """Redimensiona manteniendo aspect ratio y rellenando con negro (padding).
        Añade un borde extra para evitar plantillas sólidas."""
        h, w = img.shape[:2]
        if h == 0 or w == 0: return np.zeros((size, size), dtype=np.uint8)
        
        # Target inner size (margen de 2px)
        target_size = size - 4
        
        # Calcular escala
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Asegurar dimensiones mínimas de 1px
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        
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

    def recognize_char(self, char_img: np.ndarray, original_wh: Tuple[int, int] = None) -> Tuple[str, float]:
        """
        Compara la imagen del caracter con todas las plantillas.
        Devuelve (mejor_etiqueta, confianza).
        Args:
            char_img: Imagen del caracter (normalizada a 32x32)
            original_wh: Tupla (width, height) original antes de normalizar, para comparar aspect ratio.
        """
        best_score = -1.0
        best_label = "?"
        
        # Optimización: Si la imagen está vacía
        if cv2.countNonZero(char_img) == 0:
            return "", 0.0

        # Iterar sobre todas las clases
        for label, templates_list in self.templates.items():
            for idx, templ in enumerate(templates_list):
                # Coincidencia de plantillas
                # Usamos correlación normalizada
                res = cv2.matchTemplate(char_img, templ, cv2.TM_CCOEFF_NORMED)
                score = res[0][0]
                
                # Penalización por aspect ratio
                if original_wh and original_wh[1] > 0:
                    input_ratio = original_wh[0] / original_wh[1]
                    
                    if label in self.template_ratios and idx < len(self.template_ratios[label]):
                        tmpl_ratio = self.template_ratios[label][idx]
                        diff = abs(input_ratio - tmpl_ratio)
                        
                        # Penalización adaptativa
                        # Si la diferencia es muy grande (ej. 1.0 vs 0.3), penalizar fuerte
                        if diff > 0.4:
                            score -= diff * 0.5
                        elif diff > 0.15:
                            score -= diff * 0.25
                
                if score > best_score:
                    best_score = score
                    best_label = label
        
        # === LÓGICA DE DECISIÓN AVANZADA ===
        
        # 1. Penalización para símbolos si la confianza es baja
        # Los símbolos (puntos, guiones) suelen dar falsos positivos con ruido
        if best_label in [".", "-", "_", "@"]:
            if best_score < 0.75: # Exigimos mucha confianza para un símbolo
                # Intentar buscar la segunda mejor opción que sea ALFANUMÉRICA
                # (Simplificación: por ahora solo bajamos la confianza o descartamos)
                # O mejor: si es símbolo y score bajo, retornamos vacío o ?
                pass 
        
        # 2. Umbral mínimo global
        if best_score < 0.45:
            return "?", best_score
            
        return best_label, best_score

    def recognize_line(self, chars_data: List[Dict]) -> str:
        """
        Convierte una lista de datos de caracteres en un string.
        Inserta espacios basándose en la distancia horizontal.
        """
        if not chars_data:
            return ""
            
        text = ""
        # Calcular ancho promedio de caracteres para estimar el espacio
        widths = [c['w'] for c in chars_data]
        avg_width = np.mean(widths) if widths else 10
        
        # Umbral para espacio: 40% del ancho promedio
        space_threshold = avg_width * 0.4
        
        for i, char_data in enumerate(chars_data):
            char_img = char_data['img']
            
            # Espacios
            if i > 0:
                prev_char = chars_data[i-1]
                gap = char_data['x'] - (prev_char['x'] + prev_char['w'])
                if gap > space_threshold:
                    text += " "
            
            orig_w = char_data.get('w', 0)
            orig_h = char_data.get('h', 0)
            label, score = self.recognize_char(char_img, original_wh=(orig_w, orig_h))
            
            # Solo añadir si no es basura
            if label != "?":
                text += label
            
        return self._refine_text_context(text)

    def _refine_text_context(self, text: str) -> str:
        """
        Aplica heurísticas simples para corregir confusiones comunes (0 vs O, 1 vs I, etc.)
        basándose en el contexto de la palabra (si es mayormente letras o números).
        """
        words = text.split(" ")
        refined_words = []
        
        for word in words:
            # Contar dígitos y letras
            n_digits = sum(c.isdigit() for c in word)
            n_alpha = sum(c.isalpha() for c in word)
            length = len(word)
            
            if length == 0:
                refined_words.append("")
                continue
            
            new_word = list(word)
            
            # Caso 1: Palabra mayormente alfabética (ej. "HOLA") -> corregir números que parecen letras
            # Umbral: Más del 50% son letras y tiene algún dígito
            if n_alpha > n_digits:
                for i, char in enumerate(new_word):
                    if char == '0': new_word[i] = 'o'
                    elif char == '6': new_word[i] = 'e' # e vs 6 es común en este dataset
                    elif char == '1': new_word[i] = 'l' if i > 0 else 'I' # l intermedia, I inicial
                    elif char == '5': new_word[i] = 's'
                    elif char == '8': new_word[i] = 'B'
                    elif char == '2': new_word[i] = 'z'
            
            # Caso 2: Palabra mayormente numérica (ej. "2023") -> corregir letras que parecen números
            elif n_digits > n_alpha:
                for i, char in enumerate(new_word):
                    if char.lower() == 'o': new_word[i] = '0'
                    elif char == 'l' or char == 'I': new_word[i] = '1'
                    elif char.lower() == 's': new_word[i] = '5'
                    elif char == 'B': new_word[i] = '8'
                    elif char.lower() == 'z': new_word[i] = '2'
            
            refined_words.append("".join(new_word))
            
        return " ".join(refined_words)

import pickle

class CNNRecognizer:
    def __init__(self, templates_dir: str, model_path: str = "models/cnn_ocr.h5"):
        self.templates_dir = templates_dir
        self.model_path = model_path
        self.model = None
        self.classes: List[str] = []
        if tf is None:
            return
        
        # Intentar cargar modelo existente
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                
                # Intentar cargar mapeo .pkl (prioridad)
                mapping_path = self.model_path.replace(".h5", "") + "_mapping.pkl"
                # Compatibilidad con nombre solicitado por usuario: char_mapping_digital.pkl si el modelo es model_digital.h5
                # Vamos a inferir el nombre del mapping basado en el nombre del modelo
                # Si model_path es "models/model_digital.h5", mapping debería ser "models/char_mapping_digital.pkl"
                
                base_name = os.path.basename(self.model_path)
                if "digital" in base_name:
                    mapping_name = "char_mapping_digital.pkl"
                elif "manuscrito" in base_name:
                    mapping_name = "char_mapping_manuscrito.pkl"
                else:
                    mapping_name = base_name.replace(".h5", "_mapping.pkl")
                
                mapping_path = os.path.join(os.path.dirname(self.model_path), mapping_name)

                if os.path.exists(mapping_path):
                    with open(mapping_path, "rb") as f:
                        self.classes = pickle.load(f)
                else:
                    # Fallback a .labels.txt
                    meta_path = self.model_path + ".labels.txt"
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            self.classes = [line.strip() for line in f if line.strip()]
            except Exception:
                self.model = None

        # Si no hay modelo, intentar entrenar (pero solo si se llama explícitamente o es el default)
        # Para evitar reentrenamientos accidentales en producción, mejor dejar que el script de entrenamiento lo haga.
        # Pero mantendremos la lógica original de intentar entrenar si no existe, por robustez.
        if self.model is None and os.path.exists(self.templates_dir):
            try:
                print(f"Modelo no encontrado en {self.model_path}, intentando entrenar...")
                self.train_from_templates(epochs=5)
            except Exception as e:
                print(f"Error entrenando modelo automático: {e}")
                self.model = None

    def _enumerate_templates(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X: List[np.ndarray] = []
        y: List[int] = []
        labels: List[str] = []
        label_to_idx: Dict[str, int] = {}
        
        if not os.path.exists(self.templates_dir):
            return np.empty((0, 32, 32, 1), dtype=np.float32), np.empty((0,), dtype=np.int64), []
            
        ignored_containers = ["mayusculas", "minusculas", "numeros", "simbolos", "digital", "manuscrito", "templates"]
        ignored_labels = ["dot", "colon", "semicolon", "comma", "slash", "backslash", "quote", "doublequote"]
        
        print(f"Buscando templates en: {os.path.abspath(self.templates_dir)}")
        for root, dirs, files in os.walk(self.templates_dir):
            label_name = os.path.basename(root)
            
            # Ignorar raíz
            if os.path.abspath(root) == os.path.abspath(self.templates_dir):
                continue
                
            # Ignorar contenedores
            if label_name in ignored_containers:
                continue
                
            # Verificar si tiene imágenes
            has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for f in files)
            if not has_images:
                continue
            
            # Mapeos especiales
            if label_name == "at": label = "@"
            elif label_name == "underscore": label = "_"
            elif label_name == "hyphen": label = "-"
            elif label_name == "lparen": label = "(" # Posibles nombres seguros
            elif label_name == "rparen": label = ")"
            elif label_name == "plus": label = "+"
            else: label = label_name
            
            # Permitir que el nombre de la carpeta sea el label directo (ej. "(", "+") si el sistema de archivos lo permite
            
            if label not in label_to_idx:
                label_to_idx[label] = len(labels)
                labels.append(label)
                
            for filename in files:
                if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue
                img_path = os.path.join(root, filename)
                try:
                    stream = open(img_path, "rb")
                    bytes_data = bytearray(stream.read())
                    numpyarray = np.asarray(bytes_data, dtype=np.uint8)
                    img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
                    stream.close()
                except Exception:
                    continue
                if img is None:
                    continue
                    
                # Preprocesamiento
                _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                n_white = cv2.countNonZero(bin_img)
                n_pixels = bin_img.shape[0] * bin_img.shape[1]
                if n_white > n_pixels // 2:
                    bin_img = cv2.bitwise_not(bin_img)
                    
                coords = cv2.findNonZero(bin_img)
                if coords is None:
                    continue
                    
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(coords)
                roi = bin_img[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]
                roi = self._make_square(roi, 32)
                
                X.append(roi.astype(np.float32) / 255.0)
                y.append(label_to_idx[label])
                
        if len(X) == 0:
            return np.empty((0, 32, 32, 1), dtype=np.float32), np.empty((0,), dtype=np.int64), []
            
        X = np.expand_dims(np.stack(X, axis=0), axis=-1)
        y = np.asarray(y, dtype=np.int64)
        return X, y, labels

    def _make_square(self, img: np.ndarray, size: int = 32) -> np.ndarray:
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((size, size), dtype=np.uint8)
        target = size - 4
        scale = min(target / h, target / w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
        sq = np.zeros((size, size), dtype=np.uint8)
        y0 = (size - nh) // 2
        x0 = (size - nw) // 2
        sq[y0:y0+nh, x0:x0+nw] = resized
        return sq

    def _build_model(self, num_classes: int) -> tf.keras.Model:
        inputs = layers.Input(shape=(32, 32, 1))
        x = layers.Conv2D(32, 3, activation="relu")(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train_from_templates(self, epochs: int = 5):
        if tf is None:
            return
        X, y, labels = self._enumerate_templates()
        if X.shape[0] == 0:
            return
        self.classes = labels
        self.model = self._build_model(num_classes=len(labels))
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=1, validation_split=0.1)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        
        # Guardar mapping en pickle
        base_name = os.path.basename(self.model_path)
        if "digital" in base_name:
            mapping_name = "char_mapping_digital.pkl"
        elif "manuscrito" in base_name:
            mapping_name = "char_mapping_manuscrito.pkl"
        else:
            mapping_name = base_name.replace(".h5", "_mapping.pkl")
        
        mapping_path = os.path.join(os.path.dirname(self.model_path), mapping_name)
        
        with open(mapping_path, "wb") as f:
            pickle.dump(self.classes, f)
        print(f"Modelo guardado en {self.model_path}")
        print(f"Mapping guardado en {mapping_path}")

    def predict_char(self, char_img: np.ndarray) -> Tuple[str, float]:
        if self.model is None or tf is None:
            return "?", 0.0
        x = char_img.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=(0, -1))
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def recognize_line(self, chars_data: List[Dict]) -> str:
        if not chars_data:
            return ""
        widths = [c['w'] for c in chars_data]
        avg_width = np.mean(widths) if widths else 10
        space_threshold = avg_width * 0.4
        text = ""
        for i, c in enumerate(chars_data):
            if i > 0:
                prev = chars_data[i-1]
                gap = c['x'] - (prev['x'] + prev['w'])
                if gap > space_threshold:
                    text += " "
            label, score = self.predict_char(c['img'])
            if label != "?":
                text += label
        return text
