"""
Generador de plantillas para OCR.
Crea un dataset base de caracteres tipográficos (A-Z, 0-9) usando fuentes del sistema o PIL.
Esto sirve como referencia "perfecta" para la comparación por plantillas.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string

def generate_templates(output_dir: str = "templates"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Caracteres a generar: Mayúsculas, Minúsculas, Dígitos y Puntuación
    categories = {
        "mayusculas": string.ascii_uppercase,
        "minusculas": string.ascii_lowercase,
        "numeros": string.digits,
        "simbolos": ".,:;@-_()[]/+"
    }
    
    # Tamaño de la imagen de plantilla
    size = 32 # Cambiado a 32 para coincidir con recognition/segmentation default
    font_size = 24
    
    # Fuentes a probar (Windows paths comunes)
    fonts_to_try = [
        "arial.ttf", 
        "calibri.ttf", 
        "times.ttf", 
        "cour.ttf", # Courier
        "seguiemj.ttf", # Segoe UI
        "verdana.ttf"
    ]
    
    loaded_fonts = []
    
    # Cargar todas las fuentes disponibles
    for font_name in fonts_to_try:
        try:
            # Intentar ruta local directa
            font = ImageFont.truetype(font_name, font_size)
            loaded_fonts.append((font_name, font))
        except IOError:
            try:
                # Intentar ruta absoluta Windows
                font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", font_name)
                font = ImageFont.truetype(font_path, font_size)
                loaded_fonts.append((font_name, font))
            except IOError:
                continue
    
    if not loaded_fonts:
        print("[WARN] No se encontraron fuentes del sistema. Usando defecto.")
        loaded_fonts.append(("default", ImageFont.load_default()))

    print(f"[INFO] Generando plantillas en '{output_dir}' usando {len(loaded_fonts)} fuentes...")

    count = 0
    for cat_name, chars_in_cat in categories.items():
        cat_dir = os.path.join(output_dir, cat_name)
        
        for char in chars_in_cat:
            # Manejo especial para nombres de carpeta
            folder_name = char
            if char == ".": folder_name = "dot"
            elif char == ":": folder_name = "colon"
            elif char == "/": folder_name = "slash"
            elif char == "\\": folder_name = "backslash"
            elif char == "*": folder_name = "asterisk"
            elif char == "?": folder_name = "question"
            elif char == "\"": folder_name = "quote"
            elif char == "<": folder_name = "lt"
            elif char == ">": folder_name = "gt"
            elif char == "|": folder_name = "pipe"
            elif char.isupper() and cat_name != "mayusculas": folder_name = char + "_upper" # Por si acaso
            
            # Crear carpeta para el caracter
            char_dir = os.path.join(cat_dir, folder_name)
            if not os.path.exists(char_dir):
                os.makedirs(char_dir)
            
            # Generar variaciones para cada fuente
            for font_name, font in loaded_fonts:
                # Variación 1: Normal
                img = _create_char_image(char, font, size)
                save_path = os.path.join(char_dir, f"tpl_{font_name}_norm.png")
                img.save(save_path)
                count += 1
                
                # Variación 2: "Bold" simulado (dilatación en imagen blanca sobre negro = trazo más grueso)
                cv_img = np.array(img)
                kernel = np.ones((2,2), np.uint8)
                bold_img = cv2.dilate(cv_img, kernel, iterations=1)
                Image.fromarray(bold_img).save(os.path.join(char_dir, f"tpl_{font_name}_bold.png"))
                count += 1

    print(f"[INFO] Generación completa. {count} plantillas creadas.")

def _create_char_image(char, font, size):
    # Crear imagen negra
    img = Image.new('L', (size, size), color=0) 
    draw = ImageDraw.Draw(img)
    
    # Calcular posición para centrar
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback para versiones viejas de PIL
        w, h = draw.textsize(char, font=font)
    
    x = (size - w) // 2
    y = (size - h) // 2
    
    # Dibujar texto blanco
    draw.text((x, y), char, font=font, fill=255)
    return img

if __name__ == "__main__":
    generate_templates()
