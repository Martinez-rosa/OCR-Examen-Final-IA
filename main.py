"""
Script Principal OCR sin ML.

Orquesta el proceso de OCR utilizando la clase unificada OCRSystem.
"""

import os
import argparse
import time
from ocr_system import OCRSystem
import template_gen
import cv2

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser(description="Sistema OCR CV Puro (Sin ML)")
    parser.add_argument("--image", required=True, help="Ruta de la imagen a procesar")
    parser.add_argument("--output_dir", default="output", help="Carpeta de salida")
    parser.add_argument("--templates_dir", default="templates", help="Carpeta de plantillas de referencia")
    parser.add_argument("--generate_templates", action="store_true", help="Generar plantillas básicas antes de procesar")
    parser.add_argument("--debug", action="store_true", help="Habilitar modo debug (guardar caracteres)")
    
    args = parser.parse_args()

    # 0. Preparación
    if args.debug:
        print("[INFO] Modo DEBUG activado. Limpiando carpeta debug_chars...")
        import shutil
        if os.path.exists("debug_chars"):
            try:
                shutil.rmtree("debug_chars")
            except Exception as e:
                print(f"[WARN] No se pudo limpiar debug_chars: {e}")
        
        if not os.path.exists("debug_chars"):
            os.makedirs("debug_chars")

    if args.generate_templates or not os.path.exists(args.templates_dir):
        print("[INFO] Generando/Verificando plantillas...")
        template_gen.generate_templates(args.templates_dir)

    ensure_dir(args.output_dir)
    
    # Inicializar Sistema OCR
    ocr = OCRSystem(templates_dir=args.templates_dir, debug=args.debug)
    
    print(f"[INFO] Procesando: {args.image}")
    start_time = time.time()
    
    try:
        result = ocr.process_image(args.image)
    except Exception as e:
        print(f"[ERROR] Fallo en el procesamiento: {e}")
        return

    elapsed = time.time() - start_time
    
    # Guardar Resultados
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # 1. Guardar Texto
    text_path = os.path.join(args.output_dir, f"{base_name}_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
        
    print(f"[DONE] Texto guardado en: {text_path}")
    
    # 2. Guardar Tablas
    if result["tables_data"]:
        tables_dir = os.path.join(args.output_dir, "tables")
        ensure_dir(tables_dir)
        print(f"[INFO] Guardando {len(result['tables_data'])} tablas...")
        
        for i, (table_data, table_img) in enumerate(zip(result["tables_data"], result["tables_imgs"])):
            # Guardar CSV
            csv_path = os.path.join(tables_dir, f"{base_name}_table_{i}.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                for row in table_data:
                    f.write(",".join([f'"{c}"' for c in row]) + "\n")
            
            # Guardar Imagen
            img_path = os.path.join(tables_dir, f"{base_name}_table_{i}.png")
            cv2.imwrite(img_path, table_img)

    # 3. Guardar Imágenes Detectadas
    if result["images"]:
        images_dir = os.path.join(args.output_dir, "images")
        ensure_dir(images_dir)
        print(f"[INFO] Guardando {len(result['images'])} imágenes detectadas...")
        for i, img in enumerate(result["images"]):
            cv2.imwrite(os.path.join(images_dir, f"{base_name}_image_{i}.png"), img)
            
    # 4. Guardar Códigos
    if result["codes"]:
        codes_dir = os.path.join(args.output_dir, "codes")
        ensure_dir(codes_dir)
        print(f"[INFO] Guardando {len(result['codes'])} códigos...")
        for i, code in enumerate(result["codes"]):
            cv2.imwrite(os.path.join(codes_dir, f"{base_name}_code_{i}.png"), code)

    print(f"       Tiempo total: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
