"""
Servidor web para interfaz OCR
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from datetime import datetime

from ocr_system import OCRSystem
from image_table_detector import ImageTableDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear carpetas si no existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/extracted', exist_ok=True)

# Inicializar sistema OCR (carga una sola vez)
print("Inicializando sistema OCR...")
ocr_system = OCRSystem('datasets/templates', methods='all')
detector = ImageTableDetector()
print("Sistema listo!")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró imagen'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó archivo'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Formato no permitido'}), 400
    
    try:
        # Guardar archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Cargar imagen
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'No se pudo leer la imagen'}), 400
        
        # 1. DETECTAR TABLAS E IMÁGENES
        print("Detectando tablas e imágenes...")
        tables = detector.detect_tables(img)
        images = detector.detect_images(img)
        
        # Guardar tablas extraídas
        table_files = []
        for i, table_data in enumerate(tables):
            table_img = table_data['image']
            table_filename = f"{timestamp}_table_{i+1}.png"
            table_path = os.path.join('static/extracted', table_filename)
            cv2.imwrite(table_path, table_img)
            table_files.append({
                'filename': table_filename,
                'url': f'/static/extracted/{table_filename}',
                'position': table_data['position']
            })
        
        # Guardar imágenes extraídas
        image_files = []
        for i, img_data in enumerate(images):
            extracted_img = img_data['image']
            img_filename = f"{timestamp}_image_{i+1}.png"
            img_path = os.path.join('static/extracted', img_filename)
            cv2.imwrite(img_path, extracted_img)
            image_files.append({
                'filename': img_filename,
                'url': f'/static/extracted/{img_filename}',
                'position': img_data['position']
            })
        
        # 2. OCR DEL TEXTO (excluyendo áreas de tablas e imágenes)
        print("Realizando OCR...")
        
        # Crear máscara para excluir tablas e imágenes
        mask = np.ones(img.shape, dtype=np.uint8) * 255
        
        for table_data in tables:
            x, y, w, h = table_data['position']
            mask[y:y+h, x:x+w] = 0
        
        for img_data in images:
            x, y, w, h = img_data['position']
            mask[y:y+h, x:x+w] = 0
        
        # Aplicar máscara
        img_masked = cv2.bitwise_and(img, mask)
        
        # Reconocer texto
        result = ocr_system.recognize_text(img_masked, verbose=True)
        
        # Preparar respuesta
        response = {
            'success': True,
            'text': result['text'],
            'confidence': f"{result['avg_confidence']:.2%}",
            'total_chars': result['total_chars'],
            'tables': table_files,
            'images': image_files,
            'processing_info': {
                'tables_found': len(tables),
                'images_found': len(images)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/static/extracted/<filename>')
def serve_extracted(filename):
    return send_from_directory('static/extracted', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)