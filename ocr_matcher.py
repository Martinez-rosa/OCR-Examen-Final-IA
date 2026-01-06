"""
Motor de reconocimiento OCR basado en comparación con plantillas
"""

import os
import cv2
import numpy as np
from pathlib import Path
from preprocessor import ImagePreprocessor
from feature_comparator import FeatureComparator

class TemplateDatabase:
    """Base de datos de plantillas de caracteres"""
    
    def __init__(self, templates_path, target_size=(32, 32)):
        """
        Args:
            templates_path: Ruta a carpeta con plantillas organizadas por clase
            target_size: Tamaño objetivo para todas las plantillas
        """
        self.templates_path = templates_path
        self.target_size = target_size
        self.preprocessor = ImagePreprocessor(target_size)
        
        # Diccionario: clase -> lista de imágenes preprocesadas
        self.templates = {}
        
        # Cargar plantillas
        self.load_templates()
    
    def load_templates(self):
        """Carga todas las plantillas del dataset"""
        print(f"Cargando plantillas desde: {self.templates_path}")
        
        if not os.path.exists(self.templates_path):
            raise ValueError(f"No existe la ruta: {self.templates_path}")
        
        # Listar todas las clases (subcarpetas)
        classes = sorted([d for d in os.listdir(self.templates_path) 
                         if os.path.isdir(os.path.join(self.templates_path, d))])
        
        total_templates = 0
        
        for class_name in classes:
            class_path = os.path.join(self.templates_path, class_name)
            
            # Listar imágenes en la carpeta
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            self.templates[class_name] = []
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Cargar y preprocesar
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    # Preprocesar
                    processed = self.preprocessor.preprocess(img)
                    
                    # Guardar
                    self.templates[class_name].append(processed)
                    total_templates += 1
                    
                except Exception as e:
                    print(f"  ⚠ Error cargando {img_path}: {e}")
        
        print(f"  ✓ Cargadas {len(self.templates)} clases")
        print(f"  ✓ Total plantillas: {total_templates}")
        
        # Mostrar distribución
        print("\n  Distribución de plantillas:")
        for class_name in sorted(self.templates.keys()):
            count = len(self.templates[class_name])
            print(f"    '{class_name}': {count} muestras")
    
    def get_classes(self):
        """Retorna lista de todas las clases disponibles"""
        return list(self.templates.keys())
    
    def get_templates(self, class_name):
        """Retorna todas las plantillas de una clase"""
        return self.templates.get(class_name, [])


class CharacterMatcher:
    """Compara caracteres desconocidos con plantillas usando múltiples métricas"""
    
    def __init__(self, template_db, methods='all'):
        """
        Args:
            template_db: Instancia de TemplateDatabase
            methods: 'all', 'fast', o lista de métodos específicos
        """
        self.template_db = template_db
        self.comparator = FeatureComparator()
        
        # Definir métodos de comparación disponibles
        self.available_methods = {
            'template_matching': {
                'func': self.comparator.template_matching,
                'weight': 1.5,  # Peso en votación
                'higher_better': True
            },
            'ssim': {
                'func': self.comparator.ssim_similarity,
                'weight': 1.5,
                'higher_better': True
            },
            'hog': {
                'func': self.comparator.hog_similarity,
                'weight': 1.2,
                'higher_better': True
            },
            'hu_moments': {
                'func': self.comparator.hu_moments_similarity,
                'weight': 0.8,
                'higher_better': True
            },
            'pixel_correlation': {
                'func': self.comparator.pixel_correlation,
                'weight': 1.0,
                'higher_better': True
            },
            'contour': {
                'func': self.comparator.contour_similarity,
                'weight': 0.7,
                'higher_better': True
            },
            'histogram': {
                'func': self.comparator.histogram_similarity,
                'weight': 0.6,
                'higher_better': True
            }
        }
        
        # Seleccionar métodos activos
        if methods == 'all':
            self.active_methods = list(self.available_methods.keys())
        elif methods == 'fast':
            # Solo los métodos más rápidos
            self.active_methods = ['template_matching', 'pixel_correlation', 'histogram']
        elif isinstance(methods, list):
            self.active_methods = methods
        else:
            self.active_methods = ['template_matching', 'ssim', 'hog']
        
        print(f"Métodos activos: {', '.join(self.active_methods)}")
    
    def compare_with_template(self, img, template, method_name):
        """
        Compara imagen con una plantilla usando un método específico
        
        Args:
            img: Imagen a reconocer
            template: Plantilla de referencia
            method_name: Nombre del método a usar
        Returns:
            Score de similitud [0, 1]
        """
        method_info = self.available_methods[method_name]
        func = method_info['func']
        
        try:
            score = func(img, template)
            return score
        except Exception as e:
            print(f"    ⚠ Error en método {method_name}: {e}")
            return 0.0
    
    def match_character(self, img, top_n=3, verbose=False):
        """
        Reconoce un carácter comparándolo con todas las plantillas
        
        Args:
            img: Imagen preprocesada del carácter a reconocer
            top_n: Número de mejores candidatos a retornar
            verbose: Si True, muestra progreso
        Returns:
            Diccionario con:
                - 'character': carácter reconocido
                - 'confidence': confianza [0, 1]
                - 'top_n': lista de mejores candidatos
                - 'method_scores': scores por método
        """
        if verbose:
            print("\n  Comparando con plantillas...")
        
        # Diccionario para almacenar scores por clase
        # scores[clase][método] = lista de scores con plantillas de esa clase
        class_scores = {class_name: {method: [] for method in self.active_methods}
                        for class_name in self.template_db.get_classes()}
        
        # Comparar con cada plantilla de cada clase
        for class_name in self.template_db.get_classes():
            templates = self.template_db.get_templates(class_name)
            
            for template in templates:
                for method_name in self.active_methods:
                    score = self.compare_with_template(img, template, method_name)
                    class_scores[class_name][method_name].append(score)
        
        # Calcular score promedio por clase para cada método
        class_avg_scores = {}
        
        for class_name in self.template_db.get_classes():
            method_scores = {}
            
            for method_name in self.active_methods:
                scores = class_scores[class_name][method_name]
                
                if scores:
                    # Usar el score máximo (mejor match) de esa clase
                    method_scores[method_name] = max(scores)
                else:
                    method_scores[method_name] = 0.0
            
            class_avg_scores[class_name] = method_scores
        
        # Votación ponderada: combinar scores de todos los métodos
        final_scores = {}
        
        for class_name in self.template_db.get_classes():
            weighted_sum = 0.0
            total_weight = 0.0
            
            for method_name in self.active_methods:
                score = class_avg_scores[class_name][method_name]
                weight = self.available_methods[method_name]['weight']
                
                weighted_sum += score * weight
                total_weight += weight
            
            # Score final normalizado
            final_scores[class_name] = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Ordenar por score (mayor a menor)
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Mejores candidatos
        top_candidates = sorted_results[:top_n]
        
        # Resultado final
        best_char = top_candidates[0][0]
        confidence = top_candidates[0][1]
        
        # Información detallada por método (para debugging)
        method_details = {}
        for method_name in self.active_methods:
            method_details[method_name] = {
                'winner': max(class_avg_scores.items(), 
                             key=lambda x: x[1][method_name])[0],
                'score': max(class_avg_scores[class_name][method_name] 
                            for class_name in self.template_db.get_classes())
            }
        
        if verbose:
            print(f"\n  Resultados por método:")
            for method_name, details in method_details.items():
                print(f"    {method_name}: '{details['winner']}' ({details['score']:.4f})")
            
            print(f"\n  Top {top_n} candidatos:")
            for i, (char, score) in enumerate(top_candidates, 1):
                print(f"    {i}. '{char}': {score:.4f} ({score*100:.2f}%)")
        
        return {
            'character': best_char,
            'confidence': confidence,
            'top_n': top_candidates,
            'method_scores': method_details
        }


if __name__ == '__main__':
    # Prueba del matcher
    print("="*60)
    print("PRUEBA DE OCR MATCHER")
    print("="*60)
    
    # Cargar base de datos de plantillas
    templates_path = 'datasets/templates'  # Ajustar ruta
    
    if os.path.exists(templates_path):
        # Crear base de datos
        db = TemplateDatabase(templates_path)
        
        # Crear matcher
        matcher = CharacterMatcher(db, methods='all')
        
        # Cargar imagen de test
        test_img_path = 'test_char.png'  # Ajustar ruta
        
        if os.path.exists(test_img_path):
            # Cargar y preprocesar
            preprocessor = ImagePreprocessor()
            img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
            img = preprocessor.preprocess(img)
            
            # Reconocer
            result = matcher.match_character(img, top_n=5, verbose=True)
            
            print(f"\n{'='*60}")
            print(f"✓ CARÁCTER RECONOCIDO: '{result['character']}'")
            print(f"  Confianza: {result['confidence']:.2%}")
            print(f"{'='*60}")
        else:
            print(f"\n⚠ No se encontró imagen de test: {test_img_path}")
    else:
        print(f"\n⚠ No se encontró carpeta de plantillas: {templates_path}")