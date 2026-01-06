"""
Script de testing y evaluación del sistema OCR
Calcula métricas de rendimiento sobre dataset de test
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path
import json
from collections import defaultdict
from ocr_system import OCRSystem
from preprocessor import ImagePreprocessor

class OCRTester:
    """Evalúa el rendimiento del sistema OCR"""
    
    def __init__(self, ocr_system, test_data_path):
        """
        Args:
            ocr_system: Instancia de OCRSystem
            test_data_path: Ruta a carpeta de test (misma estructura que templates)
        """
        self.ocr = ocr_system
        self.test_data_path = test_data_path
        self.results = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'by_class': defaultdict(lambda: {'total': 0, 'correct': 0}),
            'confusion_matrix': defaultdict(lambda: defaultdict(int)),
            'low_confidence': [],
            'processing_times': []
        }
    
    def test_character(self, img, true_label, verbose=False):
        """
        Prueba reconocimiento de un carácter individual
        
        Args:
            img: Imagen del carácter
            true_label: Etiqueta verdadera
            verbose: Mostrar detalles
        Returns:
            Diccionario con resultado de la prueba
        """
        start_time = time.time()
        
        # Reconocer
        result = self.ocr.recognize_character(img, verbose=False)
        
        processing_time = time.time() - start_time
        
        predicted = result['character']
        confidence = result['confidence']
        is_correct = (predicted == true_label)
        
        # Actualizar estadísticas
        self.results['total'] += 1
        self.results['by_class'][true_label]['total'] += 1
        self.results['processing_times'].append(processing_time)
        
        if is_correct:
            self.results['correct'] += 1
            self.results['by_class'][true_label]['correct'] += 1
        else:
            self.results['incorrect'] += 1
        
        # Matriz de confusión
        self.results['confusion_matrix'][true_label][predicted] += 1
        
        # Casos de baja confianza
        if confidence < 0.5:
            self.results['low_confidence'].append({
                'true': true_label,
                'predicted': predicted,
                'confidence': confidence
            })
        
        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"  {status} Real: '{true_label}' | Predicho: '{predicted}' | Confianza: {confidence:.2%} | Tiempo: {processing_time:.3f}s")
        
        return {
            'correct': is_correct,
            'predicted': predicted,
            'confidence': confidence,
            'time': processing_time
        }
    
    def run_tests(self, max_per_class=None, verbose=True):
        """
        Ejecuta pruebas sobre todo el dataset de test
        
        Args:
            max_per_class: Máximo de muestras a probar por clase (None = todas)
            verbose: Mostrar progreso
        """
        if verbose:
            print("\n" + "="*60)
            print("EJECUTANDO PRUEBAS")
            print("="*60)
        
        # Listar todas las clases
        classes = sorted([d for d in os.listdir(self.test_data_path) 
                         if os.path.isdir(os.path.join(self.test_data_path, d))])
        
        if verbose:
            print(f"\nClases a probar: {len(classes)}")
            print(f"Clases: {', '.join(classes)}\n")
        
        # Probar cada clase
        for class_idx, class_name in enumerate(classes, 1):
            class_path = os.path.join(self.test_data_path, class_name)
            
            # Listar imágenes
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            # Limitar cantidad si se especifica
            if max_per_class is not None:
                image_files = image_files[:max_per_class]
            
            if verbose:
                print(f"[{class_idx}/{len(classes)}] Probando clase '{class_name}' ({len(image_files)} muestras)...")
            
            # Probar cada imagen
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Cargar imagen
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    # Probar
                    self.test_character(img, class_name, verbose=False)
                    
                except Exception as e:
                    if verbose:
                        print(f"  ⚠ Error con {img_path}: {e}")
        
        if verbose:
            print("\n✓ Pruebas completadas")
    
    def calculate_metrics(self):
        """Calcula métricas de rendimiento"""
        metrics = {}
        
        # Accuracy global
        if self.results['total'] > 0:
            metrics['accuracy'] = self.results['correct'] / self.results['total']
        else:
            metrics['accuracy'] = 0.0
        
        # Accuracy por clase
        metrics['accuracy_by_class'] = {}
        for class_name, stats in self.results['by_class'].items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
            else:
                acc = 0.0
            metrics['accuracy_by_class'][class_name] = acc
        
        # Accuracy por tipo (dígitos, mayúsculas, minúsculas)
        metrics['accuracy_by_type'] = {}
        
        digits = [c for c in self.results['by_class'].keys() if c.isdigit()]
        uppercase = [c for c in self.results['by_class'].keys() if c.isupper() and c.isalpha()]
        lowercase = [c for c in self.results['by_class'].keys() if c.islower() and c.isalpha()]
        
        for category, chars in [('digits', digits), ('uppercase', uppercase), ('lowercase', lowercase)]:
            if chars:
                total = sum(self.results['by_class'][c]['total'] for c in chars)
                correct = sum(self.results['by_class'][c]['correct'] for c in chars)
                metrics['accuracy_by_type'][category] = correct / total if total > 0 else 0.0
        
        # Tiempo promedio
        if self.results['processing_times']:
            metrics['avg_time'] = np.mean(self.results['processing_times'])
            metrics['min_time'] = np.min(self.results['processing_times'])
            metrics['max_time'] = np.max(self.results['processing_times'])
        else:
            metrics['avg_time'] = 0.0
            metrics['min_time'] = 0.0
            metrics['max_time'] = 0.0
        
        # Casos de baja confianza
        metrics['low_confidence_count'] = len(self.results['low_confidence'])
        metrics['low_confidence_percent'] = (len(self.results['low_confidence']) / 
                                             self.results['total'] * 100 
                                             if self.results['total'] > 0 else 0.0)
        
        return metrics
    
    def print_report(self):
        """Imprime reporte detallado de resultados"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("REPORTE DE EVALUACIÓN")
        print("="*60)
        
        # Resultados generales
        print("\n📊 RESULTADOS GENERALES:")
        print(f"  Total pruebas: {self.results['total']}")
        print(f"  Correctos: {self.results['correct']} ({metrics['accuracy']:.2%})")
        print(f"  Incorrectos: {self.results['incorrect']}")
        
        # Por tipo
        print("\n📈 ACCURACY POR TIPO:")
        for tipo, acc in metrics['accuracy_by_type'].items():
            print(f"  {tipo.capitalize()}: {acc:.2%}")
        
        # Tiempos
        print("\n⏱️  TIEMPOS DE PROCESAMIENTO:")
        print(f"  Promedio: {metrics['avg_time']:.4f}s")
        print(f"  Mínimo: {metrics['min_time']:.4f}s")
        print(f"  Máximo: {metrics['max_time']:.4f}s")
        
        # Baja confianza
        print("\n⚠️  CASOS DE BAJA CONFIANZA (<50%):")
        print(f"  Total: {metrics['low_confidence_count']} ({metrics['low_confidence_percent']:.2f}%)")
        
        if metrics['low_confidence_count'] > 0:
            print("\n  Ejemplos:")
            for case in self.results['low_confidence'][:10]:
                print(f"    Real: '{case['true']}' → Predicho: '{case['predicted']}' (confianza: {case['confidence']:.2%})")
        
        # Top 5 mejores clases
        print("\n🏆 TOP 5 MEJOR RECONOCIDAS:")
        sorted_classes = sorted(metrics['accuracy_by_class'].items(), 
                               key=lambda x: x[1], reverse=True)
        for i, (char, acc) in enumerate(sorted_classes[:5], 1):
            print(f"  {i}. '{char}': {acc:.2%}")
        
        # Top 5 peores clases
        print("\n❌ TOP 5 PEOR RECONOCIDAS:")
        for i, (char, acc) in enumerate(sorted_classes[-5:][::-1], 1):
            print(f"  {i}. '{char}': {acc:.2%}")
        
        # Confusiones más comunes
        print("\n🔀 CONFUSIONES MÁS COMUNES:")
        confusions = []
        for true_label, predictions in self.results['confusion_matrix'].items():
            for predicted, count in predictions.items():
                if true_label != predicted and count > 0:
                    confusions.append((true_label, predicted, count))
        
        confusions.sort(key=lambda x: x[2], reverse=True)
        
        for i, (true_label, predicted, count) in enumerate(confusions[:10], 1):
            print(f"  {i}. '{true_label}' confundido con '{predicted}': {count} veces")
        
        print("\n" + "="*60)
    
    def save_report(self, output_path):
        """Guarda reporte en archivo de texto"""
        metrics = self.calculate_metrics()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("REPORTE DE EVALUACIÓN OCR\n")
            f.write("="*60 + "\n\n")
            
            f.write("RESULTADOS GENERALES:\n")
            f.write(f"  Total pruebas: {self.results['total']}\n")
            f.write(f"  Correctos: {self.results['correct']} ({metrics['accuracy']:.2%})\n")
            f.write(f"  Incorrectos: {self.results['incorrect']}\n\n")
            
            f.write("ACCURACY POR TIPO:\n")
            for tipo, acc in metrics['accuracy_by_type'].items():
                f.write(f"  {tipo.capitalize()}: {acc:.2%}\n")
            f.write("\n")
            
            f.write("ACCURACY POR CLASE:\n")
            for char in sorted(metrics['accuracy_by_class'].keys()):
                acc = metrics['accuracy_by_class'][char]
                total = self.results['by_class'][char]['total']
                correct = self.results['by_class'][char]['correct']
                f.write(f"  '{char}': {acc:.2%} ({correct}/{total})\n")
            f.write("\n")
            
            f.write("TIEMPOS:\n")
            f.write(f"  Promedio: {metrics['avg_time']:.4f}s\n")
            f.write(f"  Mínimo: {metrics['min_time']:.4f}s\n")
            f.write(f"  Máximo: {metrics['max_time']:.4f}s\n\n")
            
            f.write(f"CASOS DE BAJA CONFIANZA: {metrics['low_confidence_count']} ({metrics['low_confidence_percent']:.2f}%)\n")
        
        print(f"✓ Reporte guardado en: {output_path}")
    
    def save_confusion_matrix(self, output_path):
        """Guarda matriz de confusión en formato JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convertir defaultdict a dict normal
        matrix = {k: dict(v) for k, v in self.results['confusion_matrix'].items()}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(matrix, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Matriz de confusión guardada en: {output_path}")


def main():
    """Función principal de testing"""
    import sys
    
    print("="*60)
    print("EVALUACIÓN DEL SISTEMA OCR")
    print("="*60)
    
    # Configuración
    TEMPLATES_PATH = 'datasets/templates'  # Plantillas para el sistema
    TEST_DATA_PATH = 'datasets/test'       # Datos de test
    REPORT_PATH = 'results/evaluation_report.txt'
    CONFUSION_PATH = 'results/confusion_matrix.json'
    
    # Verificar rutas
    if not os.path.exists(TEMPLATES_PATH):
        print(f"\n❌ Error: No se encontró carpeta de plantillas: {TEMPLATES_PATH}")
        sys.exit(1)
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"\n❌ Error: No se encontró carpeta de test: {TEST_DATA_PATH}")
        sys.exit(1)
    
    # Inicializar sistema OCR
    print("\nInicializando sistema OCR...")
    ocr = OCRSystem(
        templates_path=TEMPLATES_PATH,
        methods='all',
        target_size=(32, 32)
    )
    
    # Crear tester
    tester = OCRTester(ocr, TEST_DATA_PATH)
    
    # Ejecutar pruebas
    print("\nEjecutando pruebas...")
    tester.run_tests(max_per_class=50, verbose=True)  # Máximo 50 por clase para rapidez
    
    # Mostrar reporte
    tester.print_report()
    
    # Guardar resultados
    print("\nGuardando resultados...")
    tester.save_report(REPORT_PATH)
    tester.save_confusion_matrix(CONFUSION_PATH)
    
    print("\n" + "="*60)
    print("✓ EVALUACIÓN COMPLETADA")
    print("="*60)


if __name__ == '__main__':
    main()