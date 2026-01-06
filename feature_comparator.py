"""
Módulo de comparación de características para OCR
Implementa múltiples métricas de similitud entre imágenes
"""

import cv2
import numpy as np
from skimage.feature import hog
from skimage.metrics import structural_similarity
from scipy.spatial.distance import euclidean, cityblock

class FeatureComparator:
    """Compara características entre dos imágenes de caracteres"""
    
    def __init__(self):
        """Inicializa el comparador"""
        pass
    
    # ============ COMPARACIÓN DIRECTA DE PÍXELES ============
    
    def pixel_distance_euclidean(self, img1, img2):
        """
        Distancia euclidiana entre píxeles
        
        Args:
            img1, img2: Imágenes del mismo tamaño
        Returns:
            Distancia (menor = más similar)
        """
        # Normalizar imágenes a [0, 1]
        img1_norm = img1.astype(float) / 255.0
        img2_norm = img2.astype(float) / 255.0
        
        # Calcular distancia euclidiana
        distance = np.sqrt(np.sum((img1_norm - img2_norm) ** 2))
        
        return distance
    
    def pixel_distance_manhattan(self, img1, img2):
        """
        Distancia Manhattan (L1) entre píxeles
        
        Args:
            img1, img2: Imágenes del mismo tamaño
        Returns:
            Distancia (menor = más similar)
        """
        distance = np.sum(np.abs(img1.astype(float) - img2.astype(float)))
        return distance / (img1.shape[0] * img1.shape[1])  # Normalizar
    
    def pixel_correlation(self, img1, img2):
        """
        Correlación entre píxeles
        
        Args:
            img1, img2: Imágenes del mismo tamaño
        Returns:
            Similitud [0, 1] (mayor = más similar)
        """
        # Aplanar imágenes
        vec1 = img1.flatten().astype(float)
        vec2 = img2.flatten().astype(float)
        
        # Calcular correlación de Pearson
        correlation = np.corrcoef(vec1, vec2)[0, 1]
        
        # Convertir a [0, 1]
        similarity = (correlation + 1) / 2
        
        return similarity
    
    # ============ TEMPLATE MATCHING ============
    
    def template_matching(self, img1, img2, method=cv2.TM_CCOEFF_NORMED):
        """
        Template matching usando OpenCV
        
        Args:
            img1: Imagen template
            img2: Imagen a comparar
            method: Método de cv2.matchTemplate
        Returns:
            Similitud [0, 1] para métodos normalizados
        """
        # Asegurar mismo tamaño
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Template matching
        result = cv2.matchTemplate(img2, img1, method)
        
        # Para métodos SQDIFF, invertir (menor es mejor)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            similarity = 1 - result[0, 0]
        else:
            similarity = result[0, 0]
        
        return max(0, min(1, similarity))  # Clamp a [0, 1]
    
    # ============ SSIM (STRUCTURAL SIMILARITY) ============
    
    def ssim_similarity(self, img1, img2):
        """
        Similitud estructural (SSIM)
        
        Args:
            img1, img2: Imágenes del mismo tamaño
        Returns:
            Similitud [-1, 1], típicamente [0, 1]
        """
        # Asegurar mismo tamaño
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Calcular SSIM
        similarity = structural_similarity(img1, img2)
        
        # Convertir a [0, 1]
        similarity = (similarity + 1) / 2
        
        return similarity
    
    # ============ HOG (HISTOGRAM OF ORIENTED GRADIENTS) ============
    
    def extract_hog(self, img):
        """
        Extrae características HOG de una imagen
        
        Args:
            img: Imagen en escala de grises
        Returns:
            Vector de características HOG
        """
        features = hog(
            img,
            orientations=9,           # 9 direcciones de gradiente
            pixels_per_cell=(8, 8),   # Celdas de 8x8
            cells_per_block=(2, 2),   # Bloques de 2x2 celdas
            block_norm='L2-Hys',      # Normalización
            visualize=False
        )
        return features
    
    def hog_distance(self, img1, img2):
        """
        Distancia entre características HOG
        
        Args:
            img1, img2: Imágenes del mismo tamaño
        Returns:
            Distancia (menor = más similar)
        """
        # Extraer HOG
        hog1 = self.extract_hog(img1)
        hog2 = self.extract_hog(img2)
        
        # Distancia euclidiana
        distance = euclidean(hog1, hog2)
        
        return distance
    
    def hog_similarity(self, img1, img2, max_distance=50):
        """
        Similitud basada en HOG
        
        Args:
            img1, img2: Imágenes
            max_distance: Distancia máxima esperada (para normalizar)
        Returns:
            Similitud [0, 1]
        """
        distance = self.hog_distance(img1, img2)
        similarity = 1 - min(distance / max_distance, 1)
        return similarity
    
    # ============ MOMENTOS DE HU ============
    
    def extract_hu_moments(self, img):
        """
        Extrae momentos de Hu (invariantes a escala, rotación, traslación)
        
        Args:
            img: Imagen en escala de grises
        Returns:
            Vector de 7 momentos de Hu
        """
        # Calcular momentos
        moments = cv2.moments(img)
        
        # Momentos de Hu
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Transformación logarítmica para normalizar
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return hu_moments
    
    def hu_moments_distance(self, img1, img2):
        """
        Distancia entre momentos de Hu
        
        Args:
            img1, img2: Imágenes
        Returns:
            Distancia (menor = más similar)
        """
        hu1 = self.extract_hu_moments(img1)
        hu2 = self.extract_hu_moments(img2)
        
        distance = euclidean(hu1, hu2)
        return distance
    
    def hu_moments_similarity(self, img1, img2, max_distance=5):
        """
        Similitud basada en momentos de Hu
        
        Args:
            img1, img2: Imágenes
            max_distance: Distancia máxima esperada
        Returns:
            Similitud [0, 1]
        """
        distance = self.hu_moments_distance(img1, img2)
        similarity = 1 - min(distance / max_distance, 1)
        return similarity
    
    # ============ CONTORNOS ============
    
    def contour_similarity(self, img1, img2):
        """
        Similitud de contornos usando cv2.matchShapes
        
        Args:
            img1, img2: Imágenes binarias
        Returns:
            Similitud [0, 1] (mayor = más similar)
        """
        # Encontrar contornos
        contours1, _ = cv2.findContours(cv2.bitwise_not(img1), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(cv2.bitwise_not(img2), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours1 or not contours2:
            return 0.0
        
        # Usar el contorno más grande
        cnt1 = max(contours1, key=cv2.contourArea)
        cnt2 = max(contours2, key=cv2.contourArea)
        
        # Comparar contornos (método Hu)
        distance = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0)
        
        # Convertir a similitud
        similarity = 1 / (1 + distance)
        
        return similarity
    
    # ============ HISTOGRAMA ============
    
    def histogram_similarity(self, img1, img2):
        """
        Similitud de histogramas
        
        Args:
            img1, img2: Imágenes
        Returns:
            Similitud [0, 1]
        """
        # Calcular histogramas
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        
        # Normalizar
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # Correlación de histogramas
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return max(0, similarity)  # Asegurar [0, 1]


if __name__ == '__main__':
    # Prueba del comparador
    print("="*60)
    print("PRUEBA DE FEATURE COMPARATOR")
    print("="*60)
    
    # Crear dos imágenes de ejemplo
    img1 = np.ones((32, 32), dtype=np.uint8) * 255
    cv2.putText(img1, 'A', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    
    img2 = np.ones((32, 32), dtype=np.uint8) * 255
    cv2.putText(img2, 'A', (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    
    img3 = np.ones((32, 32), dtype=np.uint8) * 255
    cv2.putText(img3, 'B', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    
    # Crear comparador
    comparator = FeatureComparator()
    
    print("\nComparando 'A' con 'A' (similar):")
    print(f"  Template Matching: {comparator.template_matching(img1, img2):.4f}")
    print(f"  SSIM: {comparator.ssim_similarity(img1, img2):.4f}")
    print(f"  HOG Similarity: {comparator.hog_similarity(img1, img2):.4f}")
    print(f"  Hu Moments: {comparator.hu_moments_similarity(img1, img2):.4f}")
    
    print("\nComparando 'A' con 'B' (diferente):")
    print(f"  Template Matching: {comparator.template_matching(img1, img3):.4f}")
    print(f"  SSIM: {comparator.ssim_similarity(img1, img3):.4f}")
    print(f"  HOG Similarity: {comparator.hog_similarity(img1, img3):.4f}")
    print(f"  Hu Moments: {comparator.hu_moments_similarity(img1, img3):.4f}")
    
    print("\n✓ Comparador funcionando correctamente")