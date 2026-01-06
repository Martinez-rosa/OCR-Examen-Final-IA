# Sistema OCR Híbrido (CNN + Template Matching)
**Autor:** Maura Martinez

## 1. Introducción

Este proyecto presenta una solución avanzada de Reconocimiento Óptico de Caracteres (OCR) capaz de procesar tanto texto **digital** como **manuscrito**. A diferencia de los enfoques tradicionales puramente algorítmicos, este sistema implementa una arquitectura híbrida que combina la robustez de las **Redes Neuronales Convolucionales (CNN)** con la interpretabilidad del **Template Matching**.

El sistema ha sido diseñado para ser flexible, permitiendo el entrenamiento automático de modelos personalizados a partir de una estructura de directorios de plantillas, y ofrece una interfaz web moderna para una fácil interacción.

### Características Principales
- **Doble Modalidad**: Modelos especializados para texto digital (fuentes de computadora) y texto manuscrito.
- **Deep Learning**: Uso de TensorFlow/Keras para entrenar clasificadores de caracteres robustos.
- **Interfaz Web**: Aplicación Flask con frontend HTML5/CSS3 moderno para carga y visualización inmediata.
- **Segmentación Inteligente**: Algoritmos de visión artificial para detectar líneas y separar caracteres antes del reconocimiento.
- **Entrenamiento Automático**: Script dedicado que escanea carpetas de plantillas y genera modelos `.h5` listos para usar.

---

## 2. Arquitectura del Sistema

El proyecto se estructura en módulos modulares que separan la lógica de interfaz, procesamiento y reconocimiento.

### Componentes Clave

1.  **`gui.py` (Backend & Frontend)**
    - Servidor web basado en **Flask**.
    - Sirve la interfaz de usuario (HTML/CSS/JS embebido).
    - Expone el endpoint `/ocr` para recibir imágenes y devolver texto procesado en formato JSON.

2.  **`recognition.py` (Motor de Reconocimiento)**
    - **Clase `CNNRecognizer`**: El núcleo inteligente. Carga modelos entrenados (`.h5`) y realiza la predicción de caracteres.
    - **Gestión de Modelos**: Maneja la carga dinámica de `model_digital.h5` o `model_manuscrito.h5` según la solicitud del usuario.
    - **Persistencia**: Utiliza `pickle` para guardar y cargar los mapeos de etiquetas (clases de caracteres) asegurando consistencia entre entrenamiento e inferencia.
    - **Fallback**: Mantiene la lógica de `TemplateMatcher` (correlación) como respaldo o para validación.

3.  **`ocr_system.py` (Orquestador)**
    - Controlador principal que integra la segmentación y el reconocimiento.
    - Decide qué modelo (CNN) utilizar basándose en el parámetro `text_type`.
    - Procesa la imagen completa: segmenta página -> líneas -> caracteres -> reconoce -> ensambla texto.

4.  **`segmentation.py` (Visión Artificial)**
    - Responsable de "entender" la imagen antes de leerla.
    - **Binarización**: Convierte imágenes a blanco y negro usando umbralización de Otsu.
    - **Detección de Líneas**: Usa histogramas de proyección y morfología para aislar renglones.
    - **Segmentación de Caracteres**: Recorta cada letra individualmente para alimentarla a la CNN.

5.  **`train_model.py` (Entrenamiento)**
    - Script de utilidad para generar los modelos de IA.
    - Recorre la carpeta `templates/`, preprocesa las imágenes (normalización 32x32) y entrena las CNNs.
    - Guarda los modelos en `models/` automáticamente.

### Estructura de Carpetas

```
Examen Final IA/
├── models/                  # Almacenamiento de modelos entrenados (.h5) y mapeos (.pkl)
├── templates/               # Datos de entrenamiento
│   ├── digital/             # Imágenes de caracteres de fuentes digitales
│   └── manuscrito/          # Imágenes de caracteres manuscritos
├── gui.py                   # Servidor Web (Punto de entrada principal)
├── ocr_system.py            # Lógica de orquestación
├── recognition.py           # Definición de CNN y reconocedores
├── segmentation.py          # Algoritmos de procesamiento de imagen
├── train_model.py           # Script de re-entrenamiento
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Documentación
```

---

## 3. Instalación y Requisitos

Para ejecutar este sistema, necesitas Python instalado y las siguientes librerías.

### Requisitos Previos
- Python 3.8 o superior.

### Instalación de Dependencias
Ejecuta el siguiente comando para instalar todas las librerías necesarias:

```bash
pip install -r requirements.txt
```

**Nota**: Asegúrate de que `requirements.txt` incluya:
- `flask`
- `tensorflow`
- `opencv-python`
- `numpy`

---

## 4. Guía de Uso

### 1. Iniciar la Aplicación Web (Recomendado)
Esta es la forma más sencilla de usar el sistema.

1.  Abre una terminal en la carpeta del proyecto.
2.  Ejecuta el servidor:
    ```bash
    python gui.py
    ```
    *(O `py gui.py` en Windows)*
3.  Abre tu navegador web y ve a: `http://127.0.0.1:5000`
4.  **Uso**:
    - Haz clic en el área de carga o arrastra una imagen.
    - Selecciona el tipo de texto: **"Texto digital"** o **"Texto manuscrito"**.
    - Presiona **"Procesar"**.
    - El texto reconocido aparecerá en el panel de resultados.

### 2. Entrenar o Actualizar Modelos
Si añades nuevas imágenes a la carpeta `templates/`, debes reentrenar los modelos para que el sistema aprenda las nuevas formas.

1.  Organiza tus nuevas imágenes en `templates/digital/{caracter}/` o `templates/manuscrito/{caracter}/`.
2.  Ejecuta el script de entrenamiento:
    ```bash
    python train_model.py
    ```
3.  El script generará nuevos archivos `model_digital.h5` y `model_manuscrito.h5` en la carpeta `models/`.
4.  Reinicia la aplicación web para cargar los nuevos modelos.

---

## 5. Detalles Técnicos del Modelo CNN

El sistema utiliza una Red Neuronal Convolucional (CNN) diseñada para ser eficiente y precisa en imágenes de caracteres de 32x32 píxeles.

**Arquitectura:**
1.  **Input**: Imagen en escala de grises (32x32x1).
2.  **Conv2D**: 32 filtros, kernel 3x3, activación ReLU.
3.  **MaxPooling2D**: Reducción de dimensionalidad 2x2.
4.  **Conv2D**: 64 filtros, kernel 3x3, activación ReLU.
5.  **MaxPooling2D**: Reducción de dimensionalidad 2x2.
6.  **Flatten**: Aplanado de características.
7.  **Dense**: 128 neuronas, activación ReLU (Capa oculta).
8.  **Dropout**: 0.3 (Para evitar sobreajuste).
9.  **Dense (Output)**: N neuronas (según número de clases), activación Softmax.

Esta arquitectura permite extraer características visuales complejas (bordes, curvas, trazos) y clasificarlas con alta precisión, superando las limitaciones del *Template Matching* simple que es sensible a pequeñas variaciones de tamaño o grosor.

---

## 6. Solución de Problemas Comunes

-   **"Error: No models found"**: Asegúrate de ejecutar `python train_model.py` al menos una vez para generar los archivos `.h5` necesarios.
-   **La interfaz no carga**: Verifica que el puerto 5000 no esté en uso.
-   **Resultados inexactos**:
    -   Asegúrate de seleccionar el modo correcto (Digital vs Manuscrito).
    -   Verifica que la imagen tenga buen contraste y esté bien iluminada.
    -   Considera añadir más ejemplos a la carpeta `templates/` y reentrenar.

---
*Examen Final de Inteligencia Artificial - Maura Martinez*
