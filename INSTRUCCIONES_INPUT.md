# Guía de Preparación de Imágenes para OCR

Para obtener los mejores resultados con este sistema (Detector EAST + Template Matching), por favor prepara tus imágenes de prueba siguiendo estas especificaciones.

## Requisitos Generales (Para todas las imágenes)
- **Formato**: PNG (recomendado) o JPG de alta calidad.
- **Resolución**: Mínimo 1000px de ancho. Escaneos a 300 DPI son ideales.
- **Orientación**: El texto debe estar horizontal (0° de rotación). El sistema no endereza imágenes torcidas.
- **Fondo**: Blanco limpio. Evita sombras, arrugas o manchas (especialmente si tomas fotos con celular).
- **Fuente**: Letra de molde (Arial, Times, etc.). No manuscrita.

---

## Especificaciones para tus 3 Imágenes de Prueba

### Imagen 1: Foto y Texto
**Objetivo**: Verificar que el sistema distingue entre bloques de texto y fotografías.
**Contenido**:
- Incluye un párrafo de texto claro (4-5 líneas).
- Inserta una fotografía o imagen (ej. un logo, un paisaje, una foto de persona) separada del texto.
- **Importante**: Deja al menos 20px de espacio en blanco entre el texto y la foto.
- **Nombre sugerido**: `test_foto_texto.png`

### Imagen 2: Texto Solo
**Objetivo**: Verificar la precisión pura del reconocimiento de caracteres (OCR).
**Contenido**:
- Solo texto, sin imágenes ni tablas.
- Usa diferentes tamaños de letra si quieres probar la robustez (ej. un Título grande y párrafos normales).
- Incluye números y letras mezclados para probar la corrección automática (ej. "Año 2023", "Código A6B5").
- **Nombre sugerido**: `test_texto_solo.png`

### Imagen 3: Texto + Imágenes + Tablas (Compleja)
**Objetivo**: Probar la segmentación completa de documentos estructurados.
**Contenido**:
- **Texto**: Algún título o párrafo introductorio.
- **Tabla**: Incluye una tabla con bordes visibles (líneas negras).
    - *Nota*: El sistema detecta tablas buscando líneas verticales y horizontales largas. Si la tabla no tiene bordes, podría fallar.
- **Imagen**: Alguna figura o gráfico pequeño.
- **Distribución**: Intenta que no se superpongan (ej. Texto arriba, Tabla al medio, Imagen abajo).
- **Nombre sugerido**: `test_complejo.png`

---

## Formato de Salida Esperado

El sistema generará automáticamente en la carpeta `output/`:
1. **Texto**: Archivo `.txt` con todo el contenido reconocido.
2. **Imágenes**: Archivos `.png` individuales para cada foto detectada.
3. **Tablas**:
    - Archivo `.csv` con los datos de la tabla (abrible en Excel).
    - Archivo `.png` con el recorte visual de la tabla.

## Ejecución
Para procesar las imágenes una vez las tengas listas:

```bash
python main.py --image "test_foto_texto.png"
python main.py --image "test_texto_solo.png"
python main.py --image "test_complejo.png"
```
