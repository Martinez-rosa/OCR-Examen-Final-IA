import os
import tensorflow as tf
from recognition import CNNRecognizer

# Verificar existencia de modelos
model_dir = "models"
if not os.path.exists(model_dir):
    print(f"Creando directorio {model_dir}")
    os.makedirs(model_dir)

print("Iniciando entrenamiento forzado de modelos...")

# 1. Modelo Digital
print("\n--- Entrenando Modelo Digital ---")
try:
    # Ajusta el path de templates según tu estructura
    # Asumimos que templates/digital existe
    if os.path.exists("templates/digital"):
        rec_digital = CNNRecognizer(templates_dir="templates/digital", model_path="models/model_digital.h5")
        rec_digital.train_from_templates(epochs=5)
    else:
        print("WARN: No existe templates/digital. Saltando entrenamiento digital.")
except Exception as e:
    print(f"Error entrenando modelo digital: {e}")

# 2. Modelo Manuscrito
print("\n--- Entrenando Modelo Manuscrito ---")
try:
    if os.path.exists("templates/manuscrito"):
        rec_manuscrito = CNNRecognizer(templates_dir="templates/manuscrito", model_path="models/model_manuscrito.h5")
        rec_manuscrito.train_from_templates(epochs=10) # Manuscrito suele requerir más épocas
    else:
        print("WARN: No existe templates/manuscrito. Saltando entrenamiento manuscrito.")
except Exception as e:
    print(f"Error entrenando modelo manuscrito: {e}")

print("\nEntrenamiento finalizado.")
