import urllib.request
import os
import sys

def download_east_model():
    url = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
    models_dir = "models"
    file_name = "frozen_east_text_detection.pb"
    file_path = os.path.join(models_dir, file_name)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Creado directorio: {models_dir}")

    if os.path.exists(file_path):
        print(f"El modelo ya existe en: {file_path}")
        return

    print(f"Descargando modelo EAST desde {url}...")
    print("Esto puede tardar unos minutos dependiendo de tu conexión (aprox 96MB)...")
    
    try:
        def progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDescargando: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, file_path, reporthook=progress)
        print("\n¡Descarga completada!")
        
        # Verificar tamaño
        size = os.path.getsize(file_path)
        print(f"Tamaño del archivo: {size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"\n[ERROR] Falló la descarga: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    download_east_model()
