import os
from ultralytics import YOLO

# 1. Cargamos el modelo
model = YOLO("yolov8s.pt")

# 2. Entrenamos (usamos la ruta relativa al archivo que acabas de guardar)
# Importante: El archivo 'dataset.yaml' debe estar DENTRO de la carpeta 'dataset4'
yaml_path = "dataset4/dataset.yaml"

print("--- Iniciando entrenamiento final ---")

try:
    model.train(
        data=yaml_path,
        epochs=10,
        imgsz=320,
        batch=4,
        device='cpu',
        name="entrenamiento_final",
        project="runs/train"
    )
except Exception as e:
    print(f"\n❌ ERROR: {e}")


