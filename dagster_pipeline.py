from dagster import asset, define_asset_job, Definitions, get_dagster_logger
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report

EMOCIONES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MUESTRAS_POR_CLASE = 50


class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@asset(description="Carga una muestra estratificada del conjunto de test (50 imágenes por emoción = 350 total)")
def datos_test():
    logger = get_dagster_logger()
    muestras = {}
    for emocion in EMOCIONES:
        ruta = os.path.join('test', emocion)
        if os.path.exists(ruta):
            archivos = sorted(os.listdir(ruta))[:MUESTRAS_POR_CLASE]
            muestras[emocion] = [os.path.join(ruta, f) for f in archivos]
            logger.info(f"  {emocion}: {len(muestras[emocion])} imágenes")
    total = sum(len(v) for v in muestras.values())
    logger.info(f"Asset 'datos_test' completado — {total} imágenes totales")
    return muestras


@asset(description="Evalúa el modelo CNN sobre la muestra de test y calcula accuracy, F1 y reporte por clase")
def evaluacion_cnn(datos_test):
    logger = get_dagster_logger()
    modelo = EmotionCNN()
    modelo.load_state_dict(torch.load('modelo_emociones.pth', map_location='cpu', weights_only=True))
    modelo.eval()
    logger.info("Modelo CNN cargado desde 'modelo_emociones.pth'")

    y_true, y_pred = [], []
    for emocion, rutas in datos_test.items():
        idx_real = EMOCIONES.index(emocion)
        for ruta in rutas:
            try:
                img = Image.open(ruta).convert('L').resize((48, 48))
                tensor = torch.FloatTensor(np.array(img) / 255.0).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    pred = torch.argmax(modelo(tensor), dim=1).item()
                y_true.append(idx_real)
                y_pred.append(pred)
            except Exception:
                continue

    acc = round(accuracy_score(y_true, y_pred) * 100, 2)
    f1 = round(f1_score(y_true, y_pred, average='macro'), 3)
    report = classification_report(y_true, y_pred, target_names=EMOCIONES, output_dict=True)

    logger.info(f"Asset 'evaluacion_cnn' completado — Accuracy: {acc}% | F1 Macro: {f1}")
    return {'accuracy': acc, 'f1_macro': f1, 'report': report, 'n_imagenes': len(y_true)}


@asset(description="Combina métricas reales (CNN) con resultados históricos y actualiza tabla_comparativa_modelos.csv")
def tabla_comparativa_final(evaluacion_cnn):
    logger = get_dagster_logger()

    datos = {
        'Modelo':           ['CNN Base', 'VGG16', 'YOLOv8', 'ResNet-18'],
        'Épocas':           [10, 10, 10, 15],
        'Train Loss Final': [0.8197, 1.568, 0.8323, 0.0074],
        'Test Loss Final':  [1.16, 1.91, 0.8341, 0.871],
        'Accuracy (%)':     [str(evaluacion_cnn['accuracy']), '~48.00', '61.00 (mAP@50)', '61.00'],
        'F1 Macro avg':     [str(evaluacion_cnn['f1_macro']), '~0.42', '~0.58', '0.58'],
        'Mejor clase':      ['Happy (F1: 0.68)', 'Happy (~0.65)', 'Happy (mAP: 0.942)', 'Happy (F1: 0.81)'],
        'Peor clase':       ['Disgust (F1: 0.28)', 'Disgust (~0.20)', 'Disgust (mAP: 0.224)', 'Disgust (F1: 0.35)'],
        'Conclusión': [
            'Modelo base sólido, limitado por arquitectura simple',
            'Descartado — input mismatch 48x48 vs 224x224 provoca overfitting severo',
            'Transfer Learning: Train ≈ Val Loss, sin overfitting',
            'Mejor modelo: skip connections preservan rasgos en baja resolución',
        ],
    }

    df = pd.DataFrame(datos)
    df.to_csv('tabla_comparativa_modelos.csv', index=False, encoding='utf-8')
    logger.info(f"Asset 'tabla_comparativa_final' completado — CSV actualizado con CNN accuracy={evaluacion_cnn['accuracy']}%")
    return df


pipeline_evaluacion = define_asset_job("pipeline_evaluacion", selection="*")

defs = Definitions(
    assets=[datos_test, evaluacion_cnn, tabla_comparativa_final],
    jobs=[pipeline_evaluacion],
)
