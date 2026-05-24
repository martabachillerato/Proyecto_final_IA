import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el archivo que subiste
df = pd.read_csv('runs/detect/runs/train/entrenamiento_final3/results.csv')

# 2. Configurar la gráfica de Loss
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train/cls_loss'], label='Train Loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='Val Loss')
plt.title('Evolución de la Función de Pérdida')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('grafica_loss.png') # Guarda la imagen

# 3. Configurar la gráfica de Métricas
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precisión')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50')
plt.title('Evolución de Métricas de Rendimiento')
plt.xlabel('Época')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.savefig('grafica_metricas.png') # Guarda la imagen