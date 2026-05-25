# 🧠 Reconocimiento de Emociones Faciales con Deep Learning

Proyecto final de Inteligencia Artificial para la clasificación automática de imágenes en 7 estados emocionales: **Angry, Disgust, Fear, Happy, Neutral, Sad y Surprise**. Combina ingeniería de datos, análisis estadístico en R y Python, procesamiento Big Data con Dask y Deep Learning para clasificar expresiones faciales mediante una aplicación web interactiva desplegada en la nube.

🌐 **App en producción:** [detectoremociones.streamlit.app](https://detectoremociones.streamlit.app)

---

## 📋 Objetivo del Proyecto

Desarrollar un sistema de visión artificial capaz de clasificar emociones humanas a partir de imágenes faciales, evaluando y comparando cuatro arquitecturas de Deep Learning de complejidad creciente, e integrando el proceso completo en una aplicación web interactiva.

---

## ⚠️ Problema que Resuelve

La clasificación manual de emociones en grandes volúmenes de datos es ineficiente y subjetiva. La baja resolución de las imágenes faciales (48×48 píxeles) dificulta la detección de microexpresiones, lo que convierte este problema en un reto técnico real para cualquier arquitectura de Deep Learning.

---

## 📊 Dataset: FER-2013

| Característica | Detalle |
|---|---|
| Volumen | 28.709 imágenes de entrenamiento |
| Resolución | 48×48 píxeles en escala de grises |
| Clases | 7 emociones (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) |
| Desequilibrio | Happy (25.1%) vs Disgust (1.5%) → ratio 17:1 |
| Fuente | Kaggle FER-2013 Challenge |

---

## 🏗️ Evolución de Arquitecturas

| Modelo | Precisión | Observaciones |
|---|---|---|
| CNN Base | 51.35% | Arquitectura propia de 3 capas convolucionales |
| VGG16 | ~48% | Descartado: input mismatch 224×224 vs 48×48 |
| YOLOv8s | 61.0% mAP@50 | Transfer Learning, sin overfitting |
| **ResNet-18** | **61.0%** | Mejor F1 macro (0.58), skip connections |

---

## 🗂️ Estructura del Repositorio

```
Proyecto_final_IA/
│
├── app.py                        # Aplicación web Streamlit 
├── requirements.txt              # Dependencias Python para producción
├── runtime.txt                   # Versión de Python para Streamlit Cloud
│
├── streamlit/                    # Recursos cargados por la app en producción
│   ├── modelo_emociones.pth      # Pesos del modelo CNN entrenado
│   ├── best.pt                   # Pesos del modelo YOLOv8 (mejor época)
│   ├── datos_emociones.csv       # Metadatos del dataset (label, label_encoded)
│   ├── datos_eda_r.csv           # CSV usado en el análisis R/Python
│   ├── matriz_diferencias_r.csv  # Matriz outer() generada por R
│   ├── tabla_comparativa_modelos.csv
│   ├── comparativa_loss_modelos.png
│   ├── matrizConfusion_modelo4.png
│   ├── imagen_ejemplo_modelo3.png
│   ├── boxplot_emociones.png
│   ├── dispersion_categorias.jpeg
│   └── sample_train/             # Muestra de imágenes por emoción 
│       ├── angry/   (5 imágenes)
│       ├── disgust/ (5 imágenes)
│       ├── fear/    (5 imágenes)
│       ├── happy/   (5 imágenes)
│       ├── neutral/ (5 imágenes)
│       ├── sad/     (5 imágenes)
│       └── surprise/(5 imágenes)
│
├── notebooks/
│   ├── analisis_exploratorio.ipynb   # EDA completo + interoperabilidad R/Python
│   └── fer2013.ipynb                 # Entrenamiento CNN y ResNet-18 (PyTorch)
│
├── analisis_r/
│   ├── analisis_estadistico.R    # summary(), chisq.test(), dist(), ggplot2
│   ├── complemento_en_R.R        # Histogramas y gráficos de dispersión
│   └── r.ipynb                   # Ejecución de R desde Jupyter con rpy2
│
├── scripts/
│   ├── yolov8.py                 # Entrenamiento YOLOv8 con Ultralytics
│   ├── crear_etiquetas.py        # Generación de archivos .txt para YOLO
│   ├── arreglar_carpetas.py      # Reestructuración dataset al formato Ultralytics
│   └── graficar_resultados.py    # Generación de gráficas comparativas
│
├── runs/                         # Artefactos generados por YOLOv8 (matrices, curvas PR)
│   └── detect/runs/
│       ├── train/entrenamiento_final3/   # Resultados de entrenamiento
│       ├── val/                          # Resultados de validación
│       └── test/                         # Resultados de test
│
├── modelos/
│   ├── yolov8s.pt                # Pesos preentrenados YOLOv8s (base)
│   ├── yolov8n.pt                # Pesos preentrenados YOLOv8n
│   └── yolov8n-cls.pt            # Pesos preentrenados YOLOv8n clasificación
│
├── graficas/                     # Gráficas exportadas del proceso de entrenamiento
│
└── datasets/
    ├── fer2013.zip               # Dataset original comprimido
    └── runs.zip                  # Resultados YOLO comprimidos
```

> **Nota:** Los directorios `/train/` y `/test/` con el dataset completo (28.709 imágenes) no se incluyen en el repositorio por su tamaño. La app usa `streamlit/sample_train/` como muestra representativa en la versión cloud.

---

## 🔧 Instalación y Reproducibilidad

Este proyecto utiliza `pip` y el estándar `requirements.txt` para la gestión eficiente y determinista de entornos y dependencias.

### 1. Clonar el repositorio

```bash
git clone https://github.com/martabachillerato/Proyecto_final_ia.git
cd Proyecto_final_ia
```

### 2. Crear el entorno virtual

```bash
python -m venv .venv
```

### 3. Activar el entorno

- **Windows:** `.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

### 4. Instalar dependencias

Para instalar las dependencias base y ejecutar la aplicación web (Producción):

```bash
pip install -r requirements.txt
```

> *Nota:* Se recomienda **Python 3.11** (especificado en `runtime.txt`). Con versiones superiores algunas dependencias pueden requerir ajustes.

### 5. Dataset

Los pesos de los modelos ya están incluidos en el repositorio. Para **reproducir el entrenamiento completo**, descarga el dataset [FER-2013 de Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) y coloca las carpetas en la raíz del proyecto:

```
Proyecto_final_IA/
├── train/
└── test/
```

---

## 🚀 Ejecución de la Aplicación

Para lanzar la interfaz interactiva de clasificación de emociones:

```bash
streamlit run app.py
```

## 📓 Notebooks

Para ejecutar los notebooks de análisis y entrenamiento:

```bash
jupyter notebook notebooks/analisis_exploratorio.ipynb  # EDA + interoperabilidad R/Python
jupyter notebook notebooks/fer2013.ipynb                # Entrenamiento CNN y ResNet-18
```

> *Nota:* El notebook `analisis_exploratorio.ipynb` requiere **R instalado** y la librería `rpy2` para los bloques de interoperabilidad R/Python.

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Herramientas |
|---|---|
| Deep Learning | PyTorch, Ultralytics YOLOv8 |
| Visión Artificial | OpenCV, PIL/Pillow |
| Análisis de Datos | Pandas, NumPy, Dask |
| Análisis Estadístico | R (ggplot2, stats), rpy2 |
| Visualización | Matplotlib |
| Aplicación Web | Streamlit |
| Despliegue | Streamlit Cloud |
| Control de Versiones | Git / GitHub |
