# 🧠 Reconocimiento de Emociones Faciales con Deep Learning

Proyecto final de Inteligencia Artificial para la clasificación de imágenes en 7 estados emocionales utilizando diferentes modelos: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

## 📋 Análisis del Dataset
**Volumen** : 28.709 imágenes en escala de grises.
**Resolución** : 48x48 píxeles por imagen.
**Desequilibrio de clases** : Mediante un análisis estadístico realizado en lenguaje R, se identificó un sesgo significativo hacia las clases Happy y Neutral, lo que puede condicionar la estrategia del entrenamiento.

## 🛠️ Tecnologías utilizadas
· **Core** : Python y PyTorch (Modelado de Deep Learning).
· **Análisis** : R y ggplot2 (Análisis estadístico de sesgos).
· **Despliegue** : Streamlit (Aplicación web interactiva con el procedimiento y resultado).
· **Visión Artificial** : OpenCV (Preprocesamiento de imágenes).

## 🚀 Instrucciones de uso local
1. Clonar el repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar la aplicación: `python -m streamlit run app.py`

## 🛠️ Estructura del repositorio
* `analisis_exploratorio.ipynb`: Cuaderno con todo el proceso de limpieza y entrenamiento.
* `app.py`: Código de la aplicación web interactiva.
* `modelo_emociones.pth`: Pesos del modelo entrenado.
* `requirements.txt`: Librerías necesarias para ejecutar el proyecto.

## 🏗️ Evolución de Arquitecturas
Durante el desarrollo se evaluaron cuatro enfoques distintos para intentar maximizar la precisión:
1. **CNN** : Modelo simple de 3 capas convolucionales. Logró una precisión del **51.35%**.
2. **YOLO8** : Obteniendo una precisión del **61%**, este modelo está diseñado para la eficiencia, permitiendo la clasificación de emociones en un solo paso, optimizado para alto rendimiento y baja latencia. 
3. **VGG16** : Implementación de una arquitectura profunda pre-entrenada. Se analizó su baja precisión inicial (**25.40%**) debido a la discrepancia entre la resolución inicial (224x224) y nuestro dataset (48x48).
4. **ResNet-18** : Modelo final adaptado para imágenes de baja resolución mediante la eliminación de capas de reducción agresiva y la implementación de **Data Augmentation** para mitigar el desequilibrio de clases. 
