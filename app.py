import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

# 1. Definimos la misma estructura de la red
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
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
        x = self.fc2(x)
        return x

# 2. Configuración de la interfaz web
st.title("🧠 Detector de Emociones Faciales")
st.write("Sube una foto y nuestra IA intentará adivinar la emoción.")

# Cargar el modelo guardado
@st.cache_resource
def cargar_modelo():
    model = EmotionCNN()
    model.load_state_dict(torch.load('modelo_emociones.pth'))
    model.eval()
    return model

modelo = cargar_modelo()
mapa_emociones = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# 3. Lógica para subir archivos
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L') # Convertir a gris
    st.image(image, caption='Imagen subida', use_container_width=True)
    
    # Preprocesamiento idéntico al entrenamiento
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (48, 48))
    img_tensor = torch.FloatTensor(img_resized).reshape(1, 1, 48, 48) / 255.0
    
    # Predicción
    with torch.no_grad():
        output = modelo(img_tensor)
        _, pred = torch.max(output, 1)
        emocion = mapa_emociones[pred.item()]
    
    st.success(f"La emoción detectada es: **{emocion}**")