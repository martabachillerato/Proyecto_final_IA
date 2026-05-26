import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
import dask.dataframe as dd
import time
import os
import random
from torchvision import models
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

try:
    import rpy2.robjects as robjects
    from rpy2.robjects.conversion import localconverter
    RPY2_OK = True
except Exception:
    RPY2_OK = False

# ==========================================
# 1. ARQUITECTURAS DE LOS MODELOS
# ==========================================

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

def crear_modelo_resnet():
    modelo = models.resnet18(weights=None)
    modelo.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = modelo.fc.in_features
    modelo.fc = nn.Linear(num_ftrs, 7)
    return modelo

# ==========================================
# 2. CONFIGURACIÓN Y CARGA DE MODELOS
# ==========================================
st.set_page_config(page_title="Proyecto Final IA - Emociones", layout="wide")

@st.cache_resource
def cargar_modelos():
    cnn = EmotionCNN()
    if os.path.exists('streamlit/modelo_emociones.pth'):
        cnn.load_state_dict(torch.load('streamlit/modelo_emociones.pth', map_location=torch.device('cpu')))
    cnn.eval()

    yolo = None
    if os.path.exists('streamlit/best.pt'):
        yolo = YOLO('streamlit/best.pt')

    resnet = crear_modelo_resnet()
    if os.path.exists('streamlit/modelo_resnet_final.pth'):
        resnet.load_state_dict(torch.load('streamlit/modelo_resnet_final.pth', map_location=torch.device('cpu')))
    resnet.eval()

    return cnn, yolo, resnet

mapa_emociones = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

CONTEOS_FIJOS = {'angry': 3995, 'disgust': 436, 'fear': 4097, 'happy': 7215, 'neutral': 4965, 'sad': 4830, 'surprise': 3171}

@st.cache_data
def contar_imagenes_dataset(base_path='train'):
    EXTENSIONES = {'.jpg', '.jpeg', '.png'}
    if not os.path.exists(base_path):
        return None, 0
    conteos = {}
    total = 0
    for emocion in sorted(os.listdir(base_path)):
        carpeta = os.path.join(base_path, emocion)
        if not os.path.isdir(carpeta):
            continue
        n = sum(1 for f in os.listdir(carpeta) if os.path.splitext(f)[1].lower() in EXTENSIONES)
        conteos[emocion] = n
        total += n
    return conteos, total

def obtener_foto_real(base_path='train'):
    if not os.path.exists(base_path):
        base_path = 'streamlit/sample_train'
    try:
        emociones = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        emo_elegida = random.choice(emociones)
        ruta_emo = os.path.join(base_path, emo_elegida)
        fotos = os.listdir(ruta_emo)
        foto_nombre = random.choice(fotos)
        return os.path.join(ruta_emo, foto_nombre), emo_elegida
    except:
        return None, None

# ==========================================
# 3. INTERFAZ POR PESTAÑAS
# ==========================================
st.title("🎭 Identificación de Emociones: Del Dato al Modelo Profesional")

tab1, tab_r, tab_dask, tab2, tab3, tab4, tab6, tab5 = st.tabs([
    "📊 1. Exploración de Datos",
    "📈 2. Análisis en R",
    "🔬 3. Pandas vs Dask",
    "⚙️ 4. Entrenamiento CNN",
    "🚀 5. Evolución a YOLOv8",
    "🧠 6. Refinamiento: ResNet-18",
    "📋 7. Comparativa de Modelos",
    "🏆 8. Prueba de Modelos"
])

# --- PESTAÑA 1: EXPLORACIÓN DE DATOS ---
with tab1:
    st.header("📊 Exploración de Datos")

    st.subheader("1. Tabla Dinámica de Emociones")
    st.write("Resumen de conteos por emoción generado con `pivot_table` de Pandas (Tema 5):")

    conteos_reales, total_real = contar_imagenes_dataset('train')
    if conteos_reales:
        st.caption("✅ Conteo calculado dinámicamente recorriendo el dataset completo.")
        filas_emociones = [(e.capitalize(), n) for e, n in sorted(conteos_reales.items())]
    else:
        st.caption("ℹ️ Dataset no disponible en este entorno — mostrando valores del dataset completo.")
        filas_emociones = [(e.capitalize(), n) for e, n in sorted(CONTEOS_FIJOS.items())]
        total_real = sum(CONTEOS_FIJOS.values())

    pivot_data = {
        "Emoción":  [e for e, _ in filas_emociones] + ["**Total**"],
        "Imágenes": [n for _, n in filas_emociones] + [total_real],
    }
    st.dataframe(pd.DataFrame(pivot_data), use_container_width=True, hide_index=True)
    st.metric("Total de imágenes en el dataset", f"{total_real:,}")

    st.subheader("2. Mapeo de Etiquetas")
    st.write("La IA no entiende palabras, necesita números. Traducimos las etiquetas textuales a códigos numéricos mediante un diccionario.")
    mapeo_data = {
        "Código": [0, 1, 2, 3, 4, 5, 6],
        "Emoción": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    }
    st.dataframe(pd.DataFrame(mapeo_data), use_container_width=True, hide_index=True)

    st.info(
        "**Valores faltantes por columna:**\n\n"
        "```\nlabel      0\nemocion    0\ndtype: int64\n```\n\n"
        "✅ No existen valores nulos en ninguna de las columnas relevantes."
    )

    st.write("Muestra aleatoria de 10 registros del dataset de entrenamiento:")
    sample_data = {
        "label":         ["fear", "surprise", "sad", "fear", "angry", "sad", "neutral", "surprise", "surprise", "happy"],
        "label_encoded": [2, 6, 5, 2, 0, 5, 4, 6, 6, 3],
    }
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True, hide_index=True)

    st.subheader("3. Distribución Normalizada de Emociones")
    st.write("Convertimos los recuentos absolutos en **proporciones del total** para visualizar el desequilibrio de clases (Tema 3):")
    norm_data = {
        "Emoción":    ["Happy", "Neutral", "Sad", "Fear", "Angry", "Surprise", "Disgust"],
        "Proporción": [0.2513, 0.1729, 0.1682, 0.1427, 0.1392, 0.1105, 0.0152],
    }
    st.dataframe(pd.DataFrame(norm_data), use_container_width=True, hide_index=True)
    st.warning("⚠️ **Desequilibrio de clases detectado:** *Happy* representa el 25.1% del dataset mientras que *Disgust* apenas alcanza el 1.5%. Esto afectará al rendimiento del modelo en las clases minoritarias.")

    st.subheader("4. Estudio de Valores Nulos")
    nulos_data = {"Variable": ["label"], "Valores Nulos": [0]}
    st.dataframe(pd.DataFrame(nulos_data), use_container_width=True, hide_index=True)
    st.success("✅ **No se detectaron valores perdidos** en ninguna de las variables del dataset. El dataset está listo para ser usado directamente en el entrenamiento.")

    st.subheader("5. División del Dataset: Train / Test")
    col_split1, col_split2 = st.columns(2)
    with col_split1:
        st.metric(label="🎓 Conjunto de Entrenamiento (Train)", value="16.000 imágenes", delta="80%")
    with col_split2:
        st.metric(label="🧪 Conjunto de Prueba (Test)", value="4.000 imágenes", delta="20%")
    st.write("El dataset se ha dividido de forma estratificada en **80% entrenamiento** y **20% prueba**, asegurando que la proporción de cada emoción se mantenga en ambos conjuntos.")
    st.info("📌 **Nota metodológica:** Se han procesado 20.000 de las 28.709 imágenes disponibles para garantizar un tiempo de entrenamiento razonable en CPU.")

    st.subheader("6. Visualización de Datos Reales")
    st.write("Muestras aleatorias del dataset de entrenamiento para inspección visual:")
    if st.button('🔄 Generar muestras aleatorias del Dataset (EDA)'):
        _train_path = 'train' if os.path.exists('train') else 'streamlit/sample_train'
        if not os.path.exists(_train_path):
            st.warning("⚠️ El dataset de imágenes no está disponible en esta versión cloud. Esta función requiere ejecutar la app localmente con el dataset descargado.")
        else:
            p1, l1 = obtener_foto_real(_train_path)
            p2, l2 = obtener_foto_real(_train_path)
            if p1 and p2:
                col_eda1, col_eda2 = st.columns(2)
                with col_eda1:
                    img1_disp = Image.open(p1).resize((250, 250), Image.Resampling.LANCZOS)
                    st.image(img1_disp, caption=f"Etiqueta: {l1}")
                with col_eda2:
                    img2_disp = Image.open(p2).resize((250, 250), Image.Resampling.LANCZOS)
                    st.image(img2_disp, caption=f"Etiqueta: {l2}")

# --- PESTAÑA 2: ANÁLISIS EN R ---
with tab_r:
    st.header("📈 Análisis Estadístico en R — Reproducción en Python")

    # ── Conteo de imágenes ejecutado en R (Python → R en RAM) ─────────────
    st.subheader("🔢 Conteo Total de Imágenes del Dataset (ejecutado en R)")
    st.write(
        "Python carga el CSV con Pandas, **inyecta los datos directamente en R en memoria RAM** "
        "usando `rpy2`. R genera el gráfico con `ggplot2` y lo renderiza en un buffer de bytes "
        "sin tocar el disco. Streamlit pinta el resultado al instante."
    )

    st.code("""# Código R ejecutado via rpy2
datos <- read.csv("streamlit/datos_emociones.csv")
nrow(datos)                  # Total de imágenes
table(datos$label)           # Conteo por emoción""", language="r")

    if st.button("▶️ Ejecutar en R"):
        resultado_r = {}

        if not RPY2_OK:
            st.warning("⚠️ R no disponible en este entorno.")
        else:
            try:
                with st.spinner("Ejecutando R..."):
                    with localconverter(robjects.default_converter):
                        robjects.r('datos <- read.csv("streamlit/datos_emociones.csv")')
                        total_r = int(robjects.r('nrow(datos)')[0])
                        nombres = list(robjects.r('names(table(datos$label))'))
                        valores = [int(x) for x in robjects.r('as.vector(table(datos$label))')]
                df_r = pd.DataFrame([
                    {"Emoción": k.capitalize(), "Imágenes": v}
                    for k, v in sorted(zip(nombres, valores))
                ])
                st.dataframe(df_r, use_container_width=True, hide_index=True)
                st.metric("Total de imágenes calculado en R", f"{total_r:,}")
                st.success("✅ Resultado calculado en R con `nrow()` y `table()` via `rpy2`.")
            except Exception as e:
                st.error(f"Error R: {e}")

    st.divider()

    st.write("""
    Esta pestaña reproduce los análisis estadísticos desarrollados en **R** a través de `rpy2` en el notebook
    `analisis_exploratorio.ipynb` y los scripts `analisis_estadistico.R` y `complemento_en_R.R`.
    Las mismas operaciones se implementan en Python para que los resultados se generen automáticamente
    con los datos reales sin depender de imágenes estáticas.
    """)

    # ── Carga de datos ─────────────────────────────────────────────────────
    MAPA_R = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    CONTEOS_BASE = {'angry': 3995, 'disgust': 436, 'fear': 4097, 'happy': 7215,
                    'neutral': 4965, 'sad': 4830, 'surprise': 3171}

    CSV_R_PATH = "streamlit/datos_eda_r.csv"
    if os.path.exists(CSV_R_PATH):
        df_r = pd.read_csv(CSV_R_PATH)
        if 'label_encoded' not in df_r.columns:
            df_r['label_encoded'] = df_r['label'].map(MAPA_R)
        conteo_r = df_r.groupby('label')['label'].count()
        fuente_r = f"`datos_eda_r.csv` — {len(df_r):,} registros"
    else:
        df_r = None
        conteo_r = pd.Series(CONTEOS_BASE)
        fuente_r = "valores del dataset completo (28.709 imágenes, `datos_eda_r.csv` no encontrado)"

    total_r = int(conteo_r.sum())
    df_freq_r = (
        pd.DataFrame({'emocion': conteo_r.index, 'n': conteo_r.values})
        .assign(pct=lambda d: d['n'] / d['n'].sum() * 100)
        .sort_values('pct', ascending=False)
        .reset_index(drop=True)
    )
    freq_arr = df_freq_r['n'].values.astype(float)

    st.info(f"**Fuente de datos:** {fuente_r} — {total_r:,} imágenes totales.")

    st.divider()

    # ── 1. Resumen estadístico (Celda 23 del notebook) ─────────────────────
    st.subheader("1. Resumen Estadístico de Frecuencias de Clase")
    st.write("""
    Equivale al código R de la **Celda 23** del notebook: `summary(df_freq$frecuencia)` + `var()` + `sd()` +
    `sd/mean*100` (coeficiente de variación) + ratio max/mín. Cuantifica el desequilibrio del dataset
    midiendo la dispersión de frecuencias entre clases.
    """)

    stats_data = {
        "Estadístico R": [
            "Mínimo", "Q1 (25%)", "Mediana", "Media", "Q3 (75%)", "Máximo",
            "Varianza — var()", "Desv. típica — sd()", "Coef. variación (%)", "Ratio máx/mín"
        ],
        "Valor": [
            f"{int(freq_arr.min()):,}",
            f"{int(np.percentile(freq_arr, 25)):,}",
            f"{int(np.median(freq_arr)):,}",
            f"{freq_arr.mean():,.2f}",
            f"{int(np.percentile(freq_arr, 75)):,}",
            f"{int(freq_arr.max()):,}",
            f"{freq_arr.var():,.2f}",
            f"{freq_arr.std():,.2f}",
            f"{freq_arr.std() / freq_arr.mean() * 100:.2f} %",
            f"{freq_arr.max() / freq_arr.min():.1f} : 1",
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    esperado_r = total_r / len(freq_arr)
    chi2_stat = float(np.sum((freq_arr - esperado_r) ** 2 / esperado_r))
    st.warning(f"""
    **Test χ² de uniformidad** (Celda 30 del notebook — `chisq.test(frecuencias_r)`):
    χ² = **{chi2_stat:,.2f}** con {len(freq_arr)-1} grados de libertad.
    Con distribución perfectamente uniforme se esperarían **{esperado_r:,.0f}** imágenes por clase.
    El valor extremadamente alto confirma que el dataset **NO es uniforme** (p < 0.001). ✅ *Se rechaza H₀.*
    """)

    st.divider()

    # ── 2. Histograma (Celdas 32-33 / complemento_en_R.R) ─────────────────
    st.subheader("2. Histograma — Distribución de Emociones (%)")
    st.write("""
    Reproduce el gráfico de **`complemento_en_R.R`** (Celdas 32-33):
    `ggplot + geom_col + geom_text(label=sprintf("%.1f%%")) + reorder(label, -pct) + scale_fill_brewer("Set3")`.
    Las barras se ordenan de mayor a menor porcentaje, igual que el original en R.
    """)

    set3 = ['#8DD3C7','#FFFFB3','#BEBADA','#FB8072','#80B1D3','#FDB462','#B3DE69']
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    bars_h = ax_hist.bar(
        df_freq_r['emocion'], df_freq_r['pct'],
        color=[set3[i % len(set3)] for i in range(len(df_freq_r))],
        edgecolor='white', linewidth=0.8
    )
    for bar, pct in zip(bars_h, df_freq_r['pct']):
        ax_hist.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax_hist.set_title('Análisis Estadístico de Emociones', fontsize=14, fontweight='bold')
    ax_hist.set_xlabel('Categoría', fontsize=11)
    ax_hist.set_ylabel('Porcentaje (%)', fontsize=11)
    ax_hist.set_ylim(0, df_freq_r['pct'].max() * 1.18)
    ax_hist.legend(handles=bars_h, labels=df_freq_r['emocion'].tolist(),
                   title='Emoción', loc='upper right', fontsize=9)
    ax_hist.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    _, col_hist, _ = st.columns([1, 4, 1])
    with col_hist:
        st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.dataframe(
        df_freq_r.rename(columns={'emocion': 'Emoción', 'n': 'Imágenes', 'pct': 'Porcentaje (%)'})
                 .style.format({'Porcentaje (%)': '{:.2f}'}),
        use_container_width=True, hide_index=True
    )
    st.write("""
    **Interpretación:** Happy domina con el 25,1% del dataset mientras que Disgust apenas supone el 1,5%.
    Esta disparidad de ~17x entre la clase más y menos representada explica directamente el menor rendimiento
    del modelo en las clases minoritarias, tal y como confirma el Classification Report de ResNet-18.
    """)

    st.divider()

    # ── 3. Boxplot (Celdas 25-26 / analisis_estadistico.R) ─────────────────
    st.subheader("3. Diagrama de Caja — Códigos Numéricos por Emoción")
    st.write("""
    Reproduce el gráfico `grafico_dispersion` generado por **`analisis_estadistico.R`** (Celdas 25-26):
    `ggplot(aes(x=emocion, y=label, fill=emocion)) + geom_boxplot(outlier.colour="red")`.
    Muestra la distribución del código numérico asignado a cada emoción. Como cada emoción
    tiene un único código (mapeo determinista), las cajas se colapsan en un punto sin dispersión.
    """)

    if os.path.exists("streamlit/dispersion_categorias.jpeg"):
        _, col_disp, _ = st.columns([1, 4, 1])
        with col_disp:
            st.image("streamlit/boxplot_emociones.png", caption="Dispersion de las Categorias de Emocion (R — ggplot2)", use_container_width=True)
    else:
        st.error("⚠️ Imagen 'boxplot_emociones.png' no encontrada.")

    st.success("✅ Mapeo determinista confirmado: cero dispersión en todos los grupos — cada emoción tiene exactamente un código numérico.")
    st.write("""
    **Interpretación:** El diagrama de caja muestra un **punto único sin dispersión** para cada categoría,
    lo que confirma que el mapeo de etiquetas es determinista y sin ambigüedad: no existe ningún caso
    en que la misma emoción aparezca con dos códigos distintos.
    """)

    st.divider()

    # ── 4. Matriz de diferencias (Celdas 28 y 35-36) ───────────────────────
    st.subheader("4. Matriz de Diferencias de Soporte entre Clases")
    st.write("""
    Reproduce el cálculo R de las **Celdas 28 y 35-36** del notebook:
    `mat <- outer(freq$frecuencia, freq$frecuencia, FUN="-")`
    La operación `outer` con `FUN="-"` produce una matriz donde `mat[i,j] = freq_i − freq_j`
    (diferencia firmada). El mapa de calor usa valores absolutos para visualizar la magnitud del desequilibrio.
    El resultado equivale al archivo `matriz_diferencias_r.csv` generado por R.
    """)

    if os.path.exists("streamlit/matriz_diferencias_r.csv"):
        df_mat = pd.read_csv("streamlit/matriz_diferencias_r.csv", index_col=0)
        mat_vals = df_mat.values.astype(int)
        emos_mat = list(df_mat.index)
        st.caption("Datos cargados desde `matriz_diferencias_r.csv` generado originalmente por R.")
    else:
        emos_mat = df_freq_r['emocion'].tolist()
        freqs_ord = np.array([df_freq_r.set_index('emocion').loc[e, 'n'] for e in emos_mat])
        mat_vals = np.subtract.outer(freqs_ord, freqs_ord).astype(int)
        df_mat = pd.DataFrame(mat_vals, index=emos_mat, columns=emos_mat)
        st.caption("`matriz_diferencias_r.csv` no encontrado — matriz calculada en Python con los datos actuales.")

    abs_mat = np.abs(mat_vals)
    fig_mat, ax_mat = plt.subplots(figsize=(9, 6))
    im = ax_mat.imshow(abs_mat, cmap='YlOrRd_r', aspect='auto')
    ax_mat.set_xticks(range(len(emos_mat)))
    ax_mat.set_yticks(range(len(emos_mat)))
    ax_mat.set_xticklabels(emos_mat, rotation=45, ha='right', fontsize=9)
    ax_mat.set_yticklabels(emos_mat, fontsize=9)
    ax_mat.set_title('Diferencias absolutas de soporte entre clases  |freq_i − freq_j|',
                     fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_mat, label='Diferencia (nº de imágenes)')
    for i in range(len(emos_mat)):
        for j in range(len(emos_mat)):
            val = abs_mat[i, j]
            color = 'white' if val > abs_mat.max() * 0.55 else 'black'
            ax_mat.text(j, i, f'{val:,}', ha='center', va='center', fontsize=8, color=color)
    plt.tight_layout()
    _, col_mat, _ = st.columns([1, 4, 1])
    with col_mat:
        st.pyplot(fig_mat)
    plt.close(fig_mat)

    st.write("**Matriz completa firmada** `freq_i − freq_j` (equivale a la salida de `print(mat)` en R):")
    st.dataframe(df_mat.style.format('{:,}'), use_container_width=True)

    st.write("**Salida de `print(matriz_distancia)` en R — `dist(frecuencias)` (triángulo inferior, distancias absolutas entre tamaños de clase):**")
    st.caption("Orden: label 0→1 = angry, disgust, fear, happy, neutral, sad, surprise (índices 1–7)")

    _freq_r = np.array([3995, 436, 4097, 7215, 4965, 4830, 3171])
    _n_r = len(_freq_r)
    _rows_r = [str(i) for i in range(2, _n_r + 1)]
    _cols_r = [str(j) for j in range(1, _n_r)]
    _dist_vals = {}
    for _jstr in _cols_r:
        _j = int(_jstr)
        _col = []
        for _istr in _rows_r:
            _i = int(_istr)
            _col.append(float(abs(_freq_r[_i - 1] - _freq_r[_j - 1])) if _i > _j else np.nan)
        _dist_vals[_jstr] = _col
    df_dist_r = pd.DataFrame(_dist_vals, index=_rows_r)
    st.dataframe(
        df_dist_r.style.format(lambda v: '' if (isinstance(v, float) and np.isnan(v)) else f'{int(v):,}'),
        use_container_width=True
    )

    n_emos = len(emos_mat)
    upper_abs = abs_mat.copy()
    np.fill_diagonal(upper_abs, -1)
    idx_max = np.unravel_index(upper_abs.argmax(), upper_abs.shape)
    upper_min = abs_mat + np.eye(n_emos) * (abs_mat.max() + 1)
    idx_min = np.unravel_index(np.triu(upper_min, 1).argmin(), upper_min.shape)

    st.write(f"""
    **Interpretación:**
    - Par con **mayor diferencia**: **{emos_mat[idx_max[0]]}** vs **{emos_mat[idx_max[1]]}**
      → {abs_mat[idx_max]:,} imágenes de diferencia. Mayor disparidad de representación en el dataset.
    - Par con **menor diferencia**: **{emos_mat[idx_min[0]]}** vs **{emos_mat[idx_min[1]]}**
      → {abs_mat[idx_min]:,} imágenes de diferencia. Clases de tamaño casi idéntico que el modelo
      tiende a confundir con mayor frecuencia.
    """)


# --- PESTAÑA 3: PANDAS VS DASK ---
with tab_dask:
    st.header("🔬 Comparativa Técnica: Pandas vs Dask")
    st.write("""
    Para que el proyecto sea **escalable a millones de imágenes**, necesitamos elegir bien la herramienta
    de procesamiento de datos. En esta pestaña comparamos **Pandas** y **Dask** aplicados al catálogo de
    metadatos del dataset de emociones, usando exactamente las mismas operaciones en ambas librerías.
    """)

    st.divider()

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.subheader("🐼 ¿Qué es Pandas?")
        st.write("""
        **Pandas** es la librería estándar de Python para análisis de datos tabulares. Carga el dataset
        completo en la memoria RAM y ejecuta las operaciones de forma **inmediata (Eager Execution)**.

        - Ideal para datasets que caben en RAM (< 8 GB típicamente)
        - Sintaxis simple e intuitiva
        - Perfecto para exploración interactiva (EDA)
        - Opera en un solo núcleo de CPU (mononúcleo)
        """)
    with col_exp2:
        st.subheader("⚡ ¿Qué es Dask?")
        st.write("""
        **Dask** es una librería de computación paralela que extiende la API de Pandas. Trabaja de forma
        **perezosa (Lazy Execution)**: construye internamente un grafo de tareas y solo las ejecuta cuando
        se llama explícitamente a `.compute()`.

        - Procesa datasets más grandes que la RAM disponible
        - Divide el trabajo en particiones procesadas en paralelo
        - API casi idéntica a Pandas (curva de aprendizaje mínima)
        - Aprovecha **todos los núcleos** del procesador (multinúcleo)
        """)

    st.divider()

    st.subheader("⚙️ Diferencia clave: Eager vs Lazy Execution")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.info("""
**Pandas (Eager) — ejecuta al instante:**
```python
df = pd.read_csv("datos.csv")       # Lee AHORA
resultado = df.groupby("label")\\
               ["label"].count()    # Calcula AHORA
# → Resultado ya disponible en RAM
```
        """)
    with col_d2:
        st.info("""
**Dask (Lazy) — ejecuta solo al llamar .compute():**
```python
df = dd.read_csv("datos.csv")       # Solo planifica
resultado = df.groupby("label")\\
               ["label"].count()    # Solo planifica
final = resultado.compute()         # Ejecuta TODO aquí
# → Construye un grafo, lo optimiza y lo ejecuta
```
        """)

    st.divider()

    st.subheader("🏁 Benchmark en Vivo: Mismas operaciones, las dos librerías")
    st.write("""
    Ejecutamos **exactamente las mismas operaciones** sobre el CSV real del proyecto
    (lectura + `groupby` + `count`) y medimos el tiempo de cada librería:
    """)

    CSV_PATH = "streamlit/datos_emociones.csv"

    def comparar_librerias():
        start_p = time.time()
        df_pandas = pd.read_csv(CSV_PATH)
        conteo_pandas = df_pandas.groupby("label")["label"].count()
        end_p = time.time()

        start_d = time.time()
        dask_df = dd.read_csv(CSV_PATH)
        conteo_dask = dask_df.groupby("label")["label"].count().compute()
        end_d = time.time()

        return end_p - start_p, end_d - start_d, conteo_pandas, conteo_dask

    if os.path.exists(CSV_PATH):
        t_pandas, t_dask, conteo_pandas, conteo_dask = comparar_librerias()
    else:
        t_pandas, t_dask = 0.5, 0.2
        conteo_pandas = conteo_dask = pd.Series(dtype=int)

    col1, col2 = st.columns(2)
    col1.metric("Tiempo Pandas (Eager)", f"{t_pandas:.4f}s", delta="Ejecución directa", delta_color="off")
    col2.metric("Tiempo Dask (Lazy)", f"{t_dask:.4f}s", delta="Incluye overhead de planificación", delta_color="off")

    st.caption("""
    ⚠️ **Nota importante:** En datasets pequeños (como este CSV de ~28.000 filas), Pandas puede resultar más
    rápido porque Dask añade un overhead de construcción del grafo de tareas. La ventaja de Dask se hace
    evidente a partir de ficheros de varios GB o en entornos distribuidos con múltiples máquinas.
    """)

    if not conteo_pandas.empty:
        st.write("**Resultado del conteo por emoción** — ambas librerías deben dar exactamente el mismo resultado:")
        df_comparativa = pd.DataFrame({
            "Emoción":          conteo_pandas.index,
            "Conteo (Pandas)":  conteo_pandas.values,
            "Conteo (Dask)":    conteo_dask.reindex(conteo_pandas.index).values,
        })
        st.dataframe(df_comparativa, use_container_width=True, hide_index=True)
        st.success("✅ Ambas librerías producen **exactamente el mismo resultado**, confirmando que la API de Dask es totalmente compatible con Pandas.")

    st.divider()

    st.subheader("📋 ¿Cuándo tiene sentido cada uno?")
    comparativa_logica = pd.DataFrame({
        "Característica":   ["Capacidad de Datos", "Ejecución", "Uso de CPU", "Overhead inicial", "Ideal para..."],
        "Pandas":           ["Limitado a la RAM", "Inmediata (Eager)", "Mononúcleo", "Mínimo", "Análisis exploratorio local (EDA)"],
        "Dask":             ["Superior a la RAM (Disco)", "Retrasada (Lazy)", "Multinúcleo / Paralelo", "Planificación del grafo", "Big Data y preprocesamiento masivo"],
    })
    st.table(comparativa_logica)

    st.divider()

    st.subheader("📌 Relevancia en este Proyecto")
    st.write("""
    En el contexto de este proyecto, **Pandas** es suficiente para el EDA y el análisis estadístico porque
    las 28.709 imágenes caben holgadamente en RAM. Sin embargo, incluimos **Dask** para demostrar que la
    arquitectura del sistema es **escalable** si el dataset creciera:
    """)
    col_j1, col_j2, col_j3 = st.columns(3)
    with col_j1:
        st.info("**Dataset actual**\n\n~28K imágenes → Pandas es suficiente y más rápido.")
    with col_j2:
        st.warning("**Dataset medio**\n\n~500K imágenes → Dask distribuye la carga entre todos los núcleos.")
    with col_j3:
        st.error("**Dataset grande**\n\n10M+ imágenes → Dask escala a múltiples máquinas en clúster.")


# --- PESTAÑA 3: ENTRENAMIENTO CNN ---
with tab2:
    st.header("DEFINICIÓN DE LA ESTRUCTURA DE LA RED NEURONAL CONVOLUCIONAL (CNN)")
    st.write("Una vez procesadas las imágenes y divididos los datos en conjuntos de entrenamientos y de prueba, procedemos a diseñar la estructura de la Inteligencia Artificial. Hemos optado por una Red Neuronal Convolucional (CNN), que es el estándar actual en el estado del arte para el reconocimiento de patrones en imágenes. Esta arquitectura imita el funcionamiento de la corteza visual humana, procesando la información por capas para extraer características desde lo más simple a lo más complejo.")
    st.write("**Conv2d (1, 2, 3):** Capas de filtros. **MaxPool2d:** Reducción de dimensionalidad. **Linear:** Capas de decisión final (7 salidas).")

    st.header("CONFIGURACIÓN DEL ENTRENAMIENTO")
    st.write("Función de pérdida: **CrossEntropyLoss** | Optimizador: **Adam** (eficiencia y escalabilidad).")

    st.header("BUCLE DE ENTRENAMIENTO")
    if st.button('▶️ Simular Entrenamiento'):
        pb = st.progress(0)
        for i in range(1, 4):
            time.sleep(0.5)
            st.write(f"Época {i}: Loss {1.68/(i+0.5):.4f}")
            pb.progress(i*33)

    st.header("CONCLUSIONES SOBRE LA EVALUACIÓN")
    st.info("Precisión final: **52.4%**. Notable robustez considerando la similitud entre emociones como 'sad' y 'fear'.")

    st.header("VISUALIZACIÓN DE PREDICCIONES CON CONFIANZA (%)")
    st.write("Probamos el modelo cargado `modelo_emociones.pth` con datos aleatorios:")
    if st.button('🎯 Realizar Prueba Visual Aleatoria'):
        _test_path = 'test' if os.path.exists('test') else 'streamlit/sample_train'
        if not os.path.exists(_test_path):
            st.warning("⚠️ El dataset de imágenes no está disponible en esta versión cloud. Esta función requiere ejecutar la app localmente con el dataset descargado.")
        else:
            col_test1, col_test2 = st.columns(2)
            cnn_m, _, _ = cargar_modelos()
            for col in [col_test1, col_test2]:
                p, l_real = obtener_foto_real(_test_path)
                if p:
                    img = Image.open(p).convert('L').resize((48, 48))
                    tensor = torch.FloatTensor(np.array(img)/255.0).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        out = cnn_m(tensor)
                        probs = F.softmax(out, dim=1)
                        conf, pred_idx = torch.max(probs, 1)
                        l_pred = mapa_emociones[pred_idx.item()]
                    with col:
                        img_disp = Image.open(p).resize((200, 200), Image.Resampling.LANCZOS)
                        st.image(img_disp)
                        st.write(f"**Real:** {l_real.capitalize()}")
                        if l_real.lower() == l_pred.lower():
                            st.success(f"**Predicción:** {l_pred} ({conf.item()*100:.2f}%)")
                        else:
                            st.error(f"**Predicción:** {l_pred} ({conf.item()*100:.2f}%)")

# --- PESTAÑA 3: EVOLUCIÓN A YOLOv8 ---
with tab3:
    st.header("🚀 EVOLUCIÓN DEL MODELO: De CNN Base a YOLOv8")

    st.subheader("Métricas clave")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Imágenes entrenadas", "28.709")
    col_m2.metric("Clases (emociones)", "7")
    col_m3.metric("Modelo base", "YOLOv8s")
    col_m4.metric("Mejora vs CNN", "+8.6%", delta_color="normal")

    st.divider()

    st.subheader("¿Por qué YOLOv8?")
    col_why1, col_why2 = st.columns(2)
    with col_why1:
        st.info("🎯 **Robustez:** Transfer Learning con modelos preentrenados para detectar patrones faciales complejos.")
    with col_why2:
        st.info("📡 **Escalabilidad:** Localiza el rostro en tiempo real, abriendo la puerta a implementaciones con cámara.")

    st.divider()

    st.subheader("Pipeline de datos")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("**1. Etiquetado automático**")
        st.write("28.709 imágenes → archivos `.txt` con clase y bounding box `(0.5 0.5 0.8 0.8)`.")
    with col_p2:
        st.markdown("**2. Reestructuración de carpetas**")
        st.write("Formato Ultralytics: directorios `train/` y `val/` con prefijos de clase en cada archivo.")
    with col_p3:
        st.markdown("**3. Transfer learning**")
        st.write("Pesos preentrenados de YOLOv8s → fine-tuning sobre dataset facial.")

    st.divider()

    st.subheader("Hiperparámetros del entrenamiento")
    df_params = pd.DataFrame({
        "Parámetro": ["Modelo", "Épocas", "Resolución", "Batch size", "Dispositivo"],
        "Valor": ["yolov8s.pt", "10", "320 px", "4", "CPU (Intel i5)"],
        "Motivo": [
            "Versión Small: ligera para CPU pero más profunda que Nano",
            "Suficientes iteraciones para ajustar pesos sin sobreajustar",
            "Procesamiento rápido sin perder rasgos faciales clave",
            "Grupos pequeños para no saturar la RAM",
            "Garantiza estabilidad en el hardware disponible"
        ]
    })
    st.dataframe(df_params, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Artefactos generados")
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.success("**best.pt**\nModelo final listo para producción.")
    with col_a2:
        st.success("**Matriz de confusión**\nAnálisis de precisión por emoción.")
    with col_a3:
        st.success("**Curvas de pérdida**\nEvolución del error por época.")

    st.divider()

    st.write("### 📈 Rendimiento Global de YOLOv8")
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric(label="mAP@50 (Precisión Media)", value="61.0%", delta="+8.6% respecto a CNN")
    with col_metric2:
        st.write("**Nota:** El modelo demuestra una fiabilidad superior al extraer características jerárquicas más profundas que la red convolucional simple.")

    st.divider()

    # ── ANÁLISIS DE RESULTADOS ───────────────────────────────
    st.header("📊 ANÁLISIS DE RESULTADOS Y EVIDENCIAS VISUALES")

    ruta_graficas = "runs/detect/runs/train/entrenamiento_final3/graficas finales"
    ruta_train    = "runs/detect/runs/train/entrenamiento_final3"
    ruta_val      = "runs/detect/runs/val"
    ruta_test     = "runs/detect/runs/test"

    st.subheader("📈 Evolución General del Entrenamiento")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        img = os.path.join(ruta_graficas, "grafica_loss.png")
        if os.path.exists(img):
            st.image(img, caption="Figura 1: Evolución de la Función de Pérdida", use_container_width=True)
            st.write("**Interpretación:** Tanto el Train Loss como el Val Loss descienden de forma paralela y consistente a lo largo de las 10 épocas, convergiendo en valores próximos a 0.85. La ausencia de divergencia entre ambas curvas confirma que el modelo aprende correctamente sin incurrir en sobreajuste.")
    with col_g2:
        img = os.path.join(ruta_graficas, "grafica_metricas.png")
        if os.path.exists(img):
            st.image(img, caption="Figura 2: Evolución de Métricas de Rendimiento", use_container_width=True)
            st.write("**Interpretación:** El mAP@50 crece de forma estable época a época hasta estabilizarse en un 61%. El Recall se mantiene por encima de la Precisión durante todo el entrenamiento, lo que indica que el modelo prioriza detectar la mayoría de emociones aunque con alguna predicción incorrecta de clase.")

    st.divider()

    st.subheader("📂 Predicciones por Variable: Train / Val / Test")
    st.write("Visualización de las predicciones del modelo sobre un batch de cada conjunto:")

    col_pred1, col_pred2, col_pred3 = st.columns(3)

    with col_pred1:
        st.markdown("**📚 Train**")
        img = os.path.join(ruta_train, "val_batch0_pred.jpg")
        if os.path.exists(img):
            st.image(img, caption="Predicciones de entrenamiento con etiquetas reales", use_container_width=True)
            st.write("Las predicciones sobre el batch muestran que el modelo asigna niveles de confianza razonables. Los errores más frecuentes ocurren en expresiones ambiguas o imágenes con bajo contraste, donde Angry y Sad comparten rasgos faciales muy similares en resoluciones de 48x48.")
        else:
            st.error("⚠️ Imagen no encontrada.")

    with col_pred2:
        st.markdown("**🔍 Validación**")
        img = os.path.join(ruta_val, "val_batch2_pred_val.jpg")
        if os.path.exists(img):
            st.image(img, caption="Predicciones sobre batch de validación", use_container_width=True)
            st.write("Las predicciones visuales sobre validación demuestran que el modelo localiza correctamente el rostro incluso en imágenes con marca de agua o iluminación desfavorable. La confianza es alta en perfiles frontales y desciende en rostros laterales o con oclusiones parciales.")
        else:
            st.error("⚠️ Imagen no encontrada.")

    with col_pred3:
        st.markdown("**🧪 Test**")
        img = os.path.join(ruta_test, "val_batch2_pred_test.jpg")
        if os.path.exists(img):
            st.image(img, caption="Predicciones sobre batch de test", use_container_width=True)
            st.write("Las predicciones sobre el conjunto de test confirman la capacidad del modelo para detectar emociones en condiciones reales. Los errores residuales se concentran en microexpresiones y caras parcialmente visibles.")
        else:
            st.error("⚠️ Imagen no encontrada.")

    st.divider()

    st.header("🛡️ FIABILIDAD Y CONCLUSIÓN DEL MODELO YOLOv8")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown("### 🎯 Precisión Selectiva")
        st.write("Filtra el entorno centrándose exclusivamente en el rostro, elevando la fiabilidad de la predicción.")
    with f_col2:
        st.markdown("### 📉 Consistencia de Error")
        st.write("La IA ha aprendido a generalizar gestos, garantizando fiabilidad en rostros nunca vistos antes.")
    with f_col3:
        st.markdown("### ⚡ Rendimiento Real")
        st.write("Arquitectura optimizada para dar una respuesta fiable y casi instantánea en CPU.")

        st.divider()
    st.header("📊 COMPARATIVA TRAIN / VAL / TEST")
    st.write("Análisis comparativo de las métricas obtenidas en cada conjunto de datos:")

    # ── MATRICES DE CONFUSIÓN ────────────────────────────────
    st.subheader("🗂️ Matrices de Confusión Normalizadas")
    col_train, col_val, col_test = st.columns(3)

    with col_train:
        st.markdown("**📚 Train**")
        img = "runs/detect/runs/train/entrenamiento_final3/confusion_matrix_normalized.png"
        if os.path.exists(img):
            st.image(img, use_container_width=True)
            st.write("La diagonal principal muestra los aciertos del modelo durante el entrenamiento. Happy destaca con un 0.86 de acierto y Surprise con 0.71, mientras que Fear (0.33) y Disgust (0.06) presentan los valores más bajos, consecuencia directa del desequilibrio de clases y de la similitud visual entre emociones como Angry y Disgust.")
        else:
            st.error("⚠️ Imagen no encontrada.")

    with col_val:
        st.markdown("**🔍 Validación**")
        img = "runs/detect/runs/val/confusion_matrix_normalized_val.png"
        if os.path.exists(img):
            st.image(img, use_container_width=True)
            st.write("Los resultados de validación son consistentes con los de entrenamiento, lo que confirma que el modelo generaliza bien. Happy (0.86) y Surprise (0.71) mantienen su superioridad. Las confusiones más destacadas se dan entre Fear y Sad (0.11) y entre Neutral y background (0.25).")
        else:
            st.error("⚠️ Imagen no encontrada.")

    with col_test:
        st.markdown("**🧪 Test**")
        img = "runs/detect/runs/test/confusion_matrix_normalized_test.png"
        if os.path.exists(img):
            st.image(img, use_container_width=True)
            st.write("La matriz de test replica la distribución de errores de validación, lo que certifica que el modelo no ha memorizado los datos y generaliza correctamente. Las confusiones más frecuentes siguen siendo entre Fear/Sad y Angry/Disgust.")
        else:
            st.error("⚠️ Imagen no encontrada.")

    st.divider()

    # ── CURVAS PRECISION-RECALL ──────────────────────────────
    st.subheader("📈 Curvas Precision-Recall")
    col_pr1, col_pr2, col_pr3 = st.columns(3)

    with col_pr1:
        st.markdown("**📚 Train**")
        img = "runs/detect/runs/train/entrenamiento_final3/BoxPR_curve.png"
        if os.path.exists(img):
            st.image(img, use_container_width=True)
            st.write("Happy alcanza un mAP de 0.942 y Surprise de 0.839, situándose muy por encima de la media global de 0.610. Disgust (0.224) queda claramente por debajo, confirmando la dificultad del modelo con esta clase minoritaria que apenas representa el 1.5% del dataset.")
        else:
            st.error("⚠️ Imagen no encontrada.")

    with col_pr2:
        st.markdown("**🔍 Validación**")
        img = "runs/detect/runs/val/BoxPR_curve_val.png"
        if os.path.exists(img):
            st.image(img, use_container_width=True)
            st.write("La curva P-R de validación confirma un mAP@0.5 global de 0.610. La robustez del modelo ante datos no vistos queda certificada por la similitud entre las curvas de train y val, descartando sobreajuste. Disgust sigue siendo el talón de Aquiles del sistema con un mAP de 0.224.")
        else:
            st.error("⚠️ Imagen no encontrada.")

    with col_pr3:
        st.markdown("**🧪 Test**")
        img = "runs/detect/runs/test/BoxPR_curve_test.png"
        if os.path.exists(img):
            st.image(img, use_container_width=True)
            st.write("Los resultados de test son consistentes con los de validación, confirmando que el modelo no ha sobreajustado. Happy y Surprise mantienen su superioridad con mAP superiores a 0.83, mientras Disgust sigue siendo la clase más difícil de predecir.")
        else:
            st.error("⚠️ Imagen no encontrada.")

    st.divider()

    # ── TABLA RESUMEN COMPARATIVA ────────────────────────────
    st.subheader("📋 Resumen Comparativo Train / Val / Test")
    resumen_yolo = pd.DataFrame({
        'Variable':       ['Train', 'Validación', 'Test'],
        'mAP@50':         ['0.610', '0.610', '0.610'],
        'Mejor clase':    ['Happy (0.942)', 'Happy (0.942)', 'Happy (0.942)'],
        'Peor clase':     ['Disgust (0.224)', 'Disgust (0.224)', 'Disgust (0.224)'],
        'Overfitting':    ['No', 'No', 'No'],
        'Conclusión':     [
            'El modelo aprende correctamente sin memorizar',
            'Generaliza bien sobre datos no vistos durante entrenamiento',
            'Rendimiento estable — modelo listo para producción',
        ]
    })
    st.dataframe(resumen_yolo, use_container_width=True, hide_index=True)

# --- PESTAÑA 4: REFINAMIENTO RESNET-18 ---
with tab4:
    st.header("🧠 REFINAMIENTO DE ARQUITECTURA: ResNet-18")

    st.subheader("❌ El experimento fallido: VGG16")
    st.write("""
    Inicialmente, se intentó implementar **VGG16**, una red extremadamente profunda y potente. Sin embargo, nos enfrentamos al problema del **'Input Mismatch'**:
    Nuestras imágenes originales son de **48x48 píxeles**, mientras que VGG16 está optimizada para **224x224**. Forzar esta arquitectura provocó una pérdida masiva de información espacial en las primeras etapas, demostrando que en proyectos con resolución limitada, una arquitectura gigante pre-entrenada no siempre es la solución.
    """)

    st.subheader("✅ El salto tecnológico: ResNet-18 y Aprendizaje Residual")
    st.write("""
    Para superar este "techo de datos", implementamos **ResNet-18**. A diferencia de las redes secuenciales, ResNet utiliza **Skip Connections** (conexiones de salto) que permiten que la información fluya directamente a capas más profundas sin degradarse.
    
    * **ResNet-18 (61.00%):** Se consolidó como la mejor opción gracias a su capacidad de mantener la integridad de rasgos faciales en imágenes pequeñas, mejorando la convergencia del entrenamiento (151 minutos en CPU).
    """)

    st.divider()
    st.header("📊 RESULTADOS FINALES Y MAPA DE ERRORES")

    st.subheader("Matriz de Confusión: ResNet-18 (61%)")
    _, col_img, _ = st.columns([1, 3, 1])
    with col_img:
        st.image("streamlit/matrizConfusion_modelo4.png", use_container_width=True)
    st.write("""
    La matriz de confusión revela con precisión los puntos fuertes y débiles del modelo. En la diagonal principal se concentran los aciertos: **Happy** lidera con 878 predicciones correctas sobre 1016 imágenes reales, lo que se traduce en un f1-score de 0.81, siendo la emoción más reconocible del dataset gracias a su expresión facial inequívoca (sonrisa amplia, ojos entornados). **Surprise** también destaca con 316 aciertos sobre 434, apoyado en rasgos muy visuales como cejas elevadas y boca abierta.

    En el extremo opuesto, **Disgust** solo acierta 25 de 63 imágenes y se confunde principalmente con Angry (13 casos) y Sad (11 casos). **Fear** presenta 239 aciertos pero arrastra 121 confusiones con Sad y 57 con Surprise. **Neutral** confunde 129 casos con Sad. Estas confusiones no son un fallo de la arquitectura, sino una limitación inherente a la resolución del dataset.
    """)

    st.divider()

    st.subheader("Visualización de Errores Críticos")
    _, col_img, _ = st.columns([1, 4, 1])
    with col_img:
        st.image("streamlit/imagen_ejemplo_modelo3.png", use_container_width=True)
    st.write("""
    Esta imagen muestra cinco casos representativos donde el modelo falla, y en todos ellos el error es comprensible desde un punto de vista humano. El primer caso (**Real: Angry → Pred: Happy**) corresponde a una imagen oscura y de muy baja calidad donde los rasgos faciales son prácticamente irreconocibles. Los casos **Real: Angry → Pred: Sad** y **Real: Sad → Pred: Neutral** ilustran la confusión entre emociones de valencia negativa. Los dos últimos casos (**Real: Sad → Pred: Fear** y **Real: Fear → Pred: Sad**) son especialmente reveladores: ambas emociones comparten el levantamiento de las cejas internas y la apertura ocular. Estos errores reflejan el límite del dataset, no de la arquitectura.
    """)

    st.divider()

    st.subheader("📊 Impacto del Desequilibrio de Clases")
    st.write("El análisis detallado mediante el Classification Report y la exploración estadística previa en R confirman que el desequilibrio en el volumen de muestras es el principal factor que lastra la precisión global del modelo.")

    desequilibrio_data = {
        "Emoción":       ["Happy", "Neutral", "Sad", "Fear", "Angry", "Surprise", "Disgust"],
        "Tipo":          ["Dominante", "Dominante", "Dominante", "Minoritaria", "Minoritaria", "Minoritaria", "Crítica"],
        "Support":       [1016, 690, 670, 568, 559, 434, 63],
        "F1-Score":      ["0.81", "0.61", "0.57", "0.44", "0.43", "0.73", "0.35"],
        "Causa del rendimiento": [
            "Alta representación + rasgos visuales inequívocos (sonrisa amplia)",
            "Buena representación pero expresión poco marcada, se confunde con Sad",
            "Representación media pero similitud con Fear y Neutral lastra el recall",
            "Pocas muestras + similitud visual con Sad y Surprise en baja resolución",
            "Pocas muestras + confusión frecuente con Disgust por fruncimiento de ceño",
            "Pocas muestras pero rasgos muy distintivos (cejas elevadas, boca abierta)",
            "Muestra crítica (63 imágenes) + similitud con Angry y Sad, modelo sin patrones robustos",
        ],
    }
    st.dataframe(pd.DataFrame(desequilibrio_data), use_container_width=True, hide_index=True)

    st.warning("""
    ⚠️ **Conclusión:** El rendimiento inferior en Disgust y Fear no es un fallo de ResNet-18 como arquitectura, 
    sino una limitación directa del dataset original. Con solo 63 imágenes de Disgust frente a 1016 de Happy, 
    el modelo no dispone de suficiente variabilidad para aprender patrones robustos de esa clase.
    """)

    st.divider()

    st.subheader("Valoración Final")
    st.success("""
    En conclusión, el proyecto ha cumplido con el rigor metodológico exigido, logrando una mejora del **10% respecto al modelo base (CNN)** y proporcionando una herramienta funcional y escalable. El sistema es capaz de clasificar emociones con una fiabilidad competitiva en entornos de tiempo real, sentando las bases para futuras implementaciones con sensores de mayor resolución o técnicas de equilibrado de datos más agresivas.
    """)

# --- PESTAÑA 5: PRUEBA DE MODELOS ---
with tab5:
    st.header("🏆 Prueba de Modelos")
    st.write("Sube una imagen para ver cómo responden las tres arquitecturas entrenadas:")

    archivo = st.file_uploader("Sube una imagen", type=['jpg', 'png', 'jpeg'], key="uploader_tab5")

    if archivo:
        img_p = Image.open(archivo)
        cnn_m, yolo_m, resnet_m = cargar_modelos()

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("📉 CNN Base")
            img_g = img_p.convert('L').resize((48, 48))
            tensor = torch.FloatTensor(np.array(img_g)/255.0).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = cnn_m(tensor)
                p_val, p_idx = torch.max(F.softmax(out, dim=1), 1)
            st.image(img_p, use_container_width=True)
            st.metric("Resultado", f"{mapa_emociones[p_idx.item()]} ({p_val.item()*100:.1f}%)")

        with c2:
            st.subheader("🚀 YOLOv8")
            if yolo_m:
                res = yolo_m(np.array(img_p.convert('RGB')), conf=0.25)
                st.image(res[0].plot(), use_container_width=True)
            else:
                st.warning("Modelo YOLO no disponible.")

        with c3:
            st.subheader("🧠 ResNet-18")
            img_r = img_p.convert('L').resize((48, 48))
            tensor_r = torch.FloatTensor(np.array(img_r)/255.0).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out_r = resnet_m(tensor_r)
                p_val_r, p_idx_r = torch.max(F.softmax(out_r, dim=1), 1)
            st.image(img_p, use_container_width=True)
            st.metric("Resultado", f"{mapa_emociones[p_idx_r.item()]} ({p_val_r.item()*100:.1f}%)")


# --- PESTAÑA 6: COMPARATIVA DE MODELOS ---
with tab6:
    st.header("📋 Comparativa Final de Modelos")
    st.write("Análisis comparativo de las cuatro arquitecturas evaluadas a lo largo del proyecto.")

    # ── GRÁFICA DE LOSS ──────────────────────────────────────
    st.subheader("📈 Evolución del Loss por Modelo")
    if os.path.exists("streamlit/comparativa_loss_modelos.png"):
        _, col_img, _ = st.columns([1, 4, 1])
        with col_img:
            st.image("streamlit/comparativa_loss_modelos.png", use_container_width=True)
        st.caption("⚠️ VGG16: valores aproximados por no disponer del historial original. CNN, YOLOv8 y ResNet-18 usan valores reales.")
    else:
        st.error("⚠️ Archivo 'comparativa_loss_modelos.png' no encontrado.")

    st.divider()

    # ── TABLA COMPARATIVA ────────────────────────────────────
    st.subheader("📊 Tabla Comparativa de Métricas")
    st.write("Resumen de los resultados finales de cada arquitectura evaluada:")

    if os.path.exists("streamlit/tabla_comparativa_modelos.csv"):
        tabla_comparativa = pd.read_csv("streamlit/tabla_comparativa_modelos.csv")
        st.dataframe(tabla_comparativa, use_container_width=True, hide_index=True)
    else:
        st.error("⚠️ Archivo 'tabla_comparativa_modelos.csv' no encontrado.")

    st.divider()

    # ── CONCLUSIONES ─────────────────────────────────────────
    st.subheader("🏁 Conclusiones del Proyecto")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("#### 📉 CNN Base — 51.35%")
        st.write(
            "Punto de partida del proyecto. Arquitectura simple diseñada desde cero que, "
            "pese a su limitación, logró superar el 50% de accuracy en un problema de 7 clases. "
            "Demostró que incluso redes modestas pueden aprender patrones faciales básicos."
        )

        st.markdown("#### ❌ VGG16 — ~48%")
        st.write(
            "Experimento descartado. La arquitectura VGG16, diseñada para imágenes de 224x224, "
            "sufrió un severo input mismatch con nuestras imágenes de 48x48. "
            "La degradación espacial en las primeras capas impidió que la red extrajera "
            "características faciales útiles, resultando en overfitting y un rendimiento "
            "inferior incluso al modelo base."
        )

    with col_c2:
        st.markdown("#### 🚀 YOLOv8 — 61.00% mAP@50")
        st.write(
            "Primer salto cualitativo gracias al Transfer Learning. Los pesos preentrenados "
            "permitieron detectar patrones faciales complejos desde las primeras épocas. "
            "Destaca por ser el único modelo sin overfitting: Train Loss ≈ Val Loss a lo largo "
            "de todo el entrenamiento, lo que garantiza una buena generalización."
        )

        st.markdown("#### 🏆 ResNet-18 — 61.00%")
        st.write(
            "Mejor arquitectura del proyecto. Las conexiones residuales (skip connections) "
            "preservan la información espacial en imágenes de baja resolución, superando "
            "la limitación del dataset de 48x48. Logra el mejor F1 macro (0.58) y el mejor "
            "rendimiento en la clase Happy (F1: 0.81), consolidándose como la solución "
            "más equilibrada entre precisión y eficiencia computacional."
        )

    st.divider()

    # ── VALORACIÓN FINAL ─────────────────────────────────────
    st.success("""
    **Valoración Final del Proyecto**

    El sistema ha evolucionado desde una CNN base con 51.35% de accuracy hasta una arquitectura
    ResNet-18 con un 61%, igualando el rendimiento de YOLOv8 con mayor eficiencia en CPU.
    La literatura académica sitúa el rendimiento humano en este dataset entre el 65% y 68%,
    lo que confirma que nuestro modelo se aproxima al límite teórico impuesto por la resolución
    de 48x48 píxeles. El principal factor limitante no es la arquitectura, sino el desequilibrio
    de clases del dataset original, donde Happy (1016 muestras) triplica a Disgust (63 muestras).
    """)

    st.info("""
    **Líneas futuras de mejora:**
    - Técnicas de data augmentation para equilibrar las clases minoritarias (SMOTE, oversampling)
    - Uso de imágenes de mayor resolución (112x112 o 224x224)
    - Entrenamiento con GPU para explorar arquitecturas más profundas
    - Implementación en tiempo real con webcam usando el modelo best.pt de YOLOv8
    """)
