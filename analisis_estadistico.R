library(dplyr)
library(ggplot2)

datos <- read.csv("datos_emociones.csv")

mapa_emociones <- c("0" = "angry", "1" = "disgust", "2" = "fear", 
                    "3" = "happy", "4" = "neutral", "5" = "sad", "6" = "surprise")

tabla_conteo <- table(datos$label)
names(tabla_conteo) <- mapa_emociones[names(tabla_conteo)]

tabla_proporciones <- prop.table(tabla_conteo) * 100

resumen_frecuencias <- data.frame(
  Emocion = names(tabla_conteo),
  Cantidad = as.numeric(tabla_conteo),
  Porcentaje = paste0(round(as.numeric(tabla_proporciones), 2), "%")
)

grafico_dispersion <- datos %>%
  mutate(emocion = mapa_emociones[as.character(label)]) %>%
  ggplot(aes(x = emocion, y = label, fill = emocion)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 8) +
  labs(title = "Dispersion de las Categorias de Emocion",
       x = "Emocion", y = "Codigo Numerico") +
  theme_classic()

calidad_datos <- data.frame(
  Variable = colnames(datos),
  Valores_Nulos = colSums(is.na(datos))
)

frecuencias <- as.numeric(table(datos$label))
matriz_distancia <- dist(frecuencias) # Distancia euclidiana entre tamaños de clases

