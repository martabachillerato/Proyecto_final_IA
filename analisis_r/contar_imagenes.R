library(dplyr)

datos <- read.csv("streamlit/datos_emociones.csv")

conteo <- datos %>% count(label, name = "imagenes")
total  <- nrow(datos)

print(conteo)
cat("Total de imagenes en el dataset:", total, "\n")
