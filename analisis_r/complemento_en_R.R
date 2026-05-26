library(ggplot2)
library(dplyr)
library(stats)

# 1. Carga tus datos reales
datos <- read.csv("streamlit/datos_emociones.csv")

# 2. Tu pipeline limpio asignado SÍ O SÍ a 'grafico2'
grafico2 <- datos %>%
  count(label) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = reorder(label, -pct), y = pct, fill = label)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.1f%%", pct)), vjust = -0.5, fontface = "bold") +
  labs(title = "Analisis Estadistico de Emociones",
       x = "Categoria", 
       y = "Porcentaje (%)",
       fill = "Emocion") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")
