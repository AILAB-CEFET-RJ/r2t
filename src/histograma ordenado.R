library(ggplot2)
library(tidyverse)
library(forcats)
library(dplyr)
library(scales)
library(hrbrthemes)



# Leitura dos dados
REsp <- read.csv('REsp_completo.csv')

# Considerar registros que tenham ao menos uma sugestao de tema
REsp <- subset(REsp, !is.na(sugerido_1))
REsp <- subset(REsp, !is.na(num_tema_cadastrado))
REsp <- subset(REsp, recurso != "")
REsp <- subset(REsp, sugestao_adotada < 7)
REsp <- subset(REsp, num_tema_cadastrado != 5090)
REsp$num_tema_cadastrado <- factor(REsp$num_tema_cadastrado)

# Calcular a frequência de cada tema
tema_counts <- table(REsp$num_tema_cadastrado)

# Ordenar os temas por frequência
ordered_levels <- names(sort(tema_counts, decreasing = TRUE))

# Reorganizar os níveis do fator com base na ordem da frequência
REsp$num_tema_cadastrado <- factor(REsp$num_tema_cadastrado, levels = ordered_levels)

# Criar o histograma
p <- REsp %>%
  ggplot(aes(x = num_tema_cadastrado)) +
  geom_bar(fill = "#69b3a2", color = "#e9ecef", alpha = 0.9, width = 1.0) +
  ggtitle("") +
  theme_ipsum() +
  theme(
    plot.title = element_text(size = 15),
    #axis.text.x = element_text(angle = -90, vjust = 0.5, hjust = 1, size = 4),
    axis.text.x = element_blank(),
    axis.text.y = element_text(size = 12),  
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14) 
  ) +
  labs(
    #title = "Histograma de tipos de temas do corpus",
    x = "Theme identification",
    y = "Frequency"
  ) +
  scale_y_log10()+
  scale_y_continuous(expand = c(0, 0))
  

print(p)

# Calcula a quantidade de temas distintos
quantidade_distintos <- length(unique(REsp$num_tema_cadastrado))

# Exibe o resultado
print(quantidade_distintos)
