
library(ggplot2)
library(tidyverse)
library(forcats)
library(dplyr)
library(scales)
library(hrbrthemes)

#Leitura dos dados
REsp <- read.csv('recurso.csv')
#REsp <-read.csv('recursos_avaliados_elasticsearch.csv')
#Considerar registros que tenham ao menos uma sugestao de tema
REsp <- subset(REsp, !is.na(sugerido_1))
REsp <- subset(REsp, !is.na(num_tema_cadastrado))
REsp <- subset(REsp, recurso != "")
REsp <- subset(REsp, sugestao_adotada < 7)
REsp <- subset(REsp, num_tema_cadastrado!=5090)
REsp$num_tema_cadastrado < - factor(REsp$num_tema_cadastrado)


p <- REsp%>%
  ggplot( aes(x=factor(num_tema_cadastrado))) +
  geom_bar( fill="#69b3a2", color="#e9ecef", alpha=0.9 , width = 1) +
  ggtitle("Bin size = 10") +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=15),axis.text.x = element_text(angle = -90, vjust = 0.5, hjust = 1 , size =5)
  )+
  labs(title = "Histograma de tipos de temas do corpus",
       x = "Identificação do tema",
       y = "Frequência (escala logarítmica)") +
  scale_y_log10()

print(p)


p <- REsp%>%
  ggplot( aes(x=num_tema_cadastrado)) +
  geom_bar( binwidth=10, fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  ggtitle("Bin size = 10") +
  theme_ipsum() +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10, simplify = FALSE)) +
  theme(
    plot.title = element_text(size=15),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)
  )+
  labs(title = "Histograma de tipos de temas do corpus",
       x = "Identificação do tema",
       y = "Frequência (escala logarítmica)") +
  scale_y_log10()

print(p)

# Calcula a quantidade de temas distintos
quantidade_distintos <- length(unique(REsp$num_tema_cadastrado))

# Exibe o resultado
print(quantidade_distintos)