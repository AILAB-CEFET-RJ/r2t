
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

estatisticas <-function(df){
  recursos <- df$recurso
  
  num_palavras <- sapply(strsplit(recursos, "\\s+"),length)
  mediana <- median(num_palavras)
  min_palavras <- min(num_palavras)
  max_palavras <- max(num_palavras)
  cat("Mediana da quantidade de palavras:", mediana, "\n")
  cat("Quantidade mínima de palavras:", min_palavras, "\n")
  cat("Quantidade máxima de palavras:", max_palavras, "\n")
}

estatisticas(REsp)