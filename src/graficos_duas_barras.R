
library(ggplot2)
library(tidyverse)
library(forcats)
library(dplyr)
library(scales)
library(hrbrthemes)

#Leitura dos dados
REsp_elastic <- read.csv('REsp_completo.csv')
#Resp_bertopic <- read.csv('resp_bertopic_unsuperv_filter_13_seed.csv')
REsp <- REsp_elastic
#REsp <- Resp_bertopic


#Dataset do Elastic
#Considerar registros que tenham ao menos uma sugestao de tema - necessario p/ dataset do elastic
REsp <- subset(REsp, !is.na(sugerido_1))
REsp <- subset(REsp, !is.na(num_tema_cadastrado))
REsp <- subset(REsp, recurso != "")
REsp <- subset(REsp, sugestao_adotada < 7)
REsp <- subset(REsp, num_tema_cadastrado!=5090)

Temas <- REsp %>% distinct(tema,num_tema_cadastrado)
Temas <- subset(Temas, !is.na(tema))




#Dataset do bertopic
#REsp <- REsp %>% mutate(sugestao_adotada = ifelse(posicao_tema_real > 6 | is.na(posicao_tema_real),0,posicao_tema_real))

sugestoes2<- REsp %>% select(sugestao_adotada)
sugestoes_positivas<-subset(sugestoes2,sugestao_adotada > 0)
sugestoes_binarias<-sugestoes2 %>% mutate(sugestao_adotada = ifelse(sugestao_adotada > 0, 1,sugestao_adotada))

#Criando dataframe com contagens e percentual das sugestoes dadas
sugestoes_pos<-sugestoes_positivas %>% count(sugestao_adotada)
sugestoes_pos<-sugestoes_pos %>% as.data.frame()
sugestoes_pos <-
  mutate(
    sugestoes_pos,
    escolha = recode(
      sugestoes_pos$sugestao_adotada,
      "1" = "1ªOpção",
      "2" = "2ªOpção",
      "3" = "3ªOpção",
      "4" = "4ªOpção",
      "5" = "5ªOpção",
      "6" = "6ªOpção"
    )
  )

total_sugestoes_pos<-sugestoes_positivas %>% nrow()


sugestoes_pos<-sugestoes_pos %>% mutate(percentual_sugestoes_pos = n/total_sugestoes_pos)

#Grafico de barras com todas as sugestoes dadas
sugestoes_pos_plot <-
  ggplot(data = sugestoes_pos, aes(
    x = escolha %>% fct_reorder(percentual_sugestoes_pos) ,
    y = percentual_sugestoes_pos
  )) + geom_bar(stat = "identity") + ggtitle("Distribuição das classificações corretas") + scale_y_continuous(limits = c(0,0.8),labels = percent) + coord_flip() +
  xlab("Sugestão adotada")+ylab("") + geom_text(aes(label = sprintf("%1.1f%%", percentual_sugestoes_pos *
                                                                      100)),
                                                vjust = 1,hjust = -0.2,
                                                colour = "black") + theme_bw() + theme(axis.text.y = element_text(size=8)) 

sugestoes_pos_plot


#Criando dataframe com contagens e percentual das sugestoes dadas
sugestoes_bin<-sugestoes_binarias %>% count(sugestao_adotada)
sugestoes_bin<-sugestoes_bin %>% as.data.frame()
sugestoes_bin <-
  mutate(
    sugestoes_bin,
    escolha = recode(
      sugestoes_bin$sugestao_adotada,
      "1" = "Corretas",
      "0" = "Erradas"
    )
  )

total_sugestoes_bin<-sugestoes_binarias %>% nrow()
total_sugestoes_bin

sugestoes_bin<-sugestoes_bin %>% mutate(percentual_sugestoes_bin = n/total_sugestoes_bin)

#Grafico de barras com todas as sugestoes dadas
sugestoes_bin_plot <-
  ggplot(data = sugestoes_bin, aes(
    x = escolha %>% fct_reorder(percentual_sugestoes_bin) ,
    y = percentual_sugestoes_bin
  )) + geom_bar(stat = "identity",width=0.25) + ggtitle("Desempenho da Classificação de documentos com Elasticsearch(BM25)") + scale_y_continuous(limits = c(0,0.8),labels = percent) +
  xlab("Sugestões")+ylab("") + geom_text(aes(label = sprintf("%1.1f%%", percentual_sugestoes_bin *
                                                                      100)),
                                                vjust = -1,hjust = 0.5,
                                                colour = "black") + theme_bw() + theme(axis.text.y = element_text(size=8)) 

sugestoes_bin_plot

Topicos <- read.csv('bertopic_nao_supervisionado_13_seed.csv')
View(Topicos)
lista_topicos <- Topicos %>% distinct(topicos)
View(lista_topicos)

temas <- read.csv('temas_repetitivos.csv')
View(temas)

#temas_inc <- read.csv('temas_inconsistentes.csv')
#View(temas_inc)

#temas_eproc <-read.csv('Temas_repetitivos_eproc.csv')
#View(temas_eproc)