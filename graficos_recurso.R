
library(ggplot2)
library(tidyverse)
library(forcats)
library(dplyr)
library(scales)
library(hrbrthemes)

#Leitura dos dados
REsp <- read.csv('recurso.csv')

#Considerar registros que tenham ao menos uma sugestao de tema
REsp <- subset(REsp, !is.na(sugerido_1))

#Considerar registros com tema cadastrado pelo analista
REsp <- subset(REsp, !is.na(num_tema_cadastrado))

#Conteudo do recurso não pode ser vazio
REsp <- subset(REsp, recurso != "")

#Limitacao em 6 sugestoes
REsp <- subset(REsp, sugestao_adotada < 7)

REsp$sugestao_adotada <- factor(REsp$sugestao_adotada)


#Criando dataframe com contagens e percentual das sugestoes dadas
sugestoes<-REsp %>% count(sugestao_adotada)
sugestoes<-sugestoes %>% as.data.frame()
sugestoes <-
  mutate(
    sugestoes,
    escolha = recode(
      sugestoes$sugestao_adotada,
      "0" = "Nenhuma",
      "1" = "1ªOpção",
      "2" = "2ªOpção",
      "3" = "3ªOpção",
      "4" = "4ªOpção",
      "5" = "5ªOpção",
      "6" = "6ªOpção"
    )
  )

total_sugestoes<-REsp %>% nrow()
total_sugestoes

sugestoes<-sugestoes %>% mutate(percentual_sugestoes = n/total_sugestoes)

#Grafico de barras com todas as sugestoes dadas
sugestoes_plot <-
  ggplot(data = sugestoes, aes(
    x = escolha %>% fct_reorder(percentual_sugestoes) ,
    y = percentual_sugestoes
  )) + geom_bar(stat = "identity") + ggtitle("Desempenho da classificação de documentos com Elastisearch(BM25)") + scale_y_continuous(limits = c(0,0.70),labels = percent) + coord_flip() +
  xlab("Sugestão adotada")+ylab("") + geom_text(aes(label = sprintf("%1.1f%%", percentual_sugestoes *
                                                                      100)),
                                                vjust = 1,hjust = -0.2,
                                                colour = "black") + theme_bw() + theme(axis.text.y = element_text(size=8)) 

print(sugestoes_plot)



#-------------------------mapa de calor --------------------------------

REsp_agrupado_tema<-REsp
REsp_agrupado_tema<-subset(REsp_agrupado_tema,num_tema_cadastrado!=5090) #Tema 5090 apresentando inconsistencia

REsp_agrupado_tema<-REsp_agrupado_tema %>%group_by(num_tema_cadastrado)
sugestoes_agrupado_tema<-REsp_agrupado_tema %>% count(sugestao_adotada)
sugestoes_agrupado_tema<-sugestoes_agrupado_tema %>% as.data.frame()
sugestoes_agrupado_tema <-
  mutate(
    sugestoes_agrupado_tema,
    escolha = recode(
      sugestoes_agrupado_tema$sugestao_adotada,
      "0" = "Nenhuma",
      "1" = "1ªOpção",
      "2" = "2ªOpção",
      "3" = "3ªOpção",
      "4" = "4ªOpção",
      "5" = "5ªOpção",
      "6" = "6ªOpção"
    )
  )
total_sugestoes_agrupado_tema<-REsp %>% nrow()
View(sugestoes_agrupado_tema)
amostra_agrupada<-sugestoes_agrupado_tema[sugestoes_agrupado_tema$num_tema_cadastrado==339 |sugestoes_agrupado_tema$num_tema_cadastrado==444 |sugestoes_agrupado_tema$num_tema_cadastrado==534 |sugestoes_agrupado_tema$num_tema_cadastrado==555 |sugestoes_agrupado_tema$num_tema_cadastrado==694 |sugestoes_agrupado_tema$num_tema_cadastrado==1081, ]


#Selecionando amostra de alguns temas pra criar mapa de calor
amostra_agrupada<-read.csv('amostra_agrupada.csv')

amostra_agrupada$num_tema_cadastrado<-factor(amostra_agrupada$num_tema_cadastrado)
mapa_calor<- ggplot(amostra_agrupada, aes(x = escolha, y = num_tema_cadastrado, fill = percentual_sugestoes)) +
  geom_tile(colour="black", size=0.25)+ labs(x = "", y = "Numeração dos Temas")+ scale_x_discrete(position="top")+labs(fill= "Percentual das sugestões")+theme_bw()

# Exibição do mapa de calor
print(mapa_calor)

#----------------------------------------------------------------------------------------------------------

VICE<-read.csv('vice.csv',sep = ';', encoding = "Windows-1252",colClasses = c("Date","numeric","character"))


#grafico tempo processo vice-presidencia
visiveis<-subset(VICE,Marcado==TRUE)

vice_plot <- ggplot(VICE, aes(x=Data, y=Dias)) +
  geom_line( color="steelblue4") + 
  geom_point(data = visiveis,size=3,color="steelblue4") +
  xlab("") +
  theme_light()+
  theme(axis.text.x=element_text( hjust=0.5)) + ylim(2500,8000) + geom_text(data = visiveis, aes(label = Dias),vjust = -0.8,hjust = -0.2, angle=45,size=4.0,colour = "black") +
  labs(title = "" ) +
  ylab("Tempo médio em dias") +
  theme(axis.title.y = element_text(size = 14,face = "bold"))

print(vice_plot)