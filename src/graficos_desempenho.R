library(ggplot2)
library(tidyverse)
library(forcats)
library(dplyr)
library(scales)
library(hrbrthemes)


results <- read.csv('resultados.csv',sep=";")

results<- results %>% mutate(REPRESENTACAO = str_replace(REPRESENTACAO,"BERTOPIC","Resumo - BERTopic"))
results<- results %>% mutate(REPRESENTACAO = str_replace(REPRESENTACAO,"GUIADO","Resumo - BERTopic guiado"))
results<- results %>% mutate(REPRESENTACAO = str_replace(REPRESENTACAO,"LEXRANK","Resumo - LexRank"))
results<- results %>% mutate(REPRESENTACAO = str_replace(REPRESENTACAO,"LEXGUIDED","Resumo - LexRank guiado"))
results<- results %>% mutate(REPRESENTACAO = str_replace(REPRESENTACAO,"NENHUMA","Baseline"))
results<- results %>% mutate(REPRESENTACAO = str_replace(REPRESENTACAO,"TEXTO","Texto completo"))

resultsf1<- results %>% select(Recall,F1,REPRESENTACAO)
resultsndcg<- results %>% select(Recall,NDCG,REPRESENTACAO)
View(resultsf1)
View(results)

plotmap <- ggplot(results,aes(MAP,Recall)) + geom_point(aes(colour = REPRESENTACAO),size=3)+ 
  geom_hline(yintercept = results$Recall[results$REPRESENTACAO == "Baseline"], linetype = "dashed") +  
  theme_classic()+
  labs(color = "Tipo de Representação", )+
  theme(
    text = element_text(family = "Times New Roman",size=12, face="bold"),
    legend.text = element_text(family = "Times New Roman",size=12), 
    legend.title = element_text(family = "Times New Roman",size=12), 
     axis.text.x = element_text(size = 13),
      axis.text.y = element_text(size = 13)
    
  )
print(plotmap)

plot_f1 <- ggplot(resultsf1,aes(F1,Recall)) + geom_point(aes(colour = factor(REPRESENTACAO)),size=3)+ 
  geom_hline(yintercept = results$Recall[results$REPRESENTACAO == "Baseline"], linetype = "dashed") +  
  theme_classic()+
  labs(color = "Tipo de Representação")+
  theme(
    text = element_text(family = "Times New Roman",size=12, face="bold"),
    legend.text = element_text(family = "Times New Roman",size=12), 
    legend.title = element_text(family = "Times New Roman",size=12), 
    axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 13)
    
  )
print(plot_f1)

plotndcg <- ggplot(resultsndcg,aes(NDCG,Recall)) + geom_point(aes(colour = factor(REPRESENTACAO)),size=3)+ 
  geom_hline(yintercept = results$Recall[results$REPRESENTACAO == "Baseline"], linetype = "dashed") +  
  theme_classic()+
  labs(color = "Tipo de Representação")+
  theme(
    text = element_text(family = "Times New Roman",size=12, face="bold"),
    legend.text = element_text(family = "Times New Roman",size=12), 
    legend.title = element_text(family = "Times New Roman",size=12), 
    axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 13)
    
  )
print(plotndcg)

resultsTam <-results %>% select(Recall, REPRESENTACAO,Tamanho,Similaridade,REMOCAO)

resultsTam<-resultsTam %>% filter(!is.na(resultsTam$Tamanho))
resultsTamCos<-resultsTam %>% filter(resultsTam$Similaridade =='COSSENO')
resultsTamCos<-resultsTamCos %>% filter(resultsTamCos$REMOCAO=='S')
resultsTamCos<-resultsTamCos %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - BERTopic","BERTopic"))
resultsTamCos<-resultsTamCos %>% mutate(REPRESENTACAO =str_replace(REPRESENTACAO,"Resumo - BERTopic guiado","BERTopic guiado"))
resultsTamCos<-resultsTamCos %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - LexRank","LexRank"))
resultsTamCos<-resultsTamCos %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - LexRank guiado","LexRank guiado"))



plotTamCos <- ggplot(resultsTamCos, aes(x = Tamanho, y = Recall, color = REPRESENTACAO, group = REPRESENTACAO)) +
  geom_line() +
  geom_text(data = subset(resultsTamCos, Tamanho == max(Tamanho)), aes(label = REPRESENTACAO), hjust = -0.1, vjust = 0) +
  theme_classic()+
  xlim(10,70)+
  theme(
    text = element_text(family = "Times New Roman", face="bold", size =12),legend.position = "none",  axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 13),
    plot.title = element_text(face = "plain", size = 12)
  )+
  ggtitle("Similaridade: Cosseno")
  

print(plotTamCos)




resultsTamBM25<-resultsTam %>% filter(resultsTam$Similaridade =='BM25')
resultsTamBM25<-resultsTamBM25 %>% filter(resultsTamBM25$REMOCAO=='S')
resultsTamBM25<-resultsTamBM25 %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - BERTopic","BERTopic"))
resultsTamBM25<-resultsTamBM25 %>% mutate(REPRESENTACAO =str_replace(REPRESENTACAO,"Resumo - BERTopic guiado","BERTopic guiado"))
resultsTamBM25<-resultsTamBM25 %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - LexRank","LexRank"))
resultsTamBM25<-resultsTamBM25 %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - LexRank guiado","LexRank guiado"))

plotTamBM25 <- ggplot(resultsTamBM25, aes(x = Tamanho, y = Recall, color = REPRESENTACAO, group = REPRESENTACAO)) +
  geom_line() +
  geom_text(data = subset(resultsTamBM25, Tamanho == max(Tamanho)), aes(label = REPRESENTACAO), hjust = -0.1, vjust = 0) +
  theme_classic()+
  xlim(10,70)+
  theme(
    text = element_text(family = "Times New Roman", face="bold", size =12),legend.position = "none",  axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 13),
    plot.title = element_text(face = "plain", size = 12)
  )+
  ggtitle("Similaridade: BM25")
print(plotTamBM25)




resultsTamBM25b<-resultsTam %>% filter(resultsTam$Similaridade =='BM25')
resultsTamBM25b<-resultsTamBM25b %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - BERTopic","BERTopic"))
resultsTamBM25b<-resultsTamBM25b %>% mutate(REPRESENTACAO =str_replace(REPRESENTACAO,"Resumo - BERTopic guiado","BERTopic guiado"))
resultsTamBM25b<-resultsTamBM25b %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - LexRank","LexRank"))
resultsTamBM25b<-resultsTamBM25b %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - LexRank guiado","LexRank guiado"))


plotTamBM25b <- ggplot(resultsTamBM25b, aes(x = Tamanho, y = Recall, color = REMOCAO, group = REMOCAO)) +
  geom_line() +
  facet_wrap(~REPRESENTACAO)+
  theme_classic()+
  scale_color_discrete(name = "Similaridade: bm25 \n\n\nRemoção de termos : " )+
  theme(
    text = element_text(family = "Times New Roman", face="bold", size =12), axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 13)
  )+
  labs(color = "Similaridade - bm25")

print(plotTamBM25b)

resultsTamCosb<-resultsTam %>% filter(resultsTam$Similaridade =='COSSENO')
resultsTamCosb<-resultsTamCosb %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - BERTopic","BERTopic"))
resultsTamCosb<-resultsTamCosb %>% mutate(REPRESENTACAO =str_replace(REPRESENTACAO,"Resumo - BERTopic guiado","BERTopic guiado"))
resultsTamCosb<-resultsTamCosb %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - LexRank","LexRank"))
resultsTamCosb<-resultsTamCosb %>% mutate(REPRESENTACAO=str_replace(REPRESENTACAO,"Resumo - LexRank guiado","LexRank guiado"))

plotTamCosb <- ggplot(resultsTamCosb, aes(x = Tamanho, y = Recall, color = REMOCAO, group = REMOCAO)) +
  geom_line() +
  facet_wrap(~REPRESENTACAO)+
  theme_classic()+
  scale_color_discrete(name = "Similaridade: cosseno \n\n\nRemoção de termos : " )+
  theme(
    text = element_text(family = "Times New Roman", face="bold", size =12),  axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 13)
  )+
  labs(color = "Similaridade - cosseno")

print(plotTamCosb)
mediana <- median(results$Recall)
