
library(ggplot2)
library(dplyr)

# Ler o arquivo CSV
data <- read.csv("CLASSFIED_TOPICS_X15_BM25.csv")

# Processar os dados
data_processed <- data %>%
  mutate(recall_flag = ifelse(posicao_tema_real >= 1 & posicao_tema_real <= 6, 1, 0)) %>%
  group_by(num_tema_cadastrado) %>%
  summarise(
    recall = sum(recall_flag) / n(),
    count = n()  # Calcular a contagem de cada num_tema_cadastrado
  ) %>%
  filter(count <=100) %>%
  arrange(recall)  # Ordenar por recall

# Reordenar num_tema_cadastrado com base no recall
data_processed$num_tema_cadastrado <- factor(data_processed$num_tema_cadastrado, levels = data_processed$num_tema_cadastrado)

# Visualizar o resultado
print(data_processed)

# Criar o gráfico de tendência
ggplot(data_processed, aes(x = num_tema_cadastrado, y = recall)) +
  geom_line(aes(group = 1)) +  # Para garantir que a linha conecta os pontos na ordem correta
  geom_point() +
  scale_x_discrete(labels = data_processed$count) +  # Usar as contagens como rótulos do eixo X
  labs(title = "Gráfico de Tendência do Recall por Tema Cadastrado",
       x = "Contagem de Tema Cadastrado",
       y = "Recall") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size=6))  # Rotacionar os rótulos do eixo X se necessário
