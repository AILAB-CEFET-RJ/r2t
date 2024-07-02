library(tidyverse)

# Carregar os dados do CSV
dados <- read_csv("REsp_completo.csv")

# Definir tamanho desejado de X_test e X_train
total_teste <- round(0.2 * nrow(dados))
total_treinamento <- nrow(dados) - total_teste

# Definir exclusivos_train e obrigatorios (substitua com suas listas)
exclusivos_train <- c(182, 214, 225, 247, 250, 280, 353, 408, 425, 432, 439, 473, 480, 529, 546, 556, 668, 684, 700, 702, 709, 732, 861, 863, 874, 948, 1007, 1036, 1039, 1085, 1104, 1123, 1142, 1157, 1176, 1190, 1193, 1197, 1226, 1090, 12, 1178, 7, 1033, 6, 163, 118, 1140, 14, 11, 1067,1081, 3, 1080, 1125, 1046, 20, 72, 1083, 1048, 534, 1002, 769)

obrigatorios <- c(4, 217, 315, 445, 491, 492, 505, 526, 531, 542, 699, 885, 888, 996, 1017, 1049, 1060, 1129, 1137, 1147, 1224, 8, 271, 424, 449,459, 554, 698, 777, 1018, 1023, 1055, 1182, 1184, 32, 483, 1037, 1047, 1095, 1130, 544, 555, 660, 666, 1012, 1162, 372, 412, 504, 589, 692, 740,897, 899, 992, 994, 1042, 1070, 1098, 1115, 571, 1186, 395, 503, 914, 990, 1141, 567, 570, 906, 1013, 1026, 568, 569, 930, 1160, 1102, 359, 1150,973, 995, 1024, 1031, 9, 1093, 895, 1170, 1105, 1088, 1169, 1100, 1064, 1075, 961, 966, 5, 76, 478, 479, 737, 739, 1109, 1086, 1011, 339, 738, 444,661, 1124, 1014, 1076, 1199, 1005, 1174, 962, 1057, 1079, 999, 1164, 694, 1209, 1, 10, 1071, 1056)

# Adicionar coluna de índice único para controle
dados <- dados %>%
  mutate(indice = row_number())

# Separar registros exclusivos de X_train
X_train_exclusivos <- dados %>%
  filter(num_tema_cadastrado %in% exclusivos_train)

# Filtrar registros para distribuição aleatória entre X_train e X_test
dados_restantes <- dados %>%
  filter(!num_tema_cadastrado %in% exclusivos_train)

# Inicializar X_train e X_test com os registros exclusivos de X_train
X_train <- X_train_exclusivos
X_test <- data.frame()

# Selecionar um exemplo de cada classe em X_obrigatorios para X_train
X_train_obrigatorios <- dados %>%
  filter(num_tema_cadastrado %in% obrigatorios) %>%
  distinct(num_tema_cadastrado, .keep_all = TRUE)

# Selecionar um exemplo de cada classe em X_obrigatorios para X_test
X_test_obrigatorios <- X_train_obrigatorios %>%
  slice(1) %>%
  mutate(conjunto = "test")

# Adicionar exemplos obrigatórios em X_train
X_train <- bind_rows(X_train, X_train_obrigatorios)

# Adicionar exemplos obrigatórios em X_test
X_test <- bind_rows(X_test, X_test_obrigatorios)

# Remover exemplos obrigatórios de dados_restantes
dados_restantes <- anti_join(dados_restantes, X_train_obrigatorios, by = "indice")

# Distribuir aleatoriamente registros restantes entre X_test e X_train
while (nrow(X_train) < total_treinamento | nrow(X_test) < total_teste) {
  # Selecionar uma classe aleatória dos dados restantes
  num_tema_cadastrado_aleatorio <- sample(unique(dados_restantes$num_tema_cadastrado), 1)
  
  # Filtrar registros dessa classe para X_train ou X_test
  registros_mover <- dados_restantes %>%
    filter(num_tema_cadastrado == num_tema_cadastrado_aleatorio) %>%
    slice(1)  # Seleciona apenas um registro
  
  # Verificar onde mover o registro
  if (nrow(X_train) < total_treinamento) {
    X_train <- bind_rows(X_train, registros_mover)
  } else {
    X_test <- bind_rows(X_test, registros_mover)
  }
  
  # Atualizar dados_restantes removendo registros movidos
  dados_restantes <- anti_join(dados_restantes, registros_mover, by = "indice")
}

# Verificar quantidades finais
cat("Total de registros em X_train:", nrow(X_train), "\n")
cat("Total de registros em X_test:", nrow(X_test), "\n")
cat("Soma dos registros em X_train e X_test:", nrow(X_train) + nrow(X_test), "\n")

# Salvar os conjuntos em arquivos CSV sem 'indice' e 'conjunto'
X_train %>% select(-indice) %>% write_csv("X_train.csv")
X_test %>% select(-indice, -conjunto) %>% write_csv("X_test.csv")
