"""
config_experimentos.py

Configuração usada por rodarExperimentos.py -- substitui a antiga seção
"CONFIGURAÇÃO -- EDITE AQUI" que ficava hardcoded no topo do script.

Passado via: python rodarExperimentos.py --config config_experimentos.py
(esse é o próprio default, então rodar sem --config reproduz exatamente
esta grade).
"""

# Caminho para o config de AMBIENTE correspondente a este experimento --
# usado para saber com quais seeds/estratégia-tamanho de resumo os splits
# de treino foram preparados (necessário em resolver_appeals_treino, para
# apontar para o arquivo certo dentro de split/treino_variantes/).
# Caminho relativo é resolvido em relação à pasta deste arquivo.
CONFIG_AMBIENTE = "config_ambiente.py"

# -----------------------------------------------------
# Quais datasets entram na grade
# -----------------------------------------------------

# Kinds de split a rodar. "zeroshot" não usa seed (split único).
KINDS = ["baseline", "stratified", "minority", "zeroshot"]

# Seeds a rodar -- deve ser um subconjunto das SEEDS geradas pelo
# CONFIG_AMBIENTE referenciado acima. Se uma seed pedida aqui não tiver
# split gerado no disco, rodarExperimentos.py avisa e pula só aquele
# dataset (não aborta a execução inteira).
# SEEDS = [42, 7, 9, 67, 19]
SEEDS = [42]

# -----------------------------------------------------
# Eixos de pré-processamento / método / modelo
# -----------------------------------------------------

USAR_CLEAN_VALORES = [False, True]
BEGIN_POINT_VALORES = [None]  # [None, "cabimento"]  -- só importa se usar_clean=True

USAR_RESUMO_VALORES = [False]
RESUMO_TAMANHO_VALORES = [15]                    # só importa se usar_resumo=True
RESUMO_ESTRATEGIA_VALORES = ["guided_lexrank"]   # ["lexrank", "guided_lexrank"]  -- só importa se usar_resumo=True

METODO_SIMILARIDADE_VALORES = ["COS"]

MODELO_VALORES = [
    "distiluseMultilingual",
    "paraphraseMpnet",
    "paraphraseMiniLm",
    "bertimbauBase",
    "bertLenerBr",
    "legalBertimbauBase",
    "legalBertimbauLarge",
    "bertimbauLarge",
    "distiluseMultilingual_treinado",
    "paraphraseMpnet_treinado",
    "paraphraseMiniLm_treinado",
    "bertimbauBase_treinado",
    "bertLenerBr_treinado",
    "legalBertimbauBase_treinado",
    "legalBertimbauLarge_treinado",
    "bertimbauLarge_treinado",
]

# Variantes de pré-processamento do TREINO usadas no fine-tuning de
# modelos _treinado (ver prepararTreinoVariantes.py). Só entra na grade
# quando o modelo é _treinado; para os demais casos (BM25, modelo base)
# fica fixo em "raw" e não gera combinações extras.
# Os testes (appeals/temas) NUNCA mudam entre variantes -- só o treino.
# As variantes "resumo_*"/"clean_resumo_*" exigem que a estratégia
# correspondente esteja definida em RESUMOS_TREINO no config de ambiente.
TREINO_VARIANTE_VALORES = [
    "raw",
    "clean",
    # "resumo_lexrank",
    # "resumo_guided_lexrank",
    # "clean_resumo_lexrank",
    # "clean_resumo_guided_lexrank",
]

K_VALORES = [6]