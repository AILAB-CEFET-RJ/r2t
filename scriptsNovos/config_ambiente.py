"""
config_ambiente.py

Configuração usada por montarAmbiente.py (e, por baixo dos panos, por
splitDados.py, criarDiretorios.py e prepararTreinoVariantes.py).

Referenciado também de dentro de config_experimentos.py (campo
CONFIG_AMBIENTE), pois rodarExperimentos.py precisa saber com que
estratégia/tamanho de resumo as variantes de treino foram geradas, para
montar os caminhos corretos em split/treino_variantes/.
"""

# Seeds usadas nos splits baseline/stratified/minority. Cada seed gera uma
# partição diferente dos dados, salva com o sufixo "_<seed>" no nome do
# arquivo. zero_shot_split NÃO usa seed (split único e determinístico).
SEEDS = [42, 7, 9, 67, 19]

# Resumo aplicado aos splits de TREINO, para gerar as variantes usadas no
# fine-tuning de modelos _treinado (ver prepararTreinoVariantes.py).
# Independente do resumo aplicado ao TESTE, que é configurado em
# config_experimentos.py (RESUMO_TAMANHO_VALORES / RESUMO_ESTRATEGIA_VALORES).
#
# Cada estratégia tem o SEU PRÓPRIO tamanho -- não existe eixo de variação
# aqui (diferente do resumo de teste, que varia dentro da grade). Se uma
# estratégia não aparece neste dict, ela simplesmente não é gerada, e
# nenhuma treino_variante que dependa dela pode ser usada em
# config_experimentos.py (rodarExperimentos.py valida isso e avisa).
#
# Exemplo:
#   RESUMOS_TREINO = {
#       "guided_lexrank": 15,
#       "lexrank": 60,
#   }
RESUMOS_TREINO = {}