"""
splitDados.py

Gera os splits de treino/teste do dataset de special appeals, em 4
formas diferentes: baseline, stratified, minority e zeroshot.

Seeds
-----
baseline, stratified e minority são gerados UMA VEZ PARA CADA seed passada
em --seeds (OBRIGATÓRIO -- não há mais default hardcoded aqui; vem sempre
do config de ambiente, via montarAmbiente.py --config). Cada seed produz
arquivos com o sufixo "_<seed>" no nome (ex: special_appeal_baseline_42.csv)
-- uma partição diferente dos dados, tratada depois, no resto do pipeline
(rodarExperimentos.py), como DATASET DIFERENTE.

zeroshot é gerado UMA ÚNICA VEZ, sem seed. A escolha de quais classes
viram zero-shot é inteiramente determinística (classes mais raras,
acumuladas até ~TEST_SIZE do total) -- não há nenhuma fonte de
aleatoriedade nesse split, e a intenção é mantê-lo assim: representa o
"pior caso" fixo de generalização para temas nunca vistos, não uma
amostra que faça sentido repetir com seeds diferentes.

Para adicionar uma seed nova: basta acrescentar o valor em SEEDS no
config de ambiente (config_ambiente.py). Isso é suficiente para propagar
para todos os scripts a jusante (splitDados.py, criarDiretorios.py,
prepararTreinoVariantes.py), desde que rodados via montarAmbiente.py.
"""

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from ioUtil import salvar_csv_atomico

# =====================================================
# CONFIGURAÇÕES
# =====================================================

APPEALS_FILE = "special_appeal.csv"

# Local canônico do corpus depois que criarDiretorios.py já rodou uma vez
# (o arquivo é MOVIDO para lá, não copiado -- ver localizar_appeals_file).
APPEALS_FILE_POS_CRIAR_DIRETORIOS = os.path.join(
    "data", "appeals", "notClean", "texto", "special_appeal.csv"
)

TEXT_COLUMN = "special_appeal_text"
LABEL_COLUMN = "theme_id"

TEST_SIZE = 0.2

# Pastas de saída
TRAIN_DIR = "split/treino"
TEST_DIR = "split/teste"

# Mapas appeal_id -> posição original no special_appeal.csv completo, um
# por metade do split (treino/teste), mesma ordem das linhas do CSV do
# split correspondente. Usado por prepararTreinoVariantes.py para
# reaproveitar resumos já calculados para o CORPUS COMPLETO em vez de
# recalcular por seed/kind (ver garantir_resumo_completo lá).
IDS_TRAIN_DIR = "split/ids/treino"
IDS_TEST_DIR = "split/ids/teste"


# =====================================================
# FUNÇÕES
# =====================================================

def localizar_appeals_file() -> str:
    """
    Localiza o corpus de appeals em um dos dois lugares possíveis:

      1) raiz do projeto (./special_appeal.csv) -- primeira execução, ou
         um arquivo novo colocado manualmente ali.
      2) data/appeals/notClean/texto/special_appeal.csv -- local canônico
         depois que criarDiretorios.py já rodou uma vez (o arquivo é
         MOVIDO para lá, não copiado, então some da raiz).

    Sem essa busca em duas localizações, rodar montarAmbiente.py uma
    segunda vez (ex: para adicionar uma seed nova) falha logo no primeiro
    passo, porque o arquivo já não está mais na raiz.
    """
    candidatos = [APPEALS_FILE, APPEALS_FILE_POS_CRIAR_DIRETORIOS]
    for candidato in candidatos:
        if os.path.isfile(candidato):
            return candidato
    raise FileNotFoundError(
        f"{APPEALS_FILE} não encontrado em nenhum dos locais esperados: "
        f"{candidatos}. Rode resetarAmbiente.py se os dados-fonte foram "
        f"movidos para src/, ou verifique se o corpus existe."
    )


def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset carregado de {filepath}: {df.shape}")
    return df


def create_folders():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(IDS_TRAIN_DIR, exist_ok=True)
    os.makedirs(IDS_TEST_DIR, exist_ok=True)


def split_ja_existe(name: str) -> bool:
    """
    True se o split de TREINO com esse nome já foi gerado antes (usado
    como sinal de "já processado" -- split/treino/ nunca é movido/consumido
    por outro script, diferente de split/teste/, que criarDiretorios.py
    consome).
    """
    return os.path.isfile(os.path.join(TRAIN_DIR, f"{name}.csv"))


def ids_existem(name: str) -> bool:
    """True se os mapas appeal_id de treino E de teste já existem para 'name'."""
    return (
        os.path.isfile(os.path.join(IDS_TRAIN_DIR, f"{name}.csv"))
        and os.path.isfile(os.path.join(IDS_TEST_DIR, f"{name}.csv"))
    )


def localizar_teste_split(name: str):
    """
    Localiza o CSV de TESTE de um split específico em um dos dois lugares
    possíveis -- mesmo raciocínio de localizar_appeals_file(), mas para o
    arquivo de teste de UM split:

      1) split/teste/<name>.csv -- ainda não foi consumido por
         criarDiretorios.py.
      2) data/appeals/notClean/texto/<name>.csv -- local para onde
         criarDiretorios.py MOVE o arquivo de teste depois de rodar.

    Retorna None se não achar em nenhum dos dois (não deveria acontecer
    em uso normal, mas quem chama deve tratar esse caso).
    """
    candidatos = [
        os.path.join(TEST_DIR, f"{name}.csv"),
        os.path.join("data", "appeals", "notClean", "texto", f"{name}.csv"),
    ]
    for candidato in candidatos:
        if os.path.isfile(candidato):
            return candidato
    return None


def salvar_mapa_ids(df, path: str) -> None:
    """
    Salva um CSV de uma coluna só (appeal_id), na mesma ordem das linhas
    de df -- df.index já é a posição original da linha no
    special_appeal.csv completo (nenhuma das funções de split abaixo
    reseta o índice; ver docstrings de cada uma).
    """
    mapa = pd.DataFrame({"appeal_id": df.index})
    salvar_csv_atomico(mapa, path, index=False)


def verificar_split_bate(df_recomputado, caminho_csv: str, name: str, lado: str) -> None:
    """
    Confere que um split recomputado em memória (mesma função + mesma
    seed) bate, linha a linha, com o CSV já persistido em disco --
    condição necessária para confiar no df_recomputado.index como fonte
    do mapa de appeal_id quando o CSV já existia de uma execução anterior
    e só o mapa está faltando.

    Levanta erro (em vez de seguir silenciosamente) se não bater --
    isso indicaria, por exemplo, mudança de versão de pandas/sklearn/numpy
    entre a execução original e agora.
    """
    persistido = pd.read_csv(caminho_csv)
    recomputado = df_recomputado[[LABEL_COLUMN, TEXT_COLUMN]].reset_index(drop=True)
    persistido_chk = persistido[[LABEL_COLUMN, TEXT_COLUMN]].reset_index(drop=True)

    if len(recomputado) != len(persistido_chk) or not recomputado.equals(persistido_chk):
        raise RuntimeError(
            f"[{name}] O {lado} recomputado em memória (mesma seed) NÃO bate com "
            f"o CSV já salvo em {caminho_csv}. Abortando em vez de gravar um mapa "
            f"de appeal_id incorreto -- possível mudança de versão de "
            f"pandas/sklearn/numpy entre a execução original e agora."
        )


# -----------------------------------------------------
# 1) SPLIT BASELINE
# -----------------------------------------------------
def basic_split(df, seed: int):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=seed
    )
    return train_df, test_df


# -----------------------------------------------------
# 2) SPLIT ESTRATIFICADO
# -----------------------------------------------------
def stratified_split(df, seed: int):
    counts = df[LABEL_COLUMN].value_counts()

    # Remove classes com apenas 1 exemplo (não splittáveis de forma estratificada)
    valid_classes = counts[counts > 1].index
    removed = counts[counts == 1]
    df_filtered = df[df[LABEL_COLUMN].isin(valid_classes)]

    print(f"[Stratified seed={seed}] Classes removidas (singleton): {len(removed)}")
    print(f"[Stratified seed={seed}] Registros removidos: {len(df) - len(df_filtered)}")

    train_df, test_df = train_test_split(
        df_filtered,
        test_size=TEST_SIZE,
        stratify=df_filtered[LABEL_COLUMN],
        random_state=seed
    )

    print(f"[Stratified seed={seed}] Treino: {train_df.shape}, Teste: {test_df.shape}")
    return train_df, test_df


# -----------------------------------------------------
# 3) SPLIT FEW-SHOT (minority)
#
# Objetivo (conforme o artigo, Seção 6.6, "Performance in classes with few
# examples"): o conjunto de teste é composto majoritariamente pelas classes
# com menor representação no corpus, mas o treino tem garantidamente ao
# menos 1 exemplo de CADA classe -- incluindo as minoritárias.
#
# Regras implementadas:
#   - Respeita TEST_SIZE globalmente (≈20% do total).
#   - Prioriza classes com menos exemplos para compor o teste.
#   - Garante >= 1 exemplo de cada classe no treino antes de alocar
#     qualquer exemplo ao teste.
#   - Reproduzível via seed.
# -----------------------------------------------------
def minority_split(df, seed: int):
    rng = np.random.default_rng(seed)

    # Ordena classes da menos para a mais frequente
    counts = df[LABEL_COLUMN].value_counts().sort_values()

    test_indices = []
    guaranteed_train_indices = []  # 1 exemplo por classe reservado ao treino

    for cls in counts.index:
        cls_idx = df[df[LABEL_COLUMN] == cls].index.tolist()
        rng.shuffle(cls_idx)  # shuffle reproduzível

        # Reserva obrigatoriamente 1 exemplo para o treino
        guaranteed_train_indices.append(cls_idx[0])
        remaining = cls_idx[1:]

        # O restante vai para o pool de candidatos ao teste
        test_indices.extend(remaining)

    # Calcula quantos exemplos devem ir ao teste no total
    n_total = len(df)
    n_test_target = int(round(n_total * TEST_SIZE))

    # Os candidatos ao teste já estão ordenados das classes mais raras
    # para as mais frequentes (porque iteramos em counts ordenado).
    # Pega exatamente n_test_target exemplos desse pool (priorizando
    # as classes minoritárias, que aparecem primeiro).
    test_indices_final = test_indices[:n_test_target]

    # Tudo que não está no teste vai para o treino
    test_set = set(test_indices_final)
    train_indices = [i for i in df.index if i not in test_set]

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices_final]

    train_classes = set(train_df[LABEL_COLUMN].unique())
    test_classes = set(test_df[LABEL_COLUMN].unique())
    classes_ausentes_treino = test_classes - train_classes
    assert len(classes_ausentes_treino) == 0, (
        f"[minority_split seed={seed}] BUG: {len(classes_ausentes_treino)} classes "
        f"no teste sem nenhum exemplo no treino: {classes_ausentes_treino}"
    )

    print(f"[Minority seed={seed}] Treino: {train_df.shape}, Teste: {test_df.shape}")
    print(f"[Minority seed={seed}] Classes no treino: {train_df[LABEL_COLUMN].nunique()}, "
          f"no teste: {test_df[LABEL_COLUMN].nunique()}")
    print(f"[Minority seed={seed}] Proporção real de teste: "
          f"{len(test_df)/n_total:.2%} (alvo: {TEST_SIZE:.0%})")

    return train_df, test_df


# -----------------------------------------------------
# 4) ZERO-SHOT
#
# Objetivo (conforme o artigo, Seção 6.6.1): o conjunto de teste contém
# apenas exemplos de classes que NÃO possuem nenhum representante no
# treino -- simulando um tema inédito para o modelo supervisionado.
#
# Regras implementadas:
#   - Seleciona as classes com MENOS exemplos para serem zero-shot,
#     de forma a compor ≈TEST_SIZE do total.
#   - As classes zero-shot não têm nenhum exemplo no treino.
#   - As classes restantes ficam inteiramente no treino.
#   - Respeita TEST_SIZE globalmente (o conjunto de classes zero-shot
#     é escolhido de modo que seus exemplos totalizem ≈TEST_SIZE do corpus).
#
# NÃO usa seed: a seleção das classes é inteiramente determinística
# (ordenação por frequência + acúmulo guloso), então rodar de novo sempre
# produz exatamente a mesma partição. Ver docstring do módulo.
# -----------------------------------------------------
def zero_shot_split(df):
    counts = df[LABEL_COLUMN].value_counts().sort_values()  # menos → mais freq.

    n_total = len(df)
    n_test_target = int(round(n_total * TEST_SIZE))

    # Acumula classes (da mais rara para a mais frequente) até atingir o
    # tamanho alvo do teste. Cada classe selecionada vai INTEIRAMENTE ao teste.
    zero_classes = []
    accumulated = 0
    for cls, cnt in counts.items():
        if accumulated + cnt <= n_test_target:
            zero_classes.append(cls)
            accumulated += cnt
        else:
            # Aceita a classe se ela aproxima mais do alvo do que não aceitá-la
            if abs(accumulated + cnt - n_test_target) < abs(accumulated - n_test_target):
                zero_classes.append(cls)
                accumulated += cnt
            break  # uma vez que ultrapassamos o alvo, paramos

    test_df = df[df[LABEL_COLUMN].isin(zero_classes)]
    train_df = df[~df[LABEL_COLUMN].isin(zero_classes)]

    assert len(set(zero_classes) & set(train_df[LABEL_COLUMN].unique())) == 0, \
        "[zero_shot_split] BUG: classes zero-shot aparecem no treino."

    print(f"[Zero-shot] Classes zero-shot (apenas no teste): {len(zero_classes)}")
    print(f"[Zero-shot] Treino: {train_df.shape}, Teste: {test_df.shape}")
    print(f"[Zero-shot] Proporção real de teste: "
          f"{len(test_df)/n_total:.2%} (alvo: {TEST_SIZE:.0%})")

    return train_df, test_df


# -----------------------------------------------------
# SALVAR
# -----------------------------------------------------
def save_split(train_df, test_df, name):
    train_file = os.path.join(TRAIN_DIR, f"{name}.csv")
    test_file = os.path.join(TEST_DIR, f"{name}.csv")

    # Ordem importa: split_ja_existe() checa só o arquivo de TREINO como
    # sinal de "já pronto" (split/teste/ é consumido/movido por
    # criarDiretorios.py depois, então não é um sinal confiável de
    # persistência). Escrevendo o teste primeiro e o treino por último,
    # se o processo morrer no meio (disco cheio, kill etc.), o treino
    # nunca existe sozinho -- a checagem de cache corretamente considera
    # o par como "não pronto" e refaz os dois na próxima execução.
    salvar_csv_atomico(test_df, test_file, index=False)
    salvar_csv_atomico(train_df, train_file, index=False)

    print(f"[{name}] salvo:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")


def processar_split(name: str, gerar_func) -> None:
    """
    Garante que um split (CSV de treino/teste) e os mapas de appeal_id
    correspondentes existam, cobrindo os 3 estados possíveis:

      1) Nada existe          -> gera tudo (split + os dois mapas).
      2) Split existe,        -> NÃO regrava o CSV do split (preserva
         mapa não existe         exatamente a partição já usada em
                                  resultados anteriores); recomputa em
                                  memória só para validar (ver
                                  verificar_split_bate) e então extrai o
                                  índice para os mapas.
      3) Tudo já existe       -> [CACHE], não faz nada.

    gerar_func: () -> (train_df, test_df), determinístico (mesma seed).
    """
    csv_pronto = split_ja_existe(name)
    ids_prontos = ids_existem(name)

    if csv_pronto and ids_prontos:
        print(f"[CACHE] {name}: split + mapas de appeal_id já existem -- pulando.")
        return

    train_df, test_df = gerar_func()

    if not csv_pronto:
        save_split(train_df, test_df, name)
    else:
        print(f"[VALIDANDO] {name}: CSV já existia -- conferindo antes de gerar os mapas de id.")
        train_csv = os.path.join(TRAIN_DIR, f"{name}.csv")
        verificar_split_bate(train_df, train_csv, name, "treino")

        teste_csv = localizar_teste_split(name)
        if teste_csv is not None:
            verificar_split_bate(test_df, teste_csv, name, "teste")
        else:
            print(
                f"[AVISO] {name}: não encontrei o CSV de teste (nem em {TEST_DIR}, "
                f"nem em data/appeals/notClean/texto/) para validar -- "
                f"confiando só na validação do lado do treino."
            )

    if not ids_prontos:
        salvar_mapa_ids(train_df, os.path.join(IDS_TRAIN_DIR, f"{name}.csv"))
        salvar_mapa_ids(test_df, os.path.join(IDS_TEST_DIR, f"{name}.csv"))
        print(f"[OK] {name}: mapas de appeal_id gerados em {IDS_TRAIN_DIR}/ e {IDS_TEST_DIR}/.")


# =====================================================
# MAIN
# =====================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Gera os splits de treino/teste (baseline, stratified, minority, zeroshot)."
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, required=True,
        help="Seeds para baseline/stratified/minority (zeroshot não usa seed). Obrigatório.",
    )
    args = p.parse_args(argv)
    seeds = args.seeds

    create_folders()

    # Lista de tarefas: (nome, função geradora). A função geradora só é
    # CHAMADA (e só então precisa do corpus em memória) se a tarefa
    # estiver pendente -- ver o filtro logo abaixo.
    tarefas = []
    for seed in seeds:
        tarefas.append((f"special_appeal_baseline_{seed}", "baseline", seed))
        tarefas.append((f"special_appeal_stratified_{seed}", "stratified", seed))
        tarefas.append((f"special_appeal_minority_{seed}", "minority", seed))
    tarefas.append(("special_appeal_zeroshot", "zeroshot", None))

    # Pendente = falta o CSV do split OU falta algum dos mapas de id
    # (cobre tanto "nunca rodou" quanto "rodou antes desta mudança, CSV
    # existe mas os mapas ainda não").
    pendentes = [
        t for t in tarefas if not (split_ja_existe(t[0]) and ids_existem(t[0]))
    ]
    prontas = [t for t in tarefas if t not in pendentes]

    if prontas:
        print("[CACHE] já completos (split + mapas de id):")
        for nome, _, _ in prontas:
            print(f"         - {nome}")

    if not pendentes:
        print("\nNada a fazer: todos os splits + mapas de appeal_id pedidos já existem.")
        return

    df = load_data(localizar_appeals_file())

    print(f"\nSeeds a usar em baseline/stratified/minority: {seeds}\n")

    def construir_gerador(kind: str, seed):
        if kind == "baseline":
            return lambda: basic_split(df, seed)
        if kind == "stratified":
            return lambda: stratified_split(df, seed)
        if kind == "minority":
            return lambda: minority_split(df, seed)
        if kind == "zeroshot":
            return lambda: zero_shot_split(df)
        raise ValueError(f"kind desconhecido: {kind}")

    for nome, kind, seed in pendentes:
        processar_split(nome, construir_gerador(kind, seed))


if __name__ == "__main__":
    main()