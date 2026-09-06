"""
gerarLatex.py

Gera tabelas LaTeX a partir de resultados/experimentos4.csv, em duas
variantes:

  1) RAW (tabelas_latex/raw/): uma tabela por combinação exata de
     (dataset, usar_clean, origem-do-treino) -- sem nenhuma agregação.
     Com múltiplas seeds em baseline/stratified/minority, "dataset" inclui
     a seed (ex: "baseline_teste_42"), então cada seed gera sua própria
     tabela raw.

     "origem-do-treino" é:
       - "base"  -- para modelos SEM fine-tuning (embedding pronto).
       - "treinado_<treino_variante>" -- para modelos _treinado, UMA
         tabela por valor de treino_variante presente no CSV (raw, clean,
         resumo_lexrank, resumo_guided_lexrank, clean_resumo_lexrank,
         clean_resumo_guided_lexrank -- ver prepararTreinoVariantes.py).
         Modelos diferentes treinados com a MESMA variante aparecem juntos,
         uma linha por modelo, na mesma tabela; a MESMA base de modelo
         treinada em variantes diferentes gera tabelas separadas -- nunca
         são combinadas, pois teriam o mesmo nome de modelo na mesma
         tabela sem nenhuma forma de diferenciá-las.

  2) AGREGADA (tabelas_latex/agregado/): mesma partição acima, mas por
     (kind, usar_clean, origem-do-treino), onde "kind" é o split sem a
     seed (baseline/stratified/minority/zeroshot). Para baseline/
     stratified/minority, cada célula reporta média ± desvio padrão das
     métricas calculadas entre as seeds disponíveis PARA AQUELA variante
     específica, por modelo (uma treino_variante pode estar disponível em
     menos seeds que outra, se ainda não foi totalmente rodada). zeroshot
     não tem seed (é um único split fixo), então a tabela agregada dele
     coincide com a raw -- reportada mesmo assim por uniformidade, com
     desvio "-" (não há repetição para calcular desvio-padrão).

CSVs gerados antes da introdução do eixo treino_variante (sem essa coluna)
são tratados como se toda linha fosse treino_variante="raw" -- o único
valor que existia implicitamente antes desse eixo existir.

O nome de "dataset" precisa seguir o padrão "<kind>_teste" ou
"<kind>_teste_<seed>" (kind ∈ {baseline, stratified, minority, zeroshot})
para que a extração de kind/seed funcione. Esse padrão é o gerado por
rodarExperimentos.py a partir de config_experimentos.py (KINDS/SEEDS).
"""

import os
import re
import pandas as pd
import numpy as np

CSV_PATH = "resultados/experimentos4.csv"
OUTPUT_DIR = "tabelas_latex"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
AGREGADO_DIR = os.path.join(OUTPUT_DIR, "agregado")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(AGREGADO_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# CSVs gerados antes da introdução do eixo treino_variante não têm essa
# coluna -- trata como "raw" (única variante que existia implicitamente
# antes: todo modelo _treinado era treinado sobre o dataset cru).
if "treino_variante" not in df.columns:
    df["treino_variante"] = "raw"
df["treino_variante"] = df["treino_variante"].fillna("raw")

# Ordem de exibição preferida das variantes de treino nas tabelas/arquivos
# (variantes fora dessa lista aparecem depois, em ordem alfabética).
ORDEM_TREINO_VARIANTE = [
    "raw", "clean",
    "resumo_lexrank", "resumo_guided_lexrank",
    "clean_resumo_lexrank", "clean_resumo_guided_lexrank",
]


def ordenar_variantes(variantes) -> list:
    def chave(v):
        try:
            return (0, ORDEM_TREINO_VARIANTE.index(v))
        except ValueError:
            return (1, v)
    return sorted(variantes, key=chave)

# métricas que vão para tabela
METRICS = ["recall_at_k", "map_at_k", "ndcg_at_k", "f1_score", "tempo"]

# Kinds de split conhecidos -- usados para extrair (kind, seed) do nome
# de dataset. Mantido em sincronia manual com SPLIT_KINDS_COM_SEED em
# criarDiretorios.py / splitDados.py.
SPLIT_KINDS = ["baseline", "stratified", "minority", "zeroshot"]

PADRAO_DATASET = re.compile(
    r"^(?P<kind>" + "|".join(SPLIT_KINDS) + r")_teste(?:_(?P<seed>\d+))?$"
)


# ---------------------------
# Função para escape LaTeX
# ---------------------------
def escape_latex(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("_", r"\_")
    return text


def extrair_kind_seed(dataset_nome: str):
    """
    Extrai (kind, seed) do nome do dataset.
    seed é None para zeroshot (ou para qualquer dataset sem sufixo de seed).
    Retorna (None, None) se o nome não bater com o padrão esperado --
    nesse caso, o dataset é reportado apenas na tabela raw.
    """
    m = PADRAO_DATASET.match(dataset_nome)
    if not m:
        return None, None
    kind = m.group("kind")
    seed = m.group("seed")
    return kind, (int(seed) if seed is not None else None)


# =====================================================
# 1) TABELAS RAW -- uma por dataset exato (comportamento original)
# =====================================================

def gerar_tabela_latex_raw(df_subset, title):
    df_subset = df_subset.copy()
    df_subset = df_subset[["modelo"] + METRICS]
    df_subset["modelo"] = df_subset["modelo"].apply(escape_latex)

    header = (
        "\\begin{table}\n"
        "\\centering\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Modelo & Recall@k & MAP@k & nDCG@k & F1 & Tempo \\\\\n"
        "\\midrule\n"
    )

    rows = ""
    for _, row in df_subset.iterrows():
        rows += (
            f"{row['modelo']} & "
            f"{row['recall_at_k']:.6f} & "
            f"{row['map_at_k']:.6f} & "
            f"{row['ndcg_at_k']:.6f} & "
            f"{row['f1_score']:.6f} & "
            f"{row['tempo']:.2f} \\\\\n"
        )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{title.replace('_', ' ')}}}\n"
        "\\end{table}\n"
    )

    return header + rows + footer


def gerar_raw():
    datasets = df["dataset"].unique()
    n_gerado = 0

    for dataset in datasets:
        df_dataset = df[df["dataset"] == dataset]

        for clean in [True, False]:
            subset_clean = df_dataset[df_dataset["usar_clean"] == clean]

            # ---- modelos BASE: sem eixo de treino, uma tabela só ----
            subset_base = subset_clean[~subset_clean["modelo"].str.contains("_treinado", na=False)]
            if not subset_base.empty:
                title = f"{dataset} | clean={clean} | base"
                latex_table = gerar_tabela_latex_raw(subset_base, title)
                filename = f"{dataset}_clean{clean}_base.tex"
                with open(os.path.join(RAW_DIR, filename), "w", encoding="utf-8") as f:
                    f.write(latex_table)
                n_gerado += 1

            # ---- modelos TREINADOS: uma tabela por origem do dataset de
            # treino (raw / clean / resumo_<estrategia> / etc.) --
            # combinar diferentes treino_variante numa mesma tabela
            # produziria linhas duplicadas e ambíguas para o mesmo modelo.
            subset_treinado = subset_clean[subset_clean["modelo"].str.contains("_treinado", na=False)]
            for variante in ordenar_variantes(subset_treinado["treino_variante"].unique()):
                subset_variante = subset_treinado[subset_treinado["treino_variante"] == variante]
                if subset_variante.empty:
                    continue

                title = f"{dataset} | clean={clean} | treinado | treino={variante}"
                latex_table = gerar_tabela_latex_raw(subset_variante, title)

                filename = f"{dataset}_clean{clean}_treinado_{variante}.tex"
                with open(os.path.join(RAW_DIR, filename), "w", encoding="utf-8") as f:
                    f.write(latex_table)
                n_gerado += 1

    print(f"[RAW] {n_gerado} tabelas geradas em: {RAW_DIR}")


# =====================================================
# 2) TABELAS AGREGADAS -- média ± desvio padrão entre seeds, por kind
# =====================================================

def formatar_media_desvio(valores: pd.Series, casas: int = 6) -> str:
    """
    Formata uma célula "média ± desvio" a partir dos valores de uma
    métrica coletados entre as seeds disponíveis para (kind, modelo).
    Com apenas 1 valor (ex: zeroshot, ou seed única disponível), o
    desvio-padrão não é calculável -- reporta "-" no lugar.
    """
    media = valores.mean()
    if len(valores) > 1:
        desvio = valores.std(ddof=1)
        return f"${media:.{casas}f} \\pm {desvio:.{casas}f}$"
    return f"${media:.{casas}f} \\pm -$"


def gerar_tabela_latex_agregada(df_subset, title, n_seeds: int):
    df_subset = df_subset.copy()

    linhas_por_modelo = []
    for modelo, grupo in df_subset.groupby("modelo", sort=True):
        linha = {"modelo": escape_latex(modelo)}
        for metrica in METRICS:
            casas = 2 if metrica == "tempo" else 6
            linha[metrica] = formatar_media_desvio(grupo[metrica], casas=casas)
        linha["_n"] = len(grupo)
        linhas_por_modelo.append(linha)

    header = (
        "\\begin{table}\n"
        "\\centering\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Modelo & Recall@k & MAP@k & nDCG@k & F1 & Tempo \\\\\n"
        "\\midrule\n"
    )

    rows = ""
    for linha in linhas_por_modelo:
        rows += (
            f"{linha['modelo']} & "
            f"{linha['recall_at_k']} & "
            f"{linha['map_at_k']} & "
            f"{linha['ndcg_at_k']} & "
            f"{linha['f1_score']} & "
            f"{linha['tempo']} \\\\\n"
        )

    nota_seeds = (
        f"Agregado sobre {n_seeds} seed(s) de split."
        if n_seeds > 1 else
        "Split único, sem repetição de seed (desvio não aplicável)."
    )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{title.replace('_', ' ')} -- {nota_seeds}}}\n"
        "\\end{table}\n"
    )

    return header + rows + footer


def gerar_agregado():
    # Extrai kind/seed de cada linha do CSV
    kinds_seeds = df["dataset"].apply(extrair_kind_seed)
    df_agg = df.copy()
    df_agg["_kind"] = kinds_seeds.apply(lambda t: t[0])
    df_agg["_seed"] = kinds_seeds.apply(lambda t: t[1])

    nao_reconhecidos = df_agg[df_agg["_kind"].isna()]["dataset"].unique()
    if len(nao_reconhecidos) > 0:
        print(
            f"[AVISO] {len(nao_reconhecidos)} dataset(s) não bateram com o "
            f"padrão <kind>_teste[_<seed>] e foram ignorados na agregação: "
            f"{list(nao_reconhecidos)}"
        )

    df_agg = df_agg[df_agg["_kind"].notna()]

    n_gerado = 0
    for kind in sorted(df_agg["_kind"].unique()):
        df_kind = df_agg[df_agg["_kind"] == kind]

        for clean in [True, False]:
            subset_clean = df_kind[df_kind["usar_clean"] == clean]

            # ---- modelos BASE: sem eixo de treino ----
            subset_base = subset_clean[~subset_clean["modelo"].str.contains("_treinado", na=False)]
            if not subset_base.empty:
                n_seeds = subset_base["_seed"].nunique() if kind != "zeroshot" else 1
                title = f"{kind} | clean={clean} | base"
                latex_table = gerar_tabela_latex_agregada(subset_base, title, n_seeds=n_seeds)
                filename = f"{kind}_clean{clean}_base.tex"
                with open(os.path.join(AGREGADO_DIR, filename), "w", encoding="utf-8") as f:
                    f.write(latex_table)
                n_gerado += 1

            # ---- modelos TREINADOS: uma tabela por origem do dataset de
            # treino. n_seeds é recalculado PARA CADA variante -- nem toda
            # seed necessariamente já tem aquela variante gerada/rodada,
            # então usar a contagem do "kind" inteiro superestimaria a
            # nota de rodapé quando uma variante está parcialmente pronta.
            subset_treinado = subset_clean[subset_clean["modelo"].str.contains("_treinado", na=False)]
            for variante in ordenar_variantes(subset_treinado["treino_variante"].unique()):
                subset_variante = subset_treinado[subset_treinado["treino_variante"] == variante]
                if subset_variante.empty:
                    continue

                n_seeds = subset_variante["_seed"].nunique() if kind != "zeroshot" else 1
                title = f"{kind} | clean={clean} | treinado | treino={variante}"
                latex_table = gerar_tabela_latex_agregada(subset_variante, title, n_seeds=n_seeds)
                filename = f"{kind}_clean{clean}_treinado_{variante}.tex"
                with open(os.path.join(AGREGADO_DIR, filename), "w", encoding="utf-8") as f:
                    f.write(latex_table)
                n_gerado += 1

    print(f"[AGREGADO] {n_gerado} tabelas geradas em: {AGREGADO_DIR}")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    gerar_raw()
    gerar_agregado()