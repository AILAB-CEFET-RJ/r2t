import subprocess
from pathlib import Path
import itertools
import time
from datetime import datetime
import sys
import os
import csv

# =====================================================
# CONFIGURAÇÕES
# =====================================================

PYTHON = "python"

MODELOS = [
    'distiluse-base-multilingual-cased-v1',
    'paraphrase-multilingual-mpnet-base-v2',
    # 'paraphrase-multilingual-MiniLM-L12-v2',
    # 'neuralmind/bert-base-portuguese-cased',
    'pierreguillou/bert-base-cased-pt-lenerbr',
    'rufimelo/Legal-BERTimbau-base',
    # 'rufimelo/Legal-BERTimbau-large',
]

RESUMO_ESTRATEGIAS = ["lexrank"]
RESUMO_TAMANHOS = [10]

BASE = Path("data")

APPEALS_ORIG = BASE / "appeals" / "notClean" / "texto" / "special_appeal_teste.csv"
TEMAS_ORIG   = BASE / "temas"   / "notClean" / "texto" / "temas_repetitivos.csv"

APPEALS_CLEAN = BASE / "appeals" / "clean" / "texto" / "special_appeal_teste.csv"
TEMAS_CLEAN   = BASE / "temas"   / "clean" / "texto" / "temas_repetitivos.csv"

LOG_FILE = f"logs/pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

TIME_CSV = f"logs/pipeline_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# =====================================================
# UTIL
# =====================================================

def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")

def run(cmd):
    subprocess.run([sys.executable] + cmd[1:], check=True)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def executar_se_nao_existir(output_path: Path, cmd: list, registros: list, tempos: dict, input_path: Path):
    if output_path.exists():
        print(f"Pulando (já existe): {output_path}")
        return

    print(f"Gerando: {output_path}")
    inicio = time.time()

    run(cmd)

    fim = time.time()
    duracao = fim - inicio

    registros.append((str(output_path), duracao))

    # -------------------------------
    # PROPAGAÇÃO DE TEMPO
    # -------------------------------
    tempo_input = tempos.get(str(input_path), 0)
    tempos[str(output_path)] = tempo_input + duracao

# =====================================================
# PASSO 1 — CLEAN
# =====================================================

def gerar_clean(registros, tempos):

    ensure_dir(APPEALS_CLEAN.parent)
    ensure_dir(TEMAS_CLEAN.parent)

    executar_se_nao_existir(
        APPEALS_CLEAN,
        [
            PYTHON, "limparTexto.py",
            "--input", str(APPEALS_ORIG),
            "--output", str(APPEALS_CLEAN),
            "--text_column", "special_appeal_text"
        ],
        registros,
        tempos,
        APPEALS_ORIG
    )

    executar_se_nao_existir(
        TEMAS_CLEAN,
        [
            PYTHON, "limparTexto.py",
            "--input", str(TEMAS_ORIG),
            "--output", str(TEMAS_CLEAN),
            "--text_column", "theme_text"
        ],
        registros,
        tempos,
        TEMAS_ORIG
    )

# =====================================================
# PASSO 2 — EMBEDDINGS TEXTOS
# =====================================================

def gerar_embeddings_textos(registros, tempos):

    for modelo in MODELOS:

        modelo_sanit = sanitize_model_name(modelo)

        for origem_nome, input_path in [
            ("appeals_notClean", APPEALS_ORIG),
            ("appeals_clean", APPEALS_CLEAN),
            ("temas_notClean", TEMAS_ORIG),
            ("temas_clean", TEMAS_CLEAN),
        ]:

            if "appeals" in origem_nome:
                pasta_base = BASE / "appeals"
            else:
                pasta_base = BASE / "temas"

            clean_flag = "clean" if "clean" in origem_nome else "notClean"
            embed_dir = pasta_base / clean_flag / "embeddings"
            ensure_dir(embed_dir)

            nome_csv = input_path.stem

            # padrao do gerarEmbedding.py
            if "appeal" in nome_csv:
                sufixo_modelo = "_treinado" if os.path.isdir(f"modelos/{modelo_sanit}") else ""
                output_name = f"embedding_{modelo_sanit}{sufixo_modelo}__{nome_csv}.csv"
            else:
                sufixo_modelo = "_treinado" if os.path.isdir(f"modelos/{modelo_sanit}") else ""
                output_name = f"embedding_{modelo_sanit}{sufixo_modelo}__themes.csv"

            output_path = embed_dir / output_name
            executar_se_nao_existir(
                output_path,
                [
                    PYTHON, "gerarEmbedding.py",
                    "--input", str(input_path),
                    "--model", modelo,
                    "--output_dir", str(embed_dir)
                ],
                registros,
                tempos,
                input_path
            )

# =====================================================
# PASSO 3 — RESUMOS (APPEALS)
# =====================================================

def gerar_resumos(registros, tempos):

    for clean_flag, input_path in [
        ("clean", APPEALS_CLEAN),
        ("notClean", APPEALS_ORIG),
    ]:

        resumo_dir = BASE / "appeals" / clean_flag / "resumos"
        ensure_dir(resumo_dir)

        for estrategia, tamanho in itertools.product(
            RESUMO_ESTRATEGIAS,
            RESUMO_TAMANHOS
        ):

            output_name = f"resumo_{estrategia}_{tamanho}.csv"
            output_path = resumo_dir / output_name

            executar_se_nao_existir(
                output_path,
                [
                    PYTHON, "gerarResumos.py",
                    "--input", str(input_path),
                    "--output_dir", str(resumo_dir),
                    "--size", str(tamanho),
                    "--strategy", estrategia,
                    "--model", "distiluse-base-multilingual-cased-v1"
                ],
                registros,
                tempos,
                input_path
            )

# =====================================================
# PASSO 4 — EMBEDDINGS DOS RESUMOS
# =====================================================

def gerar_embeddings_resumos(registros, tempos):

    for modelo in MODELOS:

        modelo_sanit = sanitize_model_name(modelo)

        for clean_flag in ["clean", "notClean"]:

            resumo_dir = BASE / "appeals" / clean_flag / "resumos"
            embed_dir  = BASE / "appeals" / clean_flag / "embeddings"
            ensure_dir(embed_dir)

            for resumo_file in resumo_dir.glob("resumo_*.csv"):

                nome_csv = resumo_file.stem
                output_name = f"embedding_{modelo_sanit}_{nome_csv}.csv"
                output_path = embed_dir / output_name

                executar_se_nao_existir(
                    output_path,
                    [
                        PYTHON, "gerarEmbedding.py",
                        "--input", str(resumo_file),
                        "--model", modelo,
                        "--output_dir", str(embed_dir)
                    ],
                    registros,
                    tempos,
                    resumo_file
                )

# =====================================================
# MAIN
# =====================================================

def main():
    inicio_total = time.time()
    registros = []

    tempos = {}
    tempos[str(APPEALS_ORIG)] = 0
    tempos[str(TEMAS_ORIG)] = 0

    gerar_clean(registros, tempos)
    gerar_embeddings_textos(registros, tempos)
    gerar_resumos(registros, tempos)
    gerar_embeddings_resumos(registros, tempos)

    fim_total = time.time()
    tempo_total = fim_total - inicio_total

    # -------------------------------------------------
    # LOG FINAL
    # -------------------------------------------------

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("PIPELINE EXECUTION LOG\n")
        f.write("="*60 + "\n\n")

        for arquivo, tempo in registros:
            f.write(f"{arquivo} | {tempo:.2f} segundos\n")

        f.write("\n" + "="*60 + "\n")
        f.write(f"Tempo total: {tempo_total:.2f} segundos\n")
    
    # -------------------------------------------------
    # CSV DE TEMPOS
    # -------------------------------------------------
    
    with open(TIME_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arquivo", "tempo_total_segundos"])
    
        for arquivo, tempo in tempos.items():
            writer.writerow([arquivo, f"{tempo:.4f}"])
    
    print(f"CSV de tempos salvo em: {TIME_CSV}")

    print("\n" + "="*80)
    print("PIPELINE FINALIZADO")
    print(f"Log salvo em: {LOG_FILE}")
    print("="*80)

if __name__ == "__main__":
    main()
