import subprocess
import os
import sys
# =========================
# CONFIGURAÇÕES GLOBAIS
# =========================
PYTHON = sys.executable

# Arquivos de entrada
TEMAS_CSV = "data/temas_repetitivos.csv"
APPEALS_CSV = "data/special_appeal.csv"

# Colunas
COLUMN_TEMA = "theme_text"
COLUMN_APPEAL = "special_appeal_text"

# Modelo SBERT
MODEL = "distiluse-base-multilingual-cased-v1"

# Pré-processamento
USAR_CLEAN = True
BEGIN_POINT = "cabimento"   # None para desativar

# Sumarização
TOPIC_TYPE = "L"   # B, G, L, X
TOPIC_SIZE = 5
SEED_LIST = TEMAS_CSV  # necessário para G e X

# Similaridade
SIMILARITY_TYPE = "C"   # B = BM25, C = Cosine
RANK_K = 6

# Verbosidade
VERBOSE = True

# =========================
# FUNÇÃO AUXILIAR
# =========================

def run_command(cmd):
    print("\n========================================")
    print("Executando comando:")
    print(" ".join(cmd))
    print("========================================\n")
    
    result = subprocess.run([PYTHON] + cmd[1:])
    
    if result.returncode != 0:
        raise RuntimeError("Erro na execução do comando")

# =========================
# PASSO 1: EMBEDDINGS
# =========================

def run_create_embedding():
    # Temas
    cmd_temas = [
        "python", "createEmbedding.py",
        TEMAS_CSV,
        "tema",
        COLUMN_TEMA,
        MODEL
    ]
    
    # Apelações
    cmd_appeals = [
        "python", "createEmbedding.py",
        APPEALS_CSV,
        "recurso",
        COLUMN_APPEAL,
        MODEL
    ]
    
    if USAR_CLEAN:
        cmd_temas.append("--clean")
        cmd_appeals.append("--clean")
    
    if BEGIN_POINT:
        cmd_appeals.extend(["--begin_point", BEGIN_POINT])
    
    if VERBOSE:
        cmd_temas.append("-v")
        cmd_appeals.append("-v")
    
    run_command(cmd_temas)
    run_command(cmd_appeals)

# =========================
# PASSO 2: TOPICS / RESUMO
# =========================

def run_create_topics():
    embedding_name = os.path.splitext(os.path.basename(APPEALS_CSV))[0]
    
    if USAR_CLEAN:
        embedding_file = f"data/{embedding_name}_EMBEDDING_CLEAN.pkl"
    else:
        embedding_file = f"data/{embedding_name}_EMBEDDING.pkl"
    
    cmd = [
        "python", "createTopics.py",
        MODEL,
        embedding_file,
        str(TOPIC_SIZE),
        TOPIC_TYPE
    ]
    
    if TOPIC_TYPE in ["G", "X"]:
        cmd.extend(["--seed_list", SEED_LIST])
    
    if VERBOSE:
        cmd.append("-v")
    
    run_command(cmd)

# =========================
# PASSO 3: SIMILARIDADE
# =========================

def run_similarity():
    embedding_name = os.path.splitext(os.path.basename(APPEALS_CSV))[0]
    
    if USAR_CLEAN:
        corpus_file = f"TOPICS_{TOPIC_TYPE}{TOPIC_SIZE}CLEAN.pkl"
        themes_file = f"data/{os.path.splitext(os.path.basename(TEMAS_CSV))[0]}_EMBEDDING_CLEAN.pkl"
    else:
        corpus_file = f"TOPICS_{TOPIC_TYPE}{TOPIC_SIZE}.pkl"
        themes_file = f"data{os.path.splitext(os.path.basename(TEMAS_CSV))[0]}_EMBEDDING.pkl"
    
    cmd = [
        "python", "calcSimilarity.py",
        corpus_file,
        themes_file,
        str(RANK_K),
        SIMILARITY_TYPE
    ]
    
    if VERBOSE:
        cmd.append("-v")
    
    run_command(cmd)

# =========================
# PASSO 4: MÉTRICAS
# =========================

def run_metrics():
    embedding_name = os.path.splitext(os.path.basename(APPEALS_CSV))[0]
    
    if USAR_CLEAN:
        base = f"TOPICS_{TOPIC_TYPE}{TOPIC_SIZE}CLEAN"
    else:
        base = f"TOPICS_{TOPIC_TYPE}{TOPIC_SIZE}"
    
    if SIMILARITY_TYPE == "B":
        file_name = f"CLASSIFIED_{base}_BM25.csv"
    else:
        file_name = f"CLASSIFIED_{base}_COSINE.csv"
    
    cmd = [
        "python", "metrics.py",
        file_name
    ]
    
    if VERBOSE:
        cmd.append("-v")
    
    run_command(cmd)

# =========================
# PIPELINE COMPLETO
# =========================

def run_pipeline():
    run_create_embedding()
    run_create_topics()
    run_similarity()
    run_metrics()

# =========================
# EXECUÇÃO
# =========================

if __name__ == "__main__":
    run_pipeline()