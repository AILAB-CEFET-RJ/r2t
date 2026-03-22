import os
import shutil

# =========================
# Defina o diretório base
# =========================
BASE_DIR = "data"

# Estrutura de diretórios
directories = [
    os.path.join(BASE_DIR, "appeals", "clean", "embeddings"),
    os.path.join(BASE_DIR, "appeals", "clean", "resumos"),
    os.path.join(BASE_DIR, "appeals", "clean", "texto"),
    os.path.join(BASE_DIR, "appeals", "notClean", "embeddings"),
    os.path.join(BASE_DIR, "appeals", "notClean", "resumos"),
    os.path.join(BASE_DIR, "appeals", "notClean", "texto"),
    os.path.join(BASE_DIR, "temas", "clean", "embeddings"),
    os.path.join(BASE_DIR, "temas", "clean", "texto"),
    os.path.join(BASE_DIR, "temas", "notClean", "embeddings"),
    os.path.join(BASE_DIR, "temas", "notClean", "texto"),
]

# =========================
# Criar diretórios
# =========================
for directory in directories:
    os.makedirs(directory, exist_ok=True)

print("Estrutura de diretórios criada com sucesso!")

# =========================
# Caminhos dos arquivos origem
# =========================
arquivo_special = "special_appeal.csv"
arquivo_temas = "temas_repetitivos.csv"

# =========================
# Caminhos de destino
# =========================
destino_special = os.path.join(BASE_DIR, "appeals", "notClean", "texto", "special_appeal.csv")
destino_temas = os.path.join(BASE_DIR, "temas", "notClean", "texto", "temas_repetitivos.csv")

# =========================
# Mover arquivos
# =========================
if os.path.exists(arquivo_special):
    shutil.move(arquivo_special, destino_special)
    print(f"{arquivo_special} movido para {destino_special}")
else:
    print(f"{arquivo_special} não encontrado.")

if os.path.exists(arquivo_temas):
    shutil.move(arquivo_temas, destino_temas)
    print(f"{arquivo_temas} movido para {destino_temas}")
else:
    print(f"{arquivo_temas} não encontrado.")