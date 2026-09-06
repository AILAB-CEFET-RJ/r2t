"""
instalarDependencias.py

Instala todas as dependências do pipeline GLARE via pip:

  1) PyTorch (torch, torchvision, torchaudio) com suporte a GPU (CUDA),
     usando o índice oficial da PyTorch para a versão de CUDA escolhida.
  2) O restante das bibliotecas listadas em requirements.txt.

Por que um script separado (e não só `pip install -r requirements.txt`)
--------------------------------------------------------------------
requirements.txt lista "torch", "torchvision" e "torchaudio" sem pinar
versão/build. Um `pip install -r requirements.txt` comum baixaria, na
maioria dos casos, o build padrão do PyPI -- que é CPU-only em muitas
plataformas. Este script:

  - remove essas 3 linhas de requirements.txt antes de instalar o resto
    (para não reinstalar por cima com o build errado);
  - instala torch/torchvision/torchaudio primeiro, apontando
    explicitamente para o índice de wheels do PyTorch compilado com CUDA
    (https://download.pytorch.org/whl/<cuda>).

Como escolher a versão de CUDA (--cuda)
----------------------------------------
Rode `nvidia-smi` no terminal e olhe o campo "CUDA Version" no canto
superior direito da tabela -- é a versão MÁXIMA suportada pelo seu
driver. Escolha uma opção abaixo igual ou anterior a esse número:

    CUDA Version (nvidia-smi) >= 12.6   ->  --cuda cu126
    CUDA Version (nvidia-smi) >= 12.4   ->  --cuda cu124
    CUDA Version (nvidia-smi) >= 12.1   ->  --cuda cu121
    CUDA Version (nvidia-smi) >= 11.8   ->  --cuda cu118
    sem GPU NVIDIA / driver não instalado -> --cuda cpu

Se --cuda não for informado, o script tenta detectar automaticamente via
`nvidia-smi` e usa cu121 como aposta segura (compatível com a grande
maioria dos drivers de 2023 em diante); se não achar `nvidia-smi` no
PATH, cai para CPU e avisa.

Uso:
    python instalarDependencias.py                  # autodetecta
    python instalarDependencias.py --cuda cu121      # força uma versão
    python instalarDependencias.py --cuda cpu        # força CPU (sem GPU)
    python instalarDependencias.py --dry_run         # só mostra os comandos
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = SCRIPT_DIR / "requirements.txt"

# Pacotes do PyTorch que precisam vir do índice especial de CUDA -- são
# removidos de requirements.txt antes do pip install genérico, para não
# serem reinstalados por cima com o build CPU-only do PyPI.
PACOTES_TORCH = {"torch", "torchvision", "torchaudio"}

CUDA_VALIDAS = ["cu126", "cu124", "cu121", "cu118", "cpu"]
CUDA_PADRAO = "cu121"

INDEX_URL_BASE = "https://download.pytorch.org/whl"


def detectar_cuda() -> str:
    """
    Tenta detectar se há GPU NVIDIA disponível via `nvidia-smi`. Não lê a
    versão exata de CUDA do driver (isso varia demais em formatação) --
    só usa a presença do comando como sinal de "tem GPU NVIDIA", e nesse
    caso assume CUDA_PADRAO (cu121), que funciona com a grande maioria
    dos drivers atuais. Sem `nvidia-smi` no PATH, assume --cuda cpu.
    """
    if shutil.which("nvidia-smi") is None:
        print("[AVISO] `nvidia-smi` não encontrado no PATH -- assumindo "
              "que não há GPU NVIDIA disponível. Instalando build CPU.")
        return "cpu"

    try:
        subprocess.run(
            ["nvidia-smi"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[AVISO] `nvidia-smi` encontrado mas falhou ao rodar -- "
              "assumindo CPU.")
        return "cpu"

    print(f"[INFO] GPU NVIDIA detectada. Usando build CUDA padrão: {CUDA_PADRAO}\n"
          f"       Se souber a versão exata suportada pelo seu driver, "
          f"rode de novo com --cuda (veja o topo deste script).")
    return CUDA_PADRAO


def ler_requirements_sem_torch(caminho: Path) -> list:
    """
    Lê requirements.txt e devolve a lista de linhas (nomes de pacote),
    excluindo torch/torchvision/torchaudio -- essas são instaladas à
    parte, com o índice de CUDA correto.
    """
    if not caminho.is_file():
        raise FileNotFoundError(f"requirements.txt não encontrado em: {caminho}")

    linhas_finais = []
    for linha_bruta in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha_bruta.strip()
        if not linha or linha.startswith("#"):
            continue
        # nome do pacote = parte antes de qualquer especificador de versão
        nome_pacote = re.split(r"[=<>~!\[]", linha, maxsplit=1)[0].strip().lower()
        if nome_pacote in PACOTES_TORCH:
            continue
        linhas_finais.append(linha)
    return linhas_finais


def rodar(cmd: list, dry_run: bool) -> None:
    cmd = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Instala as dependências do pipeline GLARE (incluindo PyTorch com GPU)."
    )
    p.add_argument(
        "--cuda", choices=CUDA_VALIDAS, default=None,
        help=(
            "Versão de build do PyTorch a instalar: cu126, cu124, cu121, "
            "cu118 (com GPU) ou cpu (sem GPU). Se omitido, tenta detectar "
            f"automaticamente (default de GPU se detectada: {CUDA_PADRAO})."
        ),
    )
    p.add_argument(
        "--requirements", type=Path, default=REQUIREMENTS_PATH,
        help=f"Caminho do requirements.txt (default: {REQUIREMENTS_PATH}).",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Só mostra os comandos pip que seriam executados, sem instalar nada.",
    )
    args = p.parse_args(argv)

    cuda = args.cuda or detectar_cuda()

    print(f"\n{'=' * 60}\nInstalando PyTorch (build: {cuda})\n{'=' * 60}")

    cmd_torch = [PYTHON, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    if cuda != "cpu":
        cmd_torch += ["--index-url", f"{INDEX_URL_BASE}/{cuda}"]
    else:
        cmd_torch += ["--index-url", f"{INDEX_URL_BASE}/cpu"]
    rodar(cmd_torch, args.dry_run)

    print(f"\n{'=' * 60}\nInstalando o restante de requirements.txt\n{'=' * 60}")
    pacotes = ler_requirements_sem_torch(args.requirements)
    cmd_resto = [PYTHON, "-m", "pip", "install", *pacotes]
    rodar(cmd_resto, args.dry_run)

    if args.dry_run:
        print("\n--dry_run ativo: nada foi instalado de fato.")
        return

    print(f"\n{'=' * 60}\nInstalação concluída.\n{'=' * 60}")
    print("Verificando se o PyTorch enxerga a GPU...")
    verificacao = (
        "import torch; "
        "print('torch:', torch.__version__); "
        "print('CUDA disponível:', torch.cuda.is_available()); "
        "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma')"
    )
    subprocess.run([PYTHON, "-c", verificacao])


if __name__ == "__main__":
    main()