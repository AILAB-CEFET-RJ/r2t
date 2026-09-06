"""
montarAmbiente.py

Orquestra a preparação inicial do ambiente do pipeline GLARE:
  1) splitDados.py     -- gera os splits de treino/teste (baseline,
                           stratified, minority: um por seed; zeroshot: único)
  2) criarDiretorios.py -- cria a estrutura de pastas em data/ e move para
                            lá os CSVs de teste + o CSV de temas
  3) prepararTreinoVariantes.py -- gera as variantes pré-processadas dos
                                    splits de TREINO (clean / resumo por
                                    estratégia), usadas no fine-tuning de
                                    modelos _treinado (ver rodarExperimentos.py)

--config é OBRIGATÓRIO: não existe mais default silencioso nem constante
hardcoded de seeds em nenhum lugar do pipeline (nem aqui, nem em
splitDados.py/criarDiretorios.py). Tudo vem explicitamente do arquivo de
config de ambiente informado (SEEDS e RESUMOS_TREINO), lido UMA VEZ aqui e
repassado como --seeds/--resumos para os 3 passos -- garantindo que os
dois primeiros nunca fiquem dessincronizados entre si.

Seguro para rodar mais de uma vez (ex: para adicionar uma seed nova ou uma
estratégia de resumo nova em SEEDS/RESUMOS_TREINO): splitDados.py e
criarDiretorios.py agora pulam o que já foi processado, e
prepararTreinoVariantes.py já era idempotente.

ATENÇÃO: o passo 3 é bem mais pesado que os dois primeiros -- ele carrega
um SentenceTransformer para gerar resumos extrativos (Guided LexRank usa
BM25 + embeddings) para cada seed/kind de treino, uma vez por estratégia
configurada em RESUMOS_TREINO. Use --pular_variantes se quiser só montar
o esqueleto de pastas/splits rapidamente e rodar essa etapa depois.

Uso:
    python montarAmbiente.py --config config_ambiente.py
    python montarAmbiente.py --config config_ambiente.py --pular_variantes
"""

import argparse
import subprocess
import sys
from pathlib import Path

from configUtil import carregar_modulo

PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent


def rodar(script_nome: str, *args_extra: str) -> None:
    caminho = SCRIPT_DIR / script_nome
    cmd = [PYTHON, str(caminho), *[str(a) for a in args_extra]]
    print(f"\n{'=' * 60}\n>> RODANDO: {' '.join([script_nome, *[str(a) for a in args_extra]])}\n{'=' * 60}")
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


def main():
    p = argparse.ArgumentParser(description="Monta o ambiente inicial do pipeline GLARE.")
    p.add_argument(
        "--config", type=Path, required=True,
        help="Arquivo de config de ambiente (módulo Python com SEEDS e RESUMOS_TREINO). Obrigatório.",
    )
    p.add_argument(
        "--pular_variantes", action="store_true",
        help=(
            "Não roda prepararTreinoVariantes.py (passo 3, pesado -- carrega "
            "modelo de embeddings para gerar resumos). Rode-o manualmente "
            "depois quando quiser."
        ),
    )
    args = p.parse_args()

    config = carregar_modulo(args.config, "config_ambiente")
    seeds = getattr(config, "SEEDS", None)
    resumos_treino = getattr(config, "RESUMOS_TREINO", {})

    if seeds is None:
        p.error(f"{args.config} precisa definir SEEDS (lista de seeds, ex: [42, 7]).")

    print(f"Config de ambiente: {args.config}")
    print(f"  SEEDS = {seeds}")
    print(f"  RESUMOS_TREINO = {resumos_treino}")

    seeds_str = [str(s) for s in seeds]
    resumos_str = [f"{estrategia}:{tamanho}" for estrategia, tamanho in resumos_treino.items()]

    rodar("splitDados.py", "--seeds", *seeds_str)
    rodar("criarDiretorios.py", "--seeds", *seeds_str)

    if args.pular_variantes:
        print("\n--pular_variantes ativo: pulando prepararTreinoVariantes.py.")
    else:
        cmd_variantes = ["--seeds", *seeds_str]
        if resumos_str:
            cmd_variantes += ["--resumos", *resumos_str]
        rodar("prepararTreinoVariantes.py", *cmd_variantes)

    print(f"\n{'=' * 60}\nAmbiente montado com sucesso.\n{'=' * 60}")


if __name__ == "__main__":
    main()