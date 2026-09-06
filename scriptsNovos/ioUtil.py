"""
ioUtil.py

Utilitário mínimo para escrita atômica de CSVs no pipeline GLARE.

Problema que resolve: escrever direto em df.to_csv(caminho_final, ...) deixa
um arquivo TRUNCADO no caminho final se o processo morrer no meio da escrita
(disco cheio, kill, queda de energia etc.) -- e como o resto do pipeline usa
"o arquivo existe" como sinal de "já foi gerado" (ver executar_ou_carregar em
experimento.py e pular_ou_gerar em prepararTreinoVariantes.py), um arquivo
truncado de 0 bytes (ou parcial) fica sendo tratado como válido para sempre,
até alguém notar manualmente (ex: find data -type f -size 0).

Solução: escrever num arquivo temporário no MESMO diretório do destino e só
então renomear para o caminho final com os.replace, que é atômico dentro do
mesmo filesystem -- o caminho final só passa a existir de fato quando o
conteúdo já está 100% gravado em disco. Se o processo morrer antes disso,
sobra um ".tmp" órfão (fácil de identificar e limpar), nunca um CSV
corrompido no lugar esperado.
"""

import os
import uuid
from pathlib import Path


def salvar_csv_atomico(df, caminho, **kwargs) -> None:
    """
    Equivalente a df.to_csv(caminho, **kwargs), mas atômico: se o processo
    for interrompido durante a escrita, o caminho final nunca fica com
    conteúdo parcial -- ou o arquivo final está completo, ou não existe.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    # sufixo aleatório evita colisão entre execuções concorrentes que
    # escrevam para o mesmo caminho ao mesmo tempo
    tmp = caminho.with_name(f"{caminho.name}.tmp.{uuid.uuid4().hex[:8]}")

    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, caminho)  # atômico no mesmo filesystem
    except BaseException:
        # limpa o temporário em qualquer falha (inclusive disco cheio no
        # meio do to_csv) para não deixar lixo acumulando
        tmp.unlink(missing_ok=True)
        raise


def limpar_temporarios_orfaos(raiz) -> int:
    """
    Remove *.tmp.* órfãos deixados por execuções anteriores interrompidas
    de forma anômala (ex: kill -9, queda de energia -- casos em que o
    try/except de salvar_csv_atomico não teve chance de rodar). Retorna
    quantos foram removidos. Seguro rodar a qualquer momento: um .tmp.*
    nunca é lido por nenhum outro script do pipeline.
    """
    raiz = Path(raiz)
    removidos = 0
    for tmp in raiz.rglob("*.tmp.*"):
        if tmp.is_file():
            tmp.unlink()
            removidos += 1
    return removidos