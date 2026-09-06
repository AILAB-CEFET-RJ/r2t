# Pipeline GLARE — Guia de execução

Este README cobre as **3 etapas obrigatórias** para rodar o pipeline do zero:

1. Instalar as bibliotecas (com PyTorch usando GPU)
2. Montar o ambiente (splits + diretórios + variantes de treino)
3. Executar a grade de experimentos

---

## 1) Instalar as bibliotecas

Script: **`instalarDependencias.py`**

Ele instala tudo que está em `requirements.txt` **e** instala `torch` /
`torchvision` / `torchaudio` com suporte a GPU (CUDA), apontando para o
índice oficial de wheels do PyTorch — um `pip install -r requirements.txt`
comum não garante isso, porque `requirements.txt` não pina a versão de
build do PyTorch.

### Descobrir a versão de CUDA do seu driver

```bash
nvidia-smi
```

Olhe o campo **"CUDA Version"** no canto superior direito da tabela e
escolha a opção correspondente na tabela abaixo (escolha uma igual ou
**anterior** ao número mostrado):

| CUDA Version (nvidia-smi) | Argumento a usar |
|---|---|
| >= 12.6 | `--cuda cu126` |
| >= 12.4 | `--cuda cu124` |
| >= 12.1 | `--cuda cu121` |
| >= 11.8 | `--cuda cu118` |
| sem GPU NVIDIA / sem driver | `--cuda cpu` |

### Rodar

```bash
# autodetecta se há GPU NVIDIA (usa cu121 como padrão se detectar; senão cpu)
python instalarDependencias.py

# ou force a versão exata de CUDA que você identificou no nvidia-smi
python instalarDependencias.py --cuda cu124

# sem GPU
python instalarDependencias.py --cuda cpu

# só ver os comandos pip que seriam rodados, sem instalar nada
python instalarDependencias.py --dry_run
```

Ao final, o script imprime uma checagem automática:

```
torch: 2.x.x+cu121
CUDA disponível: True
GPU: NVIDIA GeForce ...
```

Se `CUDA disponível` vier `False` e você tem GPU, revise a versão passada
em `--cuda` (driver desatualizado costuma ser a causa).

---

## 2) Montar o ambiente

Script: **`montarAmbiente.py`**

Gera os splits de treino/teste, cria a estrutura de pastas em `data/` e
prepara as variantes de pré-processamento do treino (clean/resumo).

**Argumento obrigatório:** `--config` (arquivo de config de ambiente —
define `SEEDS` e `RESUMOS_TREINO`). Use `config_ambiente.py` como
exemplo/base.

Antes de rodar, coloque na raiz do projeto os dois arquivos de
dados-fonte: `special_appeal.csv` e `temas_repetitivos.csv`.

```bash
# monta tudo (splits + diretórios + variantes de treino)
python montarAmbiente.py --config config_ambiente.py

# monta só o esqueleto (splits + diretórios), pulando a etapa pesada
# de gerar variantes de treino (resumos) -- rode-a depois manualmente
# com prepararTreinoVariantes.py se precisar
python montarAmbiente.py --config config_ambiente.py --pular_variantes
```

- Idempotente: pode rodar de novo (ex: para adicionar uma seed nova em
  `SEEDS` dentro de `config_ambiente.py`) — o que já existe é pulado
  (`[CACHE]`).
- Para resetar tudo e voltar ao estado inicial (dados-fonte preservados
  em `src/`, o resto vira backup): `python resetarAmbiente.py`.

---

## 3) Rodar os experimentos

Script: **`rodarExperimentos.py`**

Executa a grade completa de experimentos (dataset × pré-processamento ×
modelo × k, etc.) definida no config de experimentos.

**Argumento:** `--config` (default: `config_experimentos.py`, no mesmo
diretório). Esse arquivo por sua vez **precisa apontar**, no campo
`CONFIG_AMBIENTE`, para o mesmo config de ambiente usado no passo 2 — é
assim que o script sabe com que seeds/estratégia de resumo os splits de
treino foram gerados.

```bash
# roda tudo que ainda não está em resultados/experimentos4.csv
python rodarExperimentos.py --config config_experimentos.py

# só mostra o plano de execução (quais combinações rodariam), sem executar nada
python rodarExperimentos.py --config config_experimentos.py --dry_run

# reroda mesmo combinações já presentes no CSV de resultados
python rodarExperimentos.py --config config_experimentos.py --forcar
```

Outros argumentos opcionais (defaults já apontam para os caminhos
padrão do pipeline, normalmente não precisam ser tocados):
`--base_dir` (default `data`), `--scripts_dir` (default `.`),
`--modelos_dir` (default `modelos`), `--resultados_csv`
(default `resultados/experimentos4.csv`), `--tempos_csv`
(default `resultados/tempos_arquivos.csv`).

Ao final, para gerar as tabelas LaTeX a partir dos resultados:

```bash
python gerarLatex.py
```

---

## Resumo da ordem de execução

```bash
python instalarDependencias.py --cuda <sua_versao_cuda_ou_cpu>
python montarAmbiente.py --config config_ambiente.py
python rodarExperimentos.py --config config_experimentos.py
```