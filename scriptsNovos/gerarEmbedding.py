import argparse
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =====================================================
# UTIL
# =====================================================

def sanitize_model_name(model_name: str) -> str:
    """Remove caracteres problemáticos do nome do modelo."""
    return model_name.replace("/", "_").replace("-", "_")

def carregar_modelo(model_name: str, device: str):
    """
    Verifica se existe modelo treinado em modelos/{nome_sanitizado}.
    Se existir, carrega ele.
    Caso contrário, carrega modelo original.
    Retorna (model, sufixo_nome_arquivo)
    """

    nome_sanit = sanitize_model_name(model_name)
    caminho_modelo_local = os.path.join("modelos", nome_sanit)

    if os.path.isdir(caminho_modelo_local):
        print(f"📦 Modelo treinado encontrado em: {caminho_modelo_local}")
        model = SentenceTransformer(caminho_modelo_local, device=device)
        return model, "_treinado"
    else:
        print("🌐 Modelo treinado não encontrado. Carregando modelo base.")
        model = SentenceTransformer(model_name, device=device)
        return model, ""

# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Geração de embeddings Sentence-BERT preservando theme_id"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV de entrada (theme_id + texto)"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Nome do modelo SentenceTransformer"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Diretório de saída dos embeddings"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamanho do batch para geração de embeddings"
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo: cuda ou cpu"
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Modelo
    # -------------------------------------------------
    model, sufixo_modelo = carregar_modelo(args.model, args.device)

    # -------------------------------------------------
    # Leitura
    # -------------------------------------------------
    df = pd.read_csv(args.input)

    if "theme_id" not in df.columns:
        raise ValueError("CSV deve conter a coluna 'theme_id'")

    # Detectar coluna de texto
    modelo_sanit = sanitize_model_name(args.model)

    if "special_appeal_text" in df.columns:
        text_col = "special_appeal_text"
        origem = os.path.splitext(os.path.basename(args.input))[0]
        output_name = f"embedding_{modelo_sanit}{sufixo_modelo}__{origem}.csv"

    elif "theme_text" in df.columns:
        text_col = "theme_text"
        output_name = f"embedding_{modelo_sanit}{sufixo_modelo}__themes.csv"

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_name)

    print("=" * 80)
    print("🚀 GERANDO EMBEDDINGS")
    print("=" * 80)
    print(f"📄 Arquivo: {args.input}")
    print(f"🧠 Modelo:  {args.model}")
    print(f"📝 Coluna:  {text_col}")
    print(f"💾 Saída:   {output_path}")
    print("=" * 80)


    texts = df[text_col].astype(str).tolist()
    theme_ids = df["theme_id"].tolist()

    # -------------------------------------------------
    # Embeddings
    # -------------------------------------------------
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    # -------------------------------------------------
    # Montar DataFrame
    # -------------------------------------------------
    df_out = pd.DataFrame(embeddings)
    df_out.insert(0, "theme_id", theme_ids)

    # -------------------------------------------------
    # Salvar
    # -------------------------------------------------
    df_out.to_csv(output_path, index=False)

    print(f"\n✅ Embeddings salvos em: {output_path}")
    print(f"📊 Total de vetores: {len(df_out)}")
    print(f"📐 Dimensão: {embeddings.shape[1]}")

    # Limpeza GPU
    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
