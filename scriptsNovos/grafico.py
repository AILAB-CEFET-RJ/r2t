import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Distribuição de recursos especiais por tema"
    )
    parser.add_argument("--appeals", required=True,
                         help="CSV de recursos (coluna theme_id)")
    parser.add_argument("--output", required=True,
                         help="Caminho da figura de saída (.png/.pdf)")
    args = parser.parse_args()

    df = pd.read_csv(args.appeals, encoding="latin1")
    contagem = df["theme_id"].value_counts().sort_values(ascending=False)

    print(f"Total de temas com pelo menos 1 exemplo: {len(contagem)}")
    print(f"Tema mais frequente: {contagem.index[0]} "
          f"({contagem.iloc[0]} recursos)")
    print(f"Temas com apenas 1 exemplo: {(contagem == 1).sum()}")

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(contagem)), contagem.values, color="#4C9F70")
    plt.xlabel("Tema (ordenado por frequência)")
    plt.ylabel("Frequência")
    plt.title("Distribuição de recursos especiais por tema")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Figura salva em: {args.output}")


if __name__ == "__main__":
    main()