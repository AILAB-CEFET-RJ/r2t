import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIGURAÇÕES
# =====================================================

# Arquivo
APPEALS_FILE = "special_appeal.csv"

# Nome das colunas
TEXT_COLUMN = "special_appeal_text"
LABEL_COLUMN = "theme_id"

# Proporção de teste
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Arquivos de saída
OUTPUT_TRAIN = "special_appeal_treino.csv"
OUTPUT_TEST = "special_appeal_teste.csv"


# =====================================================
# FUNÇÕES
# =====================================================

def load_data(filepath):
    """Carrega o dataset de appeals"""
    df = pd.read_csv(filepath)
    print(f"Dataset carregado: {df.shape}")
    return df


def basic_split(df):
    """
    Split simples estratificado (baseline supervisionado)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        #stratify=df[LABEL_COLUMN],
        random_state=RANDOM_STATE
    )

    print(f"Treino: {train_df.shape}")
    print(f"Teste: {test_df.shape}")

    return train_df, test_df


def save_splits(train_df, test_df):
    """Salva os datasets"""
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    print(f"Arquivos salvos:")
    print(f" - {OUTPUT_TRAIN}")
    print(f" - {OUTPUT_TEST}")


# =====================================================
# MAIN
# =====================================================

def main():
    df = load_data(APPEALS_FILE)

    # Caso simples (baseline)
    train_df, test_df = basic_split(df)

    save_splits(train_df, test_df)


if __name__ == "__main__":
    main()