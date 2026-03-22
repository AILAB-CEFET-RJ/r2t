import os
import pandas as pd
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from torch.utils.data import DataLoader
import torch


# ===============================
# CONFIGURAÇÕES
# ===============================

# TEMAS_CSV = "data/temas/notClean/texto/temas_repetitivos.csv"
# APPEALS_CSV = "data/appeals/notClean/texto/special_appeal.csv"
TEMAS_CSV = "data/temas/notClean/texto/temas_repetitivos.csv"
APPEALS_CSV = "special_appeal_treino.csv"
OUTPUT_DIR = "modelos"
BATCH_SIZE = 2
EPOCHS = 2
WARMUP_STEPS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODELOS = [
    'neuralmind/bert-base-portuguese-cased', 
    'pierreguillou/bert-base-cased-pt-lenerbr', 
    'rufimelo/Legal-BERTimbau-base', 
    #'rufimelo/Legal-BERTimbau-large',
]


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


# ===============================
# CARREGAR DADOS
# ===============================

print("Carregando dados...")

temas_df = pd.read_csv(TEMAS_CSV)
appeals_df = pd.read_csv(APPEALS_CSV)

# juntar appeal com texto do tema correto
df = appeals_df.merge(temas_df, on="theme_id")

print(f"Total de pares positivos: {len(df)}")


# ===============================
# CRIAR DATASET DE TREINO
# ===============================

train_examples = []

for _, row in df.iterrows():
    train_examples.append(
        InputExample(
            texts=[row["special_appeal_text"], row["theme_text"]]
        )
    )

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)


# ===============================
# TREINAR MODELOS
# ===============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for model_name in MODELOS:

    print(f"\n==============================")
    print(f"Treinando modelo: {model_name}")
    print(f"==============================")

    sanitized_name = sanitize_model_name(model_name)
    save_path = os.path.join(OUTPUT_DIR, sanitized_name)

    # Se já existir, pula
    if os.path.exists(save_path):
        print("Modelo já treinado. Pulando...")
        continue

    # 🔹 Se já for SentenceTransformer pronto
    if "paraphrase" in model_name:
        model = SentenceTransformer(model_name, device=DEVICE)

    else:
        # 🔹 Transformar BERT em SBERT
        transformer = models.Transformer(model_name)
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        model = SentenceTransformer(
            modules=[transformer, pooling],
            device=DEVICE
        )

    # Loss ideal para retrieval
    train_loss = losses.MultipleNegativesRankingLoss(model)
    #train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        show_progress_bar=True
    )

    model.save(save_path)

    print(f"Modelo salvo em: {save_path}")

print("\nTreinamento finalizado.")
