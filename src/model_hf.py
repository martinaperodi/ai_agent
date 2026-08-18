from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def train_hf_model(
    data_path: Path, model_save_dir: Path, artifacts_save_path: Path
):
    # 1. Caricamento Dati
    df = pd.read_csv(data_path)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["intent"])
    X = df["text"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Caricamento Tokenizer e Modello Pre-addestrato di Hugging Face
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"📦 Caricamento modello e tokenizer Hugging Face: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(label_encoder.classes_)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # 3. Tokenizzazione del dataset
    train_encodings = tokenizer(
        X_train, truncation=True, padding=True, max_length=64, return_tensors="pt"
    )
    test_encodings = tokenizer(
        X_test, truncation=True, padding=True, max_length=64, return_tensors="pt"
    )

    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_labels = torch.tensor(y_test, dtype=torch.long)

    # 4. Routine di Fine-Tuning leggeta con PyTorch
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    epochs = 5
    batch_size = 8
    num_train_samples = len(X_train)

    print("\n🚀 Inizio Fine-Tuning del Transformer Hugging Face...\n")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        # Mini-batch manuali
        for i in range(0, num_train_samples, batch_size):
            optimizer.zero_grad()

            input_ids = train_encodings["input_ids"][i : i + batch_size]
            attention_mask = train_encodings["attention_mask"][i : i + batch_size]
            labels = train_labels[i : i + batch_size]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch}/{epochs} | Training Loss: {total_loss / (num_train_samples // batch_size + 1):.4f}"
        )

    # 5. Valutazione sul Test Set
    model.eval()
    with torch.no_grad():
        test_outputs = model(
            input_ids=test_encodings["input_ids"],
            attention_mask=test_encodings["attention_mask"],
        )
        preds = torch.argmax(test_outputs.logits, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    print("\n--- Valutazione Hugging Face Transformer ---")
    print(f"Test Accuracy: {acc:.2%}")
    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test, preds, target_names=label_encoder.classes_
        )
    )

    # 6. Salvataggio del modello e tokenizer Hugging Face
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    artifacts = {
        "label_encoder": label_encoder,
        "model_name": model_name,
    }
    joblib.dump(artifacts, artifacts_save_path)

    print(f"✅ Modello Hugging Face e Tokenizer salvati in: {model_save_dir}")
    print(f"✅ Artefatti salvati in: {artifacts_save_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_file = BASE_DIR / "data" / "intents.csv"
    model_dir = BASE_DIR / "models" / "hf_transformer"
    artifacts_file = BASE_DIR / "models" / "hf_artifacts.joblib"

    train_hf_model(data_file, model_dir, artifacts_file)