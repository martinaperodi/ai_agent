from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# 1. Vocabolario ed Utilità di Tokenizzazione
class Vocabulary:

    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx: Dict[str, int] = {pad_token: 0, unk_token: 1}
        self.idx2word: Dict[int, str] = {0: pad_token, 1: unk_token}

    def build_vocab(self, texts: List[str]):
        """Costruisce il vocabolario mappando ogni parola unica ad un indice intero."""
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def encode(self, text: str) -> List[int]:
        """Converte una stringa di testo in una lista di indici interi."""
        return [
            self.word2idx.get(w, self.word2idx[self.unk_token])
            for w in text.lower().split()
        ]

    def __len__(self):
        return len(self.word2idx)



# 2. Custom Dataset per PyTorch
class IntentDataset(Dataset):

    def __init__(
        self, texts: List[str], labels: List[int], vocab: Vocabulary
    ):
        self.vocab = vocab
        self.labels = labels
        self.encoded_texts = [vocab.encode(t) for t in texts]

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def collate_fn(batch):
    """
    Collate function personalizzata per gestire sequenze di lunghezza variabile.
    Usa EmbeddingBag che richiede i vettori concatenati e gli offset di inizio frase.
    """
    texts, labels = zip(*batch)
    offsets = [0] + [len(t) for t in texts[:-1]]
    offsets = torch.tensor(offsets, dtype=torch.long).cumsum(dim=0)
    texts = torch.cat(texts)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, offsets, labels



# 3. Architettura della Rete Neurale
class IntentClassifierPyTorch(nn.Module):

    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int
    ):
        super().__init__()
        # EmbeddingBag calcola la media dei vettori embedding delle parole nella frase
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, mode="mean"
        )
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, text: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.embedding(text, offsets)
        out = self.fc1(embedded)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out)


# 4. Routine di Training ed Evaluation
def train_pytorch_model(
    data_path: Path, model_save_path: Path, artifacts_save_path: Path
):
    # Caricamento e preparazione dati
    df = pd.read_csv(data_path)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df["intent"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        labels_encoded,
        test_size=0.2,
        random_state=42,
        stratify=labels_encoded,
    )

    # Costruzione Vocabolario
    vocab = Vocabulary()
    vocab.build_vocab(X_train)

    train_dataset = IntentDataset(X_train, y_train, vocab)
    test_dataset = IntentDataset(X_test, y_test, vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )

    # Inizializzazione Modello, Loss e Optimizer
    num_classes = len(label_encoder.classes_)
    model = IntentClassifierPyTorch(
        vocab_size=len(vocab),
        embed_dim=64,
        hidden_dim=32,
        num_classes=num_classes,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Inizio addestramento modello PyTorch...\n")
    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for texts, offsets, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(texts, offsets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == epochs:
            # Valutazione rapida
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for texts, offsets, targets in test_loader:
                    outputs = model(texts, offsets)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            acc = correct / total
            print(
                f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Test Accuracy: {acc:.2%}"
            )

    # Salvataggio pesi e artefatti (Vocabolario + LabelEncoder)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    artifacts = {
        "vocab": vocab,
        "label_encoder": label_encoder,
        "vocab_size": len(vocab),
        "embed_dim": 64,
        "hidden_dim": 32,
        "num_classes": num_classes,
    }
    joblib.dump(artifacts, artifacts_save_path)

    print(f"\nModello PyTorch salvato in: {model_save_path}")
    print(f"Artefatti salvati in: {artifacts_save_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_file = BASE_DIR / "data" / "intents.csv"
    model_file = BASE_DIR / "models" / "pytorch_intent_model.pt"
    artifacts_file = BASE_DIR / "models" / "pytorch_artifacts.joblib"

    train_pytorch_model(data_file, model_file, artifacts_file)