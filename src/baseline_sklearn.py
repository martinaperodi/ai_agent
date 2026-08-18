
import joblib
# joblib: Serve per serializzare (salvare su disco in formato binario) e deserializzare (ricaricare) oggetti Python. Invece di dover riaddestrare il modello ogni volta che riavvii l'applicazione o ricevi una chiamata API, salvi la pipeline addestrata e la ricarichi all'istante.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from pathlib import Path


def train_baseline(data_path: str, model_output_path: str):
    # 1. Caricamento Dati
    df = pd.read_csv(data_path)
    X = df["text"]
    y = df["intent"]

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        # Mantiene le stesse proporzioni tra le classi sia nel train set che nel test set. Se nel dataset originale il 20% delle frasi è order_tracking, anche nel train e nel test set ci sarà esattamente il 20% di order_tracking.
    )

    # 3. Definiamo la Pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
            ),  # Unigrammi + Bigrammi
            (
                "clf",
                LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                # parametro c è per controllare complessità del modello e evitare overfitting
            ),
        ]
    )

    # 4. Addestramento
    print("Addestramento baseline TF-IDF + Logistic Regression in corso...")
    pipeline.fit(X_train, y_train)

    # 5. Valutazione
    y_pred = pipeline.predict(X_test)
    print("\n--- Evaluation Report ---")
    print(classification_report(y_test, y_pred))

    # Precision: Di tutte le volte che il modello ha detto "order_tracking", quante volte ci ha azzeccato?
    # Recall: Di tutti gli effettivi messaggi di "order_tracking", quanti ne ha intercettati?
    # F1-Score: La media armonica tra Precision e Recall.

    # 6. Serializzazione
    joblib.dump(pipeline, model_output_path)
    print(f"Modello salvato con successo in: {model_output_path}")


if __name__ == "__main__":
    # Calcola la cartella radice del progetto
    BASE_DIR = Path(__file__).resolve().parent.parent

    data_path = BASE_DIR / "data" / "intents.csv"
    model_path = BASE_DIR / "models" / "baseline_tfidf_logreg.joblib"

    # Crea la cartella 'models' se non esiste ancora
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_baseline(str(data_path), str(model_path))


