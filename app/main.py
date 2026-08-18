import sys
from pathlib import Path
from typing import Dict, List, Optional

# Aggiunge la radice del progetto a sys.path per evitare "ModuleNotFoundError: No module named 'src'"
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.nlp_spacy import EntityExtractor

# Path del modello salvato dalla baseline
SKLEARN_MODEL_PATH = BASE_DIR / "models" / "baseline_tfidf_logreg.joblib"

# Inizializzazione FastAPI
app = FastAPI(
    title="Intent Classification & Entity Extraction API",
    description="API per Conversational AI con scikit-learn e spaCy",
    version="1.0.0",
)

# Variabili globali per memorizzare i modelli caricati in memoria
classifier_pipeline = None
entity_extractor = None


# -------------------------------------------------------------------
# 1. Models & Schemas con Pydantic
# -------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        example="Vorrei sapere dove si trova il mio ordine #12345 acquistato ieri a Roma",
    )


class EntityItem(BaseModel):
    entity: str = Field(..., example="order_id")
    value: str = Field(..., example="#12345")


class PredictResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    entities: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# -------------------------------------------------------------------
# 2. Lifecycle & Startup Event
# -------------------------------------------------------------------
@app.on_event("startup")
def load_models():
    """Carica i modelli in memoria all'avvio dell'applicazione."""
    global classifier_pipeline, entity_extractor

    # 1. Carica il classificatore di Intent scikit-learn
    if SKLEARN_MODEL_PATH.exists():
        classifier_pipeline = joblib.load(SKLEARN_MODEL_PATH)
        print(f"✅ Modello di classificazione caricato da: {SKLEARN_MODEL_PATH}")
    else:
        print(
            f"⚠️ ATTENZIONE: Modello non trovato in {SKLEARN_MODEL_PATH}. Esegui prima `python3 src/baseline_sklearn.py`"
        )

    # 2. Carica l'Entity Extractor basato su spaCy
    try:
        entity_extractor = EntityExtractor()
        print("✅ EntityExtractor spaCy caricato con successo")
    except Exception as e:
        print(f"⚠️ Errore durante il caricamento di spaCy: {e}")


# -------------------------------------------------------------------
# 3. API Endpoints
# -------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Endpoint per verificare se il servizio e i modelli sono pronti."""
    is_ready = classifier_pipeline is not None and entity_extractor is not None
    return HealthResponse(
        status="ok" if is_ready else "degraded",
        model_loaded=is_ready,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict(request: PredictRequest):
    """
    Riceve un testo di input e restituisce:
    - L'intent identificato con relativo punteggio di confidence
    - Le entità estratte dal testo (es. ORDER_ID, LOC, ecc.)
    """
    if classifier_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modello di classificazione non disponibile.",
        )

    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Il testo di input non può essere vuoto.",
        )

    try:
        # 1. Predizione dell'Intent con scikit-learn
        raw_pred = classifier_pipeline.predict([text])[0]
        intent_pred = str(raw_pred)

        # Calcolo della confidence score
        if hasattr(classifier_pipeline, "predict_proba"):
            probabilities = classifier_pipeline.predict_proba([text])[0]
            confidence = float(max(probabilities))
        else:
            confidence = 1.0

        # 2. Estrazione delle Entità con spaCy
        entities = {}
        if entity_extractor:
            # Invece di: entities = entity_extractor.extract(text)
            entities = entity_extractor.extract_entities(text)

        return PredictResponse(
            text=text,
            intent=intent_pred,
            confidence=round(confidence, 4),
            entities=entities,
        )
    except Exception as e:
        print(f"❌ Errore durante l'elaborazione della richiesta: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore interno durante la prediction: {str(e)}",
        )