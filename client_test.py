import json
import requests

# URL dell'endpoint FastAPI
API_URL = "http://127.0.0.1:8000/predict"

# Lista di frasi di test con intent ed entità differenti
test_queries = [
    "Vorrei sapere dove si trova il mio ordine #12345 acquistato ieri a Roma",
    "Ho bisogno di modificare la consegna per l'ordine ORD-1029 a Milano",
    "Vorrei chiedere il rimborso per l'articolo ordinato lunedì",
    "Voglio parlare subito con un operatore umano",
]


def test_intent_api():
    print("🚀 Inizio test delle chiamate all'API FastAPI...\n")

    for text in test_queries:
        payload = {"text": text}

        try:
            response = requests.post(API_URL, json=payload, timeout=5)

            if response.status_code == 200:
                data = response.json()
                print(f"📝 Testo:      '{data['text']}'")
                print(f"🎯 Intent:     {data['intent']}")
                print(f"📊 Confidence: {data['confidence']:.2%}")
                print(f"🧩 Entità:     {data['entities']}")
                print("-" * 60)
            else:
                print(f"❌ Errore HTTP {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print(
                "❌ Impossibile connettersi all'API. Assicurati che Uvicorn sia attivo su http://127.0.0.1:8000"
            )
            break


if __name__ == "__main__":
    test_intent_api()