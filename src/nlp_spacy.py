from typing import Any, Dict
import spacy
from spacy.matcher import Matcher


class EntityExtractor:

    def __init__(self, model_name: str = "it_core_news_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"Modello '{model_name}' non trovato. "
                f"Esegui da terminale: python -m spacy download {model_name}"
            )

        self.matcher = Matcher(self.nlp.vocab)
        self._add_custom_patterns()

    def _add_custom_patterns(self):
        """
        Definisce pattern precisi per isolare SOLO il codice dell'ordine.
        """
        # Pattern 1: Cancelletto seguito da cifre (es. #12345)
        pattern_hash = [
            {"TEXT": "#"},
            {"IS_DIGIT": True},
        ]

        # Pattern 2: Codici tipo ORD-1029, ORD1029, ORD_1029 (gestendo la tokenizzazione di spaCy)
        pattern_ord = [
            {"LOWER": {"REGEX": r"^ord[-_]?\d+$"}},
        ]

        # Pattern 3: Sequenza di "ORD" + "-" + numeri (se divisi in 3 token da spaCy)
        pattern_ord_split = [
            {"LOWER": "ord"},
            {"TEXT": "-"},
            {"IS_DIGIT": True},
        ]

        self.matcher.add(
            "ORDER_ID", [pattern_hash, pattern_ord, pattern_ord_split]
        )

    def extract_entities(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        extracted = {}

        # 1. Pattern Matcher personalizzato per ORDER_ID (ha priorità sui dati e-commerce)
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            extracted[string_id.lower()] = span.text

        # 2. Entità NER di spaCy pre-addestrate (filtrando falsi positivi noti sui verbi)
        valid_labels = {"DATE", "LOC", "ORG", "PER", "MONEY"}
        for ent in doc.ents:
            label = ent.label_.upper()
            # Escludiamo parole che iniziano con maiuscole all'inizio della frase ma sono verbi
            if label in valid_labels and ent.text.lower() not in [
                "vorrei",
                "ho",
                "posso",
            ]:
                extracted[label.lower()] = ent.text

        return extracted


if __name__ == "__main__":
    extractor = EntityExtractor()

    test_sentences = [
        "Vorrei sapere dove si trova il mio ordine #12345 acquistato ieri",
        "Ho bisogno di modificare la consegna per l'ordine ORD-1029 a Roma",
        "Vorrei chiedere il rimborso per l'articolo ordinato lunedì",
    ]

    print("--- Test Estrazione Entità Corretto ---\n")
    for sentence in test_sentences:
        entities = extractor.extract_entities(sentence)
        print(f"Testo   : '{sentence}'")
        print(f"Entità  : {entities}\n")