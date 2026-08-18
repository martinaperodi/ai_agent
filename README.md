# 🛒 E-Commerce Conversational AI: Intent Classification & Entity Extraction API

An end-to-end solution for **Conversational AI** and **Customer Care Automation** based on Machine Learning, NLP, and Deep Learning.

> 🇮🇹 **Language Note:** The dataset, model pipelines, and spaCy NER rules are currently configured for **Italian language processing**. However, the modular architecture can be easily extended or adapted to support additional languages (e.g., English) by updating the training data and language models.

The system addresses both user intention classification (**Intent Classification**) and structured parameter extraction (**Entity Extraction**), exposing a high-performance hybrid architecture via **FastAPI**.

---

## 📐 System Architecture

The application decouples the problem into two complementary NLP engines exposed through a REST API:

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    Client App / Chatbot                │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                             POST /predict (JSON Payload)
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │                   app/main.py (API)                    │
                               └────────────┬──────────────────────────────┬────────────┘
                                            │                              │
                        (Text)              │                              │ (Text)
                                            ▼                              ▼
             ┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────────────┐
             │         Engine 1: Intent Classifier          │  │       Engine 2: Entity Extractor             │
             │           (src/baseline_sklearn.py)          │  │              (src/nlp_spacy.py)               │
             ├──────────────────────────────────────────────┤  ├──────────────────────────────────────────────┤
             │  • TF-IDF Vectorizer (ngram 1-2)             │  │  • spaCy Pipeline (it_core_news_sm)          │
             │  • Logistic Regression Classifier            │  │  • Custom Matcher (ORDER_ID Regex/Patterns)   │
             └──────────────────────┬───────────────────────┘  └──────────────────────┬───────────────────────┘
                                    │                                                 │
                             (Intent + Prob)                                     (Entities Dict)
                                    │                                                 │
                                    └──────────────────────┬──────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │                 Combined JSON Response                 │
                               │  { "intent": "order_tracking", "entities": {...} }     │
                               └────────────────────────────────────────────────────────┘