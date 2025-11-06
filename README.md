# iConnectXpert – Intelligent FAQ Assistant

iConneXpert is an AI-powered FAQ bot designed to provide accurate and intelligent responses about **Intelizign iConnect**.  
It leverages **Sentence Transformers** for semantic search and intent classification, and is deployed via **Streamlit** for an interactive UI.

---
## Features
- Semantic similarity search using **Sentence-BERT (all-MiniLM-L6-v2)**.
- Metadata-enhanced training for higher accuracy.
- Achieved **80.77% accuracy** on evaluation.
- Interactive **Streamlit UI** for real-time Q&A.
- "Did you mean?" suggestions for unclear queries.
- Modular training and app deployment setup.

---
## Dataset
- **Domain**: Intelizign iConnect
- **Format**: JSON (`faq_data.json`)
- **Contains**:
  - Intents  
  - Example questions  
  - Responses  
  - Metadata (`category`, `subcategory`, `tags`)
---

## Training the Model

Run the training script (`train6.py`) to generate embeddings and evaluate accuracy:


(```bash
python train6.py)

- Splits dataset (80/20)
- Trains with metadata-enhanced embeddings
- Prints test accuracy (80.77%)


## 💬 Running the Streamlit App

Launch the FAQ bot UI: 
(```bash
streamlit run app2.py)

## Accuracy

- Baseline (without metadata): ~63%
- With metadata enrichment: **80.77%**


