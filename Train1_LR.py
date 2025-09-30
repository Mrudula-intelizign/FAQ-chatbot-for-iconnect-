import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load dataset
# -------------------------------
# with open("faq_data.json", "r", encoding="utf-8") as f:
#     faq_data = json.load(f)["faqs"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "faq_data.json"), "r", encoding="utf-8") as f:
    faq_data = json.load(f)["faqs"]


sentences, labels = [], []
for item in faq_data:
    for ex in item["examples"]:
        sentences.append(ex)
        labels.append(item["intent"])

# -------------------------------
# Encode with SBERT
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X = embedder.encode(sentences)
y = np.array(labels)

# -------------------------------
# Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------
# Train classifier
# -------------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# Example prediction
# -------------------------------
query = "where can I apply for leave"
query_emb = embedder.encode([query])
pred_intent = clf.predict(query_emb)[0]
print("Predicted intent:", pred_intent)
