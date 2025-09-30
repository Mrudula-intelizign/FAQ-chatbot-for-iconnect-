import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Load dataset
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "faq_data.json"), "r", encoding="utf-8") as f:
    faq_data = json.load(f)["faqs"]

texts = []
labels = []
intent_to_response = {}

for item in faq_data:
    intent = item["intent"]
    examples = item["examples"]
    responses = item["responses"]

    for ex in examples:
        texts.append(ex)
        labels.append(intent)

    # Map intent → first response (you can later choose random/weighted)
    intent_to_response[intent] = responses[0] if responses else "No response available."

print(f"Total samples: {len(texts)}")
print(f"Unique intents: {len(set(labels))}")

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# -------------------------
# Build pipeline (TF-IDF + SVM)
# -------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
    ("clf", LinearSVC())
])

# Train
model.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Accuracy:", round(accuracy * 100, 2), "%")
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# Try custom queries
# -------------------------
while True:
    user_q = input("\nAsk a question (or type 'exit'): ")
    if user_q.lower() == "exit":
        break
    
    pred_intent = model.predict([user_q])[0]
    response = intent_to_response.get(pred_intent, "Sorry, I don't know that.")
    
    print(f"\n🔍 Predicted intent: {pred_intent}")
    print(f"🤖 Bot response: {response}")
