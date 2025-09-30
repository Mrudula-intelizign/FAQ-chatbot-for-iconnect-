import json
import random
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
import torch

# -------------------------------
# Load dataset
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "faq_data.json"), "r", encoding="utf-8") as f:
    faq_data = json.load(f)["faqs"]

examples = []
labels = []
responses = {}
metadata_map = {}

# Build dataset with metadata
for item in faq_data:
    intent = item["intent"]
    meta = item.get("metadata", {})

    # Create a metadata string (category, subcategory, tags, etc.)
    meta_str_parts = []
    if "category" in meta:
        meta_str_parts.append(f"category: {meta['category']}")
    if "subcategory" in meta:
        meta_str_parts.append(f"subcategory: {meta['subcategory']}")
    if "tags" in meta:
        meta_str_parts.append("tags: " + ", ".join(meta["tags"]))
    meta_str = " | ".join(meta_str_parts)

    for ex in item["examples"]:
        # Append metadata to example text
        enriched_example = f"{ex} | {meta_str}" if meta_str else ex
        examples.append(enriched_example)
        labels.append(intent)

    responses[intent] = item["responses"]
    metadata_map[intent] = meta  # store metadata for later if needed

print(f"Total examples: {len(examples)} | Total intents: {len(set(labels))}")

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    examples, labels, test_size=0.2, random_state=42, stratify=labels
)

# -------------------------------
# Sentence-BERT embeddings
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

train_embeddings = model.encode(X_train, convert_to_tensor=True)
test_embeddings = model.encode(X_test, convert_to_tensor=True)

# -------------------------------
# Prediction with threshold
# -------------------------------
def predict(query_embedding, k=1, threshold=0.6):
    cos_scores = util.pytorch_cos_sim(query_embedding, train_embeddings)[0]
    top_results = torch.topk(cos_scores, k=k)

    best_score = top_results.values[0].item()
    best_idx = top_results.indices[0].item()

    if best_score < threshold:
        return None, best_score
    return y_train[best_idx], best_score

# -------------------------------
# Evaluate accuracy
# -------------------------------
correct = 0
for i, test_q in enumerate(X_test):
    query_emb = test_embeddings[i]
    pred_intent, score = predict(query_emb)
    if pred_intent == y_test[i]:
        correct += 1

accuracy = correct / len(X_test) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# # -------------------------------
# # Interactive query
# # -------------------------------
while True:
    user_input = input("\nEnter your question (or 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        break

    # Enrich user input with metadata lookup (optional, if you have hints)
    query_emb = model.encode(user_input, convert_to_tensor=True)
    pred_intent, score = predict(query_emb, threshold=0.6)

    if pred_intent is None:
        print("🤔 Sorry, I’m not sure about that. Could you rephrase or try another question?")
    else:
        bot_response = random.choice(responses[pred_intent])
        meta = metadata_map.get(pred_intent, {})
        print(f"\nPredicted Intent: {pred_intent} (Confidence: {score:.2f})")
        print(f"Bot Response: {bot_response}")
       



