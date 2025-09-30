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

# Initialize empty lists and a dictionary
examples = []
labels = []
responses = {}
# Loop through each FAQ item in the dataset
for item in faq_data:
    intent = item["intent"]
    # For every example question under this intent
    for ex in item["examples"]:
        examples.append(ex) # store the question text
        labels.append(intent) # store the corresponding intent
    # Store responses for this intent (so the bot can reply later)
    responses[intent] = item["responses"]
# how many examples in total and how many unique intents
print(f"Total examples: {len(examples)} | Total intents: {len(set(labels))}")

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    examples, labels, test_size=0.2, random_state=42, stratify=labels # keep the same proportion of intents in both train and test
)

# -------------------------------
# Sentence-BERT embeddings
# -------------------------------
# Load a pre-trained sentence embedding model from SentenceTransformers
# "all-MiniLM-L6-v2" is a lightweight but powerful model for semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode the training and test examples into dense vector embeddings
train_embeddings = model.encode(X_train, convert_to_tensor=True)
test_embeddings = model.encode(X_test, convert_to_tensor=True)

# -------------------------------
# Prediction via cosine similarity
# -------------------------------
def predict(query_embedding, k=1):
    # 1. Compute cosine similarity between the query and all training embeddings
    cos_scores = util.pytorch_cos_sim(query_embedding, train_embeddings)[0]
    # 2. Get the top-k most similar training examples
    top_results = torch.topk(cos_scores, k=k)
    # 3. Take the best match (index of the most similar training example)
    best_idx = top_results.indices[0].item()
    return y_train[best_idx]

# -------------------------------
# Evaluate accuracy on test set
# -------------------------------
correct = 0 # counter for correct predictions
# Loop through each test example
for i, test_q in enumerate(X_test):
    query_emb = test_embeddings[i] # get embedding of the test question
    pred_intent = predict(query_emb) # predict the intent using nearest neighbor
    if pred_intent == y_test[i]: # compare with the true label
        correct += 1 # increment counter if prediction is correct


accuracy = correct / len(X_test) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# -------------------------------
# Interactive custom query
# -------------------------------
while True:
    user_input = input("\nEnter your question (or 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        break

    query_emb = model.encode(user_input, convert_to_tensor=True)
    pred_intent = predict(query_emb)
    bot_response = random.choice(responses[pred_intent])

    print(f"\nPredicted Intent: {pred_intent}")
    print(f"Bot Response: {bot_response}")
