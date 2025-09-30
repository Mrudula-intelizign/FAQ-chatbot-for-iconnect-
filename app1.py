import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"   # <-- blocks TF imports

import json
import random
import streamlit as st
import torch
from difflib import get_close_matches
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Load FAQ Dataset
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "faq_data.json"), "r", encoding="utf-8") as f:
    faq_data = json.load(f)["faqs"]

# Build intent dictionary
intent_examples = {}
intent_responses = {}

for item in faq_data:
    intent_examples[item["intent"]] = item["examples"]
    intent_responses[item["intent"]] = item["responses"]
# -------------------------------
# Load HuggingFace Transformer (DistilBERT Squad2)
# -------------------------------
model_name = "distilbert/distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False
)
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=-1   # CPU
)

# -------------------------------
# Embeddings Model (Sentence-BERT)
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

faq_examples = []
faq_intents = []
for intent, examples in intent_examples.items():
    for ex in examples:
        faq_examples.append(ex)
        faq_intents.append(intent)

faq_embeddings = embedder.encode(faq_examples, convert_to_tensor=True)

# -------------------------------
# Semantic Matcher (Top-N)
# -------------------------------
def semantic_match(user_input, top_k=3, threshold=0.7):
    user_emb = embedder.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_emb, faq_embeddings)[0]

    top_results = torch.topk(cos_scores, k=top_k)
    matches = []
    for score, idx in zip(top_results.values, top_results.indices):
        if score >= threshold:
            matches.append((faq_intents[idx], float(score)))

    return matches if matches else None

def match_intent(user_input):
    # 1. Rule-based fuzzy match
    for intent, examples in intent_examples.items():
        matches = get_close_matches(user_input.lower(), [ex.lower() for ex in examples], n=1, cutoff=0.9)
        if matches:
            return intent

    # 2. Semantic Top-N
    matches = semantic_match(user_input, top_k=3, threshold=0.7)
    if matches:
        # Check if top 2 intents are close in score
        if len(matches) > 1 and abs(matches[0][1] - matches[1][1]) < 0.05:
            # Only show suggestions if scores are high enough
            if matches[0][1] < 0.7:
                return None   # Unknown
            # Show friendly labels instead of intent keys
            options = []
            for m in matches[:2]:
                intent_key = m[0]
                options.append(intent_examples[intent_key][0])  # first example as label
            return {"ambiguous": options}
        return matches[0][0]

    return None  # No matching intent



# -------------------------------
# Context Builder
# -------------------------------
context = " ".join([" ".join(item["responses"]) for item in faq_data])

def get_context(user_input):
    intent = match_intent(user_input)
    if isinstance(intent, str):
        return " ".join(intent_responses[intent])
    return context

def get_response(user_input, threshold=0.5):
    intent = match_intent(user_input)

    # Case 1: Ambiguous intents (return suggestions)
    if isinstance(intent, dict) and "ambiguous" in intent:
        options = intent["ambiguous"]
        return {
            "text": "🤔 Did you mean one of these?",
            "suggestions": options
        }

    # Case 2: Clear intent found
    if isinstance(intent, str):
        if intent_responses[intent]:
            return {"text": random.choice(intent_responses[intent])}
        return {"text": "⚠️ Sorry, no response available for this intent."}

    # Case 3: Unknown question → fallback message
    return {"text": "🤔 Sorry, I’m not sure about that. Could you rephrase your question or contact support?"}


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="iConneXpert", page_icon="🤖")
st.title("💬 iConneXpert Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    response = get_response(user_input)
    st.session_state.chat_history.append(("bot", response))

# Display chat in order
for i, (sender, msg) in enumerate(st.session_state.chat_history):
    if sender == "user":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            if isinstance(msg, dict):
                st.markdown(msg["text"])
                if "suggestions" in msg:
                    for j, option in enumerate(msg["suggestions"]):   # enumerate here
                        if st.button(option, key=f"suggestion_{i}_{j}"):  # unique key
                            # Append user choice
                            st.session_state.chat_history.append(("user", option))
                            # Get bot response
                            response = get_response(option)
                            st.session_state.chat_history.append(("bot", response))
                            st.experimental_rerun()
            else:
                st.markdown(msg)
        



