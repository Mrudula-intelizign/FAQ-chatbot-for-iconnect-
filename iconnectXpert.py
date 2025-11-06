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
intent_metadata = {}

for item in faq_data:
    intent_examples[item["intent"]] = item["examples"]
    intent_responses[item["intent"]] = item["responses"]
    intent_metadata[item["intent"]] = item.get("metadata", {})
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
# def semantic_match(user_input, top_k=3, threshold=0.7):
#     user_emb = embedder.encode(user_input, convert_to_tensor=True)
#     cos_scores = util.pytorch_cos_sim(user_emb, faq_embeddings)[0]

#     top_results = torch.topk(cos_scores, k=top_k)
#     matches = []
#     for score, idx in zip(top_results.values, top_results.indices):
#         if score >= threshold:
#             matches.append((faq_intents[idx], float(score)))

#     return matches if matches else None

def semantic_match(user_input, top_k=3, threshold=0.7):
    user_emb = embedder.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_emb, faq_embeddings)[0]

    top_results = torch.topk(cos_scores, k=top_k)
    matches = []
    for score, idx in zip(top_results.values, top_results.indices):
        if score >= threshold:
            intent = faq_intents[idx]
            meta = intent_metadata.get(intent, {})
            boost = 0.0

            # Boost if user_input contains tag words
            for tag in meta.get("tags", []):
                if tag.lower() in user_input.lower():
                    boost += 0.05

            # Boost high priority intents
            if meta.get("priority") == "high":
                boost += 0.05

            matches.append((intent, float(score) + boost))

    return sorted(matches, key=lambda x: x[1], reverse=True) if matches else None


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
            if matches[0][1] < 0.7:
                return None   # Unknown
            # Deduplicate suggestion labels
            options = []
            seen = set()
            for m in matches[:2]:
                intent_key = m[0]
                label = intent_examples[intent_key][0]
                if label not in seen:
                    options.append(label)
                    seen.add(label)
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

# def get_response(user_input, threshold=0.5):
#     intent = match_intent(user_input)

#     # Case 1: Ambiguous intents (return suggestions)
#     if isinstance(intent, dict) and "ambiguous" in intent:
#         options = intent["ambiguous"]
#         return {
#             "text": "🤔 Did you mean one of these?",
#             "suggestions": options
#         }

#     # Case 2: Clear intent found
#     if isinstance(intent, str):
#         if intent_responses[intent]:
#             return {"text": random.choice(intent_responses[intent])}
#         return {"text": "⚠️ Sorry, no response available for this intent."}

#     # Case 3: Unknown question → fallback message
#     return {"text": "🤔 Sorry, I’m not sure about that. Could you rephrase your question or contact support?"}

def get_response(user_input, threshold=0.5):
    intent = match_intent(user_input)

    # Case 1: Ambiguous intents
    if isinstance(intent, dict) and "ambiguous" in intent:
        options = intent["ambiguous"]
        return {"text": "🤔 Did you mean one of these?", "suggestions": options}

    # Case 2: Clear intent found
    if isinstance(intent, str):
        if intent_responses[intent]:
            return {"text": random.choice(intent_responses[intent])}
        return {"text": "⚠️ Sorry, no response available for this intent."}

    # Case 3: Unknown question
    return {"text": "🤔 Sorry, I’m not sure about that. Could you rephrase your question or contact support?"}

# -------------------------------
# Streamlit App Config
# -------------------------------
# st.set_page_config(page_title="iConneXpert", page_icon="🤖")
# st.title("💬 iConneXpert Chatbot")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_input = st.chat_input("Type your message...")

# if user_input:
#     st.session_state.chat_history.append(("user", user_input))
#     response = get_response(user_input)
#     st.session_state.chat_history.append(("bot", response))

# # Display chat in order
# for i, (sender, msg) in enumerate(st.session_state.chat_history):
#     if sender == "user":
#         with st.chat_message("user"):
#             st.markdown(msg)
#     else:
#         with st.chat_message("assistant"):
#             if isinstance(msg, dict):
#                 st.markdown(msg["text"])
#                 if "suggestions" in msg:
#                     for j, option in enumerate(msg["suggestions"]):   # enumerate here
#                         if st.button(option, key=f"suggestion_{i}_{j}"):  # unique key
#                             # Append user choice
#                             st.session_state.chat_history.append(("user", option))
#                             # Get bot response
#                             response = get_response(option)
#                             st.session_state.chat_history.append(("bot", response))
#                             st.experimental_rerun()
#             else:
#                 st.markdown(msg)
        

# -------------------------------
# Streamlit Setup
# -------------------------------
import streamlit as st

st.set_page_config(page_title="iConnectXpert", page_icon="🤖", layout="wide")

# -------------------------------
# Load Custom CSS
# -------------------------------
def load_css(file_name="style.css"):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")
# -------------------------------
# Header Section (Custom)
# -------------------------------

import base64

def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_base64 = get_base64_of_image("iconnect.png")

st.markdown(f"""
<div class="title-container">
    <img src="data:image/png;base64,{image_base64}" class="title-logo">
    <h1 class="h1">iConnectXpert</h1>
</div>
""", unsafe_allow_html=True)

# st.markdown("""
# <div class="title-container">
#     <img src="iconnect.png" class="title-logo">
#     <h1 class="h1">iConnectXpert</h1>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("<p class='subtitle'>Empowering Your Intelizign Journey</p>", unsafe_allow_html=True)


# -------------------------------
# Session State for Chat
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Chat Input (move up to run before welcome)
# -------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    response = get_response(user_input)
    st.session_state.chat_history.append(("bot", response))

# -------------------------------
# Welcome Message (only when no chat yet)
# -------------------------------
if len(st.session_state.chat_history) == 0 and not user_input:
    st.markdown(
        """
        <div class="welcome-container">
            <h2>👋 Hi there! I am iConnect Virtual Assistant</h2>
            <p>How can I assist you today?</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Display Chat Messages
# -------------------------------
elif len(st.session_state.chat_history) > 0:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for i, (sender, msg) in enumerate(st.session_state.chat_history):
        if sender == "user":
            st.markdown(f'<div class="user-msg">{msg}</div>', unsafe_allow_html=True)
        else:
            if isinstance(msg, dict):
                st.markdown(f'<div class="bot-msg">{msg["text"]}</div>', unsafe_allow_html=True)
                if "suggestions" in msg:
                    st.markdown('<div class="suggestion-container">', unsafe_allow_html=True)
                    st.markdown('<div class="suggestion-title">🤔 Did you mean one of these?</div>', unsafe_allow_html=True)

                    for j, option in enumerate(msg["suggestions"]):
                        if st.button(option, key=f"suggestion_{i}_{j}"):
                            st.session_state.chat_history.append(("user", option))
                            response = get_response(option)
                            st.session_state.chat_history.append(("bot", response))
                            st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">{msg}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
