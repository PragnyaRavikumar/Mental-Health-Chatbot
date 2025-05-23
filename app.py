import streamlit as st
from transformers import AutoTokenizer, BertForSequenceClassification
import torch
import json
from llama_cpp import Llama

# -----------------------------
# Load BERT Emotion Classifier
# -----------------------------
@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("model/bert_emotion_model")  # Your saved model folder
    model = BertForSequenceClassification.from_pretrained("model/bert_emotion_model")
    return tokenizer, model

# -----------------------------
# Load Mistral Model (GGUF)
# -----------------------------
@st.cache_resource
def load_mistral_model():
    return Llama(
        model_path="mistral-7b.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=2,  # Safer for 4-core CPUs
        n_batch=256,
        verbose=False
    )

# -----------------------------
# Predict Emotion from Text
# -----------------------------
def predict_emotion(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    predicted_label = torch.argmax(probs, dim=1).item()
    
    with open("model/bert_emotion_model/config.json", "r") as f:
        config = json.load(f)
    id2label = config.get("id2label", {})
    return id2label[str(predicted_label)]

# -----------------------------
# Generate response with Mistral
# -----------------------------
def generate_response(user_input, emotion, llm):
    prompt = (
        f"You are a calm and friendly therapist chatbot.\n"
        f"The user seems to be feeling {emotion}.\n"
        f"User: {user_input}\n"
        f"Therapist:"
    )
    try:
        response = llm(prompt, max_tokens=128, stop=["User:", "Therapist:"], echo=False)
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"(Error generating response: {e})"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Therapist Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 Halo - Mental Health Chatbot")
st.markdown("A chatbot that understands your emotions and responds supportively.")

# Load models
bert_tokenizer, bert_model = load_bert_model()
llm = load_mistral_model()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    role = msg["role"]
    if role == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Tell me how you're feeling..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.spinner("Thinking..."):
        emotion = predict_emotion(user_input, bert_tokenizer, bert_model)
        bot_reply = generate_response(user_input, emotion, llm)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.chat_message("assistant").markdown(bot_reply)
