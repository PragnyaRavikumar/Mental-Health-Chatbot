# Mental-Health-Chatbot
An AI-powered chatbot designed to promote emotional well-being by detecting user emotions and providing supportive, calming responses. This chatbot combines a BERT-based multi-label emotion classifier (trained on the GoEmotions dataset) with the Mistral 7B language model to generate context-aware replies.
## Features
- Emotion detection using fine-tuned BERT
- Response generation with Mistral 7B
- Clean and interactive Streamlit interface
- Trained on Google’s GoEmotions dataset
- Focused on empathy and mental health support
## 📁 Dataset
- [GoEmotions by Google](https://github.com/google-research/goemotions)  
A dataset of 200k Reddit comments labeled with 28 emotion categories.
## 📥 Model Download

To download the **Mistral-7B-Instruct-v0.1.Q4_K_M.gguf** model used in this project, run:

```bash
curl -LO https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```
This model can be run locally using [llama.cpp](https://github.com/ggerganov/llama.cpp) or other GGUF-compatible backends.


