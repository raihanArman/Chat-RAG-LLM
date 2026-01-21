# Chat App LLM + RAG

[![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://streamlit.io/) 
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)](https://ollama.ai/)

A **Streamlit-based chat application** that integrates **retrieval-augmented generation (RAG)** with a local LLM (Ollama) and **Chroma vector database**.

---

## Features

- Chat with a local LLM (Ollama)
- Upload documents (PDF, DOCX, TXT) for context
- Add URLs as sources
- RAG: Retrieve relevant context from uploaded sources
- Streaming responses in the chat interface
- Toggle RAG on/off
- Clear chat session

---

## Requirements

- Python 3.12+
- Virtual environment (venv)
- Streamlit
- LangChain and LangChain Community packages
- HuggingFace sentence-transformers
- Chroma DB
- Ollama installed locally
- python-dotenv

---

## Setup

1. Clone the repo:

```bash
git clone https://github.com/username/chat-app-llm-rag.git
cd chat-app-llm-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```