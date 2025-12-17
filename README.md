# AmbedkarGPT – Graph-based RAG System

This project implements a local Retrieval-Augmented Generation (RAG) system
inspired by the SemRAG / GraphRAG approach, using Dr. B. R. Ambedkar’s writings
as the knowledge source.

The system combines semantic chunking, a knowledge graph built from named
entities, and local LLM inference using Ollama.

---

## Features

- Semantic sentence-level chunking
- Knowledge graph construction using named entities
- Louvain community detection
- Graph-based local retrieval
- Local LLM inference using Ollama (no cloud dependency)
- Interactive Gradio chat interface

---

## Project Structure
ambedkartask/
├── data/
│ └── Ambedkar_book.pdf
├── main.py
├── requirements.txt
└── README.md


---

## Setup Instructions

### 1. Install Python dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm


### 2. Install and start Ollama

Download Ollama from:
https://ollama.com

Then run:



ollama pull llama3


---

## Running the Application

From the project root directory:



python main.py


Optional parameters:

python main.py --local_top_k 7 --global_top_k 4

After running, a Gradio interface will open in your browser.

---

## Notes

- The system runs fully locally.
- No external APIs or cloud services are required.
- PDF path and retrieval parameters are configurable
