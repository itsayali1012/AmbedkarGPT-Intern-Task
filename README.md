# AmbedkarGPT – Graph-based Retrieval Augmented Generation System

AmbedkarGPT is a fully local Retrieval-Augmented Generation (RAG) system inspired
by the SemRAG / GraphRAG paradigm. It uses Dr. B. R. Ambedkar’s writings as a
knowledge source and combines semantic chunking, a knowledge graph, and local
LLM inference to answer user questions.

The goal of this project is not only to retrieve relevant text, but to
incorporate structural knowledge through entity relationships, making retrieval
more context-aware than traditional vector-only RAG systems.

---

## Why Graph-based RAG?

Standard RAG systems rely purely on vector similarity, which often misses
important contextual relationships.

This project improves retrieval by:
- Extracting named entities from text
- Building a graph of entity co-occurrences
- Using graph structure to guide retrieval
- Reducing hallucinations by grounding answers in connected concepts

---

## System Architecture

1. **Document Ingestion**
   - The Ambedkar PDF is loaded page by page.

2. **Semantic Chunking**
   - Text is split at sentence level.
   - Adjacent sentences are merged based on embedding similarity.

3. **Knowledge Graph Construction**
   - Named entities are extracted using spaCy.
   - Entities are connected if they co-occur in the same chunk.
   - Louvain community detection identifies conceptual clusters.

4. **Embedding and Indexing**
   - SentenceTransformer embeddings are generated.
   - FAISS is used for fast similarity search.
   - Embeddings and graph are cached locally.

5. **Graph-guided Retrieval**
   - Query is matched against entity embeddings.
   - Relevant chunks are retrieved through entity–chunk mappings.

6. **Answer Generation**
   - Retrieved context is passed to a local LLM via Ollama.
   - Answers are generated strictly using provided context.

---
ambedkartask/
├── data/
│   └── Ambedkar_book.pdf
├── cache/
│   ├── chunks.json
│   ├── graph.pkl
│   └── chunk_embeddings.npy
├── main.py
├── requirements.txt
└── README.md

---

## Setup Instructions

### 1. Install dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm

### 2. Install Ollama

Download Ollama from:
https://ollama.com

Then pull the model:

ollama pull llama3
Running the Application
python main.py


Optional parameters:

python main.py --local_top_k 7


A Gradio interface will open in your browser.

Notes

The system runs entirely locally.

No external APIs or cloud services are used.

Intermediate results are cached for efficiency.

The architecture is modular and easily extensible.

Future Improvements

Add global community-level retrieval

Introduce reranking strategies

Support multiple documents

Add evaluation metrics


---

### Why this will now render correctly on GitHub

- Triple backticks (` ``` `) are used for **code blocks**
- `bash` is specified for syntax highlighting
- Blank lines are placed where Markdown requires them
- Folder structure is shown as monospaced text
- Headings and bullet points are correctly separated



## Project Structure

