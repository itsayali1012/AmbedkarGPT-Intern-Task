# AmbedkarGPT Intern Task

This project is a command line Retrieval Augmented Generation system built for the Kalpit Pvt Ltd AI Intern assignment. The system uses the provided speech text by Dr. B. R. Ambedkar as the only knowledge source. It converts the text into embeddings, stores them in a local vector database, retrieves relevant context for a user query, and generates an answer through a locally running Mistral model using Ollama.

## Features
- Compatible with Python 3.8 and above
- Retrieval Augmented Generation pipeline built using LangChain
- Local vector database created with ChromaDB
- Embeddings generated using MiniLM L6 v2 from HuggingFace
- Local inference with Ollama using a Mistral model
- Fully offline after installation

## Project Structure
project-folder/
    main.py
    speech.txt
    requirements.txt
    vectorstore/        (created automatically)

## Setup

### Install Ollama
Download the Windows installer from:
https://ollama.com/download
Run it as administrator and restart your computer after installation.

### Pull a Mistral model
Use a model supported by your system memory. The default option is:
ollama pull mistral

### Create a virtual environment
python -m venv venv
venv\Scripts\activate

### Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

### Build the vector store
python main.py --rebuild

### Run the application
python main.py

Ask questions based only on the content of speech.txt. Type exit to stop the program.

- The vectorstore directory is created automatically on first run
- If your system has low memory, the model name in main.py can be changed to a lighter Mistral variant supported by your Ollama installation
