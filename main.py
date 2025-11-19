from pathlib import Path
import argparse
from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

ROOT = Path(__file__).parent.resolve()
DATA_FILE = ROOT / "speech.txt"
DB_PATH = ROOT / "vectorstore"

def load_documents():
    return TextLoader(str(DATA_FILE), encoding="utf-8").load()

def split_documents(docs):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=120
    )
    return splitter.split_documents(docs)

def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_vectorstore():
    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = create_embeddings()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_PATH)
    )
    store.persist()
    return store

def load_vectorstore():
    embeddings = create_embeddings()
    return Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings
    )

def model():
    return Ollama(model="mistral", temperature=0)

def qa_chain(store):
    retriever = store.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=model(),
        chain_type="stuff",
        retriever=retriever
    )

def interactive_session(chain):
    print("AmbedkarGPT ready. Type 'exit' to stop.\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() == "exit":
            break
        response = chain.run(query)
        print("\nAnswer:", response, "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.rebuild or not DB_PATH.exists():
        store = build_vectorstore()
    else:
        store = load_vectorstore()

    chain = qa_chain(store)
    interactive_session(chain)

if __name__ == "__main__":
    main()
