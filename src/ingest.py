import os
from dotenv import load_dotenv
import sys
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from database import DatabaseConnection

db = DatabaseConnection()
conn_str = db.connection_string()

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

def get_embeddings():
    provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY não configurada.")
        model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model, api_key=api_key)

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY não configurada.")
        model = os.getenv("GOOGLE_EMBEDDINGS_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

    raise ValueError("EMBEDDINGS_PROVIDER inválido. Use: openai ou google.")

def load_and_split(pdf_path: str) -> List:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def ingest_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"Arquivo não encontrado: {PDF_PATH}")
        sys.exit(1)

    embeddings = get_embeddings()
    collection = os.getenv("PGVECTOR_COLLECTION", "pdf_chunks")

    chunks = load_and_split(PDF_PATH)
    if not chunks:
        print("Nenhum conteúdo encontrado no PDF.")
        sys.exit(1)

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection,
        connection=conn_str,
        use_jsonb=True,
    )

    store.add_documents(chunks)

    print("Ingestão concluída!")
    print(f"- PDF: {PDF_PATH}")
    print(f"- Chunks inseridos: {len(chunks)}")
    print(f"- Coleção: {collection}")


if __name__ == "__main__":
    ingest_pdf()