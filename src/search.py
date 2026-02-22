import os
from typing import List, Tuple

from langchain_postgres import PGVector

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

from database import DatabaseConnection

db = DatabaseConnection()
conn_str = db.connection_string()


OUT_OF_CONTEXT_MSG = "Não tenho informações necessárias para responder sua pergunta."

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

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

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY não configurada.")
        model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0)

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY não configurada.")
        model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)

    raise ValueError("LLM_PROVIDER inválido. Use: openai ou google.")


def build_context(results: List[Tuple]) -> str:
    parts = []
    for doc, score in results:
        text = (doc.page_content or "").strip()
        if text:
            parts.append(text)
    return "\n\n---\n\n".join(parts)

def search_prompt(question=None):
    embeddings = get_embeddings()
    llm = get_llm()

    collection = os.getenv("PGVECTOR_COLLECTION", "pdf_chunks")

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection,
        connection=conn_str,
        use_jsonb=True,
    )

    print("Chat CLI (digite 'sair' para encerrar)")
    while True:
        pergunta = input("\nPERGUNTA: ").strip()
        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit"):
            break

        results = store.similarity_search_with_score(pergunta, k=10)

        if not results:
            print(f"RESPOSTA: {OUT_OF_CONTEXT_MSG}")
            continue

        contexto = build_context(results).strip()
        if not contexto:
            print(f"RESPOSTA: {OUT_OF_CONTEXT_MSG}")
            continue

        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)

        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
        text = text.strip()
        
        if not text:
            text = OUT_OF_CONTEXT_MSG

        print(f"RESPOSTA: {text}")