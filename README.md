# Desafio MBA Engenharia de Software com IA - Full Cycle

## Requisitos
- Docker + Docker Compose
- Python 3.10+

## Subir banco (PostgreSQL + pgVector)
1. Copie `.env.example` para `.env` e ajuste as variáveis.
2. Suba o banco:
   docker compose up -d

## Instalar dependências
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate   # windows
pip install -r requirements.txt

## Ingestão do PDF
python ./src/ingest.py

## Chat no terminal
python ./src/chat.py

Digite "sair" para encerrar.