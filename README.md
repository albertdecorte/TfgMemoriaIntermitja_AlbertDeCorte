# TfgMemoriaIntermitja\_AlbertDeCorte

It contains the code used in the development of the final degree project (TFG)



To execute it, you need to:



\# EU AI Act Compliance Chatbot



An ethical AI assistant that analyzes text and verifies compliance with the EU AI Act and other regulatory frameworks. Features real-time guardrail checking, document Q\&A, and RAG-based EU regulation reference.



\## Features



\- 🔍 \*\*Real-time Compliance Checking\*\*: Semantic similarity matching against EU AI Act prohibitions

\- 📚 \*\*EU AI Act RAG\*\*: Vector database with full EU AI Act text for contextual responses

\- 📄 \*\*Document Q\&A\*\*: Upload and query PDF/text documents

\- 🛡️ \*\*Multi-category Guardrails\*\*: Drugs, weapons, self-harm, discrimination, medical, prohibited AI practices

\- 📊 \*\*Compliance Scoring\*\*: Quantitative compliance assessment

\- 📝 \*\*Audit Logging\*\*: Complete conversation and compliance history

\- 🌐 \*\*Chainlit UI\*\*: Modern web interface with file upload support



\## Prerequisites



\- Python 3.9 or higher

\- NexaAi (look down)

\- 16GB+ RAM recommended (important)



\## Installation Guide



\### 1. Clone the Repository



\### 2. Create Virtual Environment

Windows:



bash

python -m venv venv

venv\\Scripts\\activate



macOS/Linux:



bash

python3 -m venv venv

source venv/bin/actívate



pip install -r requirements.txt



\# Core dependencies

pip install chainlit

pip install openai

pip install chromadb

pip install PyPDF2

pip install sentence-transformers



\# Data processing

pip install numpy

pip install pandas



\# Utilities

pip install python-dotenv

pip install sqlite3



\# Optional: For better PDF processing

pip install pypdf



\# Install Nexa AI via pip

pip install nexaai



\# Or install from source for latest features

git clone https://github.com/NexaAI/nexa-sdk.git

cd nexa-sdk

pip install -e .



Install QWEN3

nexa infer NexaAI/Qwen3-VL-4B-Instruct-GGUF



Start the server:

nexa serve

! In case it doesn't match the path: http://127.0.0.1:18181 (change it on the code app.py with VSCode or another Python IDE)



cd (the path to the folder where you downloaded the GitHub code repo)\\TfgMemoriaIntermitja\_AlbertDeCorte

chainlit run app.py



wait for the code to execute







