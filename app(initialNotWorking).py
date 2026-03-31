import chainlit as cl
from openai import AsyncOpenAI
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import sqlite3 
from datetime import datetime

client = AsyncOpenAI(api_key="nexa", base_url="http://127.0.0.1:18181/v1")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

settings = {
    "model": "NexaAI/Qwen3-VL-4B-Instruct-GGUF",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

@cl.on_chat_start
async def start_chat():

    # Initialize database
    init_db()

    # Initialize chat memory
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}]
    )

    # No collection initially (normal chat mode)
    cl.user_session.set("collection", None)

    # Ask user if they want to upload a document
    files = await cl.AskFileMessage(
        content="Upload a document for Q&A or press skip to chat normally.",
        accept=["text/plain", "application/pdf", "text/markdown"],
        max_size_mb=20,
        timeout=20,
    ).send()

    # If user skipped upload → normal chat mode
    if not files:
        await cl.Message(content="Chat mode activated. Ask anything!").send()
        return

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # -------------------------
    # Extract text
    # -------------------------
    if file.name.endswith(".pdf"):

        reader = PdfReader(file.path)

        text = ""
        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    else:

        with open(file.path, "r", encoding="utf-8") as f:
            text = f.read()

    # -------------------------
    # Chunk document
    # -------------------------
    chunks = chunk_text(text)

    # -------------------------
    # Reset Chroma collection
    # -------------------------
    try:
        chroma_client.delete_collection(name="documents")
    except Exception:
        pass

    collection = chroma_client.create_collection(name="documents")

    # -------------------------
    # Embed chunks
    # -------------------------
    for i, chunk in enumerate(chunks):

        embedding = get_embedding(chunk)

        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            ids=[f"chunk_{i}"]
        )

    # Store collection in session
    cl.user_session.set("collection", collection)

    msg.content = (
        f"Processing `{file.name}` complete.\n"
        f"Indexed **{len(chunks)} chunks**.\n"
        f"You can now ask questions about the document."
    )

    await msg.update()

@cl.on_message
async def main(message: cl.Message):

    collection = cl.user_session.get("collection")
    message_history = cl.user_session.get("message_history")

    # -------------------------
    # 1. GUARDRAIL CHECK
    # -------------------------
    allowed, reason = check_guardrails(message.content)

    if not allowed:
        await cl.Message(content=reason).send()
        return

    # -------------------------
    # 2. SAVE USER MESSAGE
    # -------------------------
    save_message("user", message.content)

    # -------------------------
    # 3. HANDLE CHAT OR DOC MODE
    # -------------------------
    if collection is not None:

        query_embedding = get_embedding(message.content)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        context = "\n\n".join(results["documents"][0])

        enhanced_message = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{message.content}
"""

    else:
        # No document loaded → normal chat
        enhanced_message = message.content

    temp_history = message_history.copy()
    temp_history.append({"role": "user", "content": enhanced_message})

    msg = cl.Message(content="")
    await msg.send()

    # -------------------------
    # 4. STREAM MODEL RESPONSE
    # -------------------------
    stream = await client.chat.completions.create(
        messages=temp_history,
        stream=True,
        **settings
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    # -------------------------
    # 5. UPDATE MEMORY
    # -------------------------
    message_history.append({"role": "user", "content": message.content})
    message_history.append({"role": "assistant", "content": msg.content})

    cl.user_session.set("message_history", message_history)

    # -------------------------
    # 6. SAVE ASSISTANT MESSAGE
    # -------------------------
    save_message("assistant", msg.content)

    await msg.update()

DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_message(role, content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO conversations (role, content, timestamp) VALUES (?, ?, ?)",
        (role, content, datetime.utcnow().isoformat())
    )

    conn.commit()
    conn.close()

# ===================================
# GuardrailManager: handles legal/rule documents
# ===================================
class GuardrailManager:
    def __init__(self, folder="guardrails", chunk_size=200, threshold=0.7):
        self.folder = folder
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.documents = []
        self.embeddings = []
        self.ids = []
        self.load_guardrails()

    def load_guardrails(self):
        for filename in os.listdir(self.folder):
            path = os.path.join(self.folder, filename)
            if filename.endswith(".pdf"):
                reader = PdfReader(path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            else:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

            # Chunk text
            chunks = chunk_text(text, self.chunk_size)
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.ids.append(f"{filename}_chunk_{i}")
                self.embeddings.append(get_embedding(chunk))

    def check_violation(self, text):
        query_emb = np.array(get_embedding(text))
        best_score = 0.0
        best_match = None

        for emb, doc in zip(self.embeddings, self.documents):
            emb_arr = np.array(emb)
            score = np.dot(query_emb, emb_arr) / (np.linalg.norm(query_emb) * np.linalg.norm(emb_arr))
            if score > best_score:
                best_score = score
                best_match = doc

        if best_score >= self.threshold:
            return False, f"This request may violate a rule:\n\n{best_match}"

        return True, None

# ----------------------------------------
# Initialize guardrails
# ----------------------------------------
guardrail_manager = GuardrailManager(folder="guardrails", chunk_size=200, threshold=0.7)

def check_guardrails(text):
    """Wrapper for GuardrailManager"""
    return guardrail_manager.check_violation(text)