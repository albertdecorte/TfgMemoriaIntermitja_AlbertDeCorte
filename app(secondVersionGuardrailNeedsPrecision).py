import chainlit as cl
from openai import AsyncOpenAI
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import sqlite3 
from datetime import datetime
import os
import numpy as np

client = AsyncOpenAI(api_key="nexa", base_url="http://127.0.0.1:18181/v1")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

settings = {
    "model": "NexaAI/Qwen3-VL-4B-Instruct-GGUF",
    "temperature": 0.8,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

def get_embedding(text):
    return embedding_model.encode(text).tolist()

#This function works but can split paragraphs in a way that loses context. The new version groups multiple paragraphs together to preserve more context in each chunk, i decided to
#make the files uploaded by the user in chunks and use the paragraph versio nto keep guardrails consistency.
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def chunk_text_by_paragraphs(text, chunk_paragraphs=3):
    """Split text by paragraphs, grouping multiple paragraphs per chunk"""
    paragraphs = text.split('\n\n')
    chunks = []
    for i in range(0, len(paragraphs), chunk_paragraphs):
        chunk = '\n\n'.join(paragraphs[i:i + chunk_paragraphs])
        chunks.append(chunk)
    return chunks

@cl.on_chat_start
async def start_chat():

     # Initialize database
    init_db()

    guardrail_manager = GuardrailManager(folder="guardrails")
    cl.user_session.set("guardrails", guardrail_manager)

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
        await cl.Message(
            content="✅ **Chat mode activated!**\n\nYou can now ask me anything. If you want to upload a document for Q&A, just type 'upload' and I'll guide you."
        ).send()
        return

    # Si el usuario subió un archivo, procesarlo
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
        f"✅ Processing `{file.name}` complete.\n"
        f"Indexed **{len(chunks)} chunks**.\n\n"
        f"You can now ask questions about the document, or just chat normally!"
    )

    await msg.update()

@cl.on_message
async def main(message: cl.Message):

    collection = cl.user_session.get("collection")
    message_history = cl.user_session.get("message_history")

    # -------------------------
    # 1. GUARDRAIL CHECK
    # -------------------------
    guardrail_manager = cl.user_session.get("guardrails")
    allowed, reason = guardrail_manager.check_violation(message.content)

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
    def __init__(self, folder="guardrails", chunk_size=500, threshold=0.45):
        self.folder = folder
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.document_embeddings = []  # List of document embeddings
        self.document_names = []  # List of document names
        self.document_texts = []  # Full document texts for reference
        self.document_chunks = []  # Store chunks for better retrieval
        self.load_guardrails()

    def load_guardrails(self):
        print(f"📁 Loading legal documents from: {self.folder}")
        
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            print(f"Created guardrails folder: {self.folder}")
            return
        
        for filename in os.listdir(self.folder):
            print(f"📄 Loading: {filename}")
            path = os.path.join(self.folder, filename)
            
            # Skip non-document files
            if not (filename.endswith(".pdf") or filename.endswith(".txt") or filename.endswith(".md")):
                continue
                
            try:
                # Extract text from PDF or text file
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
                
                # Skip empty documents
                if not text.strip():
                    print(f"   ⚠️ Warning: {filename} is empty, skipping")
                    continue
                
                # Store document info
                self.document_names.append(filename)
                self.document_texts.append(text)
                
                # Generate embeddings for document chunks
                chunks = chunk_text(text, self.chunk_size)
                self.document_chunks.append(chunks)
                
                chunk_embeddings = []
                for chunk in chunks:
                    chunk_emb = np.array(get_embedding(chunk))
                    chunk_embeddings.append(chunk_emb)
                
                # Check if we have any valid chunks
                if chunk_embeddings:
                    # Average all chunk embeddings to get document-level embedding
                    doc_embedding = np.mean(chunk_embeddings, axis=0)
                    self.document_embeddings.append(doc_embedding)
                    print(f"   → Document has {len(chunks)} chunks, embedding size: {len(doc_embedding)}")
                else:
                    print(f"   ⚠️ Warning: No valid chunks for {filename}, skipping")
                    # Remove the document we just added since it has no embeddings
                    self.document_names.pop()
                    self.document_texts.pop()
                    
            except Exception as e:
                print(f"   ❌ Error loading {filename}: {e}")
                continue
        
        print(f"✅ Loaded {len(self.document_names)} legal documents")

    def check_violation(self, text):
        print(f"\n🔍 Checking if user query violates legal documents: {text[:100]}...")
        
        # Skip check if no guardrails loaded
        if not self.document_embeddings:
            print("   ℹ️ No guardrails loaded, skipping check")
            return True, None
        
        # Generate embedding for user query
        query_emb = np.array(get_embedding(text))
        
        # Calculate similarity with each legal document
        best_score = 0.0
        best_doc_index = -1
        best_doc_name = None
        
        for i, doc_emb in enumerate(self.document_embeddings):
            # Calculate cosine similarity
            query_norm = np.linalg.norm(query_emb)
            doc_norm = np.linalg.norm(doc_emb)
            
            if query_norm > 0 and doc_norm > 0:
                similarity = np.dot(query_emb, doc_emb) / (query_norm * doc_norm)
            else:
                similarity = 0.0
            
            print(f"   📊 Similarity with {self.document_names[i]}: {similarity:.3f}")
            
            if similarity > best_score:
                best_score = similarity
                best_doc_index = i
                best_doc_name = self.document_names[i]
        
        print(f"\n   🎯 Best match: {best_doc_name} (score: {best_score:.3f})")
        print(f"   ⚖️ Threshold: {self.threshold}")
        
        # Check if user query violates any legal document
        if best_score >= self.threshold:
            print(f"   ⚠️ VIOLATION DETECTED! User may be violating {best_doc_name}")
            
            # Find the most relevant section from the document
            relevant_section = self._find_relevant_section(text, best_doc_index)
            
            return False, f"""
⚠️ **LEGAL GUARDRAIL TRIGGERED**

Your query appears to potentially violate **{best_doc_name}** (similarity: {best_score:.2%}).

**Relevant section from the document:**
> {relevant_section}

**Please ensure your actions comply with applicable laws and regulations.**
"""
        
        print(f"   ✅ No violation detected")
        return True, None
    
    def _find_relevant_section(self, user_query, doc_index):
        """Find the most relevant section from the document based on user query"""
        if doc_index >= len(self.document_chunks):
            return "Unable to retrieve relevant section."
        
        chunks = self.document_chunks[doc_index]
        
        # Get embedding for user query
        query_emb = np.array(get_embedding(user_query))
        
        # Find the chunk most similar to the user query
        best_chunk = ""
        best_score = 0.0
        
        for chunk in chunks:
            chunk_emb = np.array(get_embedding(chunk))
            query_norm = np.linalg.norm(query_emb)
            chunk_norm = np.linalg.norm(chunk_emb)
            
            if query_norm > 0 and chunk_norm > 0:
                similarity = np.dot(query_emb, chunk_emb) / (query_norm * chunk_norm)
            else:
                similarity = 0.0
            
            if similarity > best_score:
                best_score = similarity
                best_chunk = chunk
        
        # Return a shortened version of the relevant section
        if len(best_chunk) > 500:
            return best_chunk[:500] + "..."
        return best_chunk if best_chunk else "No relevant section found."

# ----------------------------------------
# Initialize guardrails
# ----------------------------------------
if os.path.exists("guardrails"):
    guardrail_manager = GuardrailManager(folder="guardrails", chunk_size=1000, threshold=0.45)
else:
    os.makedirs("guardrails", exist_ok=True)
    guardrail_manager = GuardrailManager(folder="guardrails", chunk_size=1000, threshold=0.45)
    print("Created empty guardrails folder. Add legal documents to enable guardrail checks.")
#def check_guardrails(text):
#    """Wrapper for GuardrailManager"""
#    return guardrail_manager.check_violation(text)