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
import json
import re
from pathlib import Path
import asyncio

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

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def split_into_sentences(text):
    """Split text into individual sentences for better guardrail matching"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return sentences

# Global EU AI Act vector store
eu_ai_act_collection = None

def load_eu_ai_act_to_vector_db():
    """Load EU AI Act documents into vector database for RAG"""
    global eu_ai_act_collection
    print("Loading EU AI Act into vector database...")
    
    eu_ai_act_folder = "guardrails/eu_ai_act"
    
    try:
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection(name="eu_ai_act_rag")
        except Exception:
            pass
        
        # Create new collection
        collection = chroma_client.create_collection(name="eu_ai_act_rag")
        
        # Load all EU AI Act files
        if os.path.exists(eu_ai_act_folder):
            file_count = 0
            chunk_count = 0
            
            for filename in os.listdir(eu_ai_act_folder):
                if filename.endswith('.txt'):
                    filepath = os.path.join(eu_ai_act_folder, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Chunk the content for better retrieval
                    chunks = chunk_text(content, chunk_size=300)
                    
                    for i, chunk in enumerate(chunks):
                        embedding = get_embedding(chunk)
                        collection.add(
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{"source": filename, "chunk": i}],
                            ids=[f"eu_ai_act_{filename}_{i}"]
                        )
                        chunk_count += 1
                    
                    file_count += 1
                    print(f"   Loaded {filename} -> {len(chunks)} chunks")
            
            print(f"Loaded {file_count} files, {chunk_count} chunks into EU AI Act RAG")
        else:
            print(f"Warning: EU AI Act folder not found at {eu_ai_act_folder}")
        
        return collection
        
    except Exception as e:
        print(f"Error loading EU AI Act: {e}")
        return None

async def get_eu_ai_act_context(query, n_results=3):
    """Retrieve relevant EU AI Act context for a query"""
    global eu_ai_act_collection
    
    if eu_ai_act_collection is None:
        return None
    
    try:
        query_embedding = get_embedding(query)
        results = eu_ai_act_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if results and results['documents'] and len(results['documents'][0]) > 0:
            context = []
            for i, doc in enumerate(results['documents'][0]):
                source = results['metadatas'][0][i]['source'] if results['metadatas'] else "Unknown"
                context.append(f"[Source: {source}]\n{doc}")
            
            return "\n\n".join(context)
        
        return None
    except Exception as e:
        print(f"Error retrieving EU AI Act context: {e}")
        return None

@cl.on_chat_start
async def start_chat():
    global eu_ai_act_collection
    
    init_db()

    # Load EU AI Act into vector database
    eu_ai_act_collection = load_eu_ai_act_to_vector_db()

    # Initialize guardrail manager with ALL files and appropriate thresholds
    guardrail_manager = GuardrailManager(
        folder="guardrails",
        thresholds={
            'default': 0.70,
            'prohibited': 0.30,
            'drugs': 0.25,
            'weapon': 0.35,
            'selfharm': 0.45,
            'discrimination': 0.30,
            'eu_ai_act': 0.35,
            'medical': 0.35
        }
    )
    cl.user_session.set("guardrails", guardrail_manager)

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant that respects all applicable laws and regulations. When discussing EU AI Act violations, provide specific references to the regulation."}]
    )

    cl.user_session.set("collection", None)
    
    # Send welcome message
    await cl.Message(
        content="Welcome! I'm your AI assistant.\n\nYou can:\n- **Chat normally** by just typing your message\n- **Upload a document** for Q&A by clicking the paperclip icon and selecting a file\n- **Ask questions** about your uploaded document\n\nI also have the EU AI Act loaded for reference when discussing AI regulations.\n\nLet's get started! What would you like to know?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Check if this is a file upload request
        if message.elements and len(message.elements) > 0:
            for element in message.elements:
                if isinstance(element, cl.File):
                    await process_uploaded_file(element)
            return
        
        collection = cl.user_session.get("collection")
        message_history = cl.user_session.get("message_history")

        # Guardrail check
        guardrail_manager = cl.user_session.get("guardrails")
        
        violation_result = guardrail_manager.check_violation(message.content)
        
        if violation_result:
            # If it's an EU AI Act violation, enhance with RAG context
            if "EU AI ACT VIOLATION" in violation_result:
                # Get relevant EU AI Act context
                eu_context = await get_eu_ai_act_context(message.content)
                
                if eu_context:
                    # Generate a detailed response with RAG context
                    enhanced_violation = f"""
{violation_result}

**Detailed EU AI Act Information:**
{eu_context}

For more information, please refer to the official EU AI Act documentation.
"""
                    await cl.Message(content=enhanced_violation).send()
                else:
                    await cl.Message(content=violation_result).send()
            else:
                await cl.Message(content=violation_result).send()
            return

        # Save user message
        save_message("user", message.content)

        # Handle chat or doc mode
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
            enhanced_message = message.content

        temp_history = message_history.copy()
        temp_history.append({"role": "user", "content": enhanced_message})

        msg = cl.Message(content="")
        await msg.send()

        try:
            stream = await client.chat.completions.create(
                messages=temp_history,
                stream=True,
                **settings
            )

            async for part in stream:
                if hasattr(part, 'choices') and part.choices and len(part.choices) > 0:
                    if hasattr(part.choices[0], 'delta') and part.choices[0].delta.content:
                        token = part.choices[0].delta.content
                        if token:
                            await msg.stream_token(token)
                else:
                    print(f"Warning: Received empty chunk: {part}")
                    continue

        except Exception as e:
            error_msg = f"Sorry, an error occurred while generating the response: {str(e)}"
            await msg.stream_token(error_msg)
            print(f"Error in stream: {e}")

        message_history.append({"role": "user", "content": message.content})
        message_history.append({"role": "assistant", "content": msg.content})

        cl.user_session.set("message_history", message_history)
        save_message("assistant", msg.content)
        await msg.update()
        
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(f"Unexpected error in main: {e}")
        await cl.Message(content=error_message).send()

async def process_uploaded_file(file):
    """Process uploaded file and create vector store"""
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

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

    chunks = chunk_text(text)

    try:
        chroma_client.delete_collection(name="documents")
    except Exception:
        pass

    collection = chroma_client.create_collection(name="documents")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            ids=[f"chunk_{i}"]
        )

    cl.user_session.set("collection", collection)

    msg.content = (
        f"Processing `{file.name}` complete.\n"
        f"Indexed {len(chunks)} chunks.\n\n"
        f"You can now ask questions about the document!"
    )

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
# GuardrailManager class (with previous improvements)
# ===================================
class GuardrailManager:
    def __init__(self, folder="guardrails", thresholds=None):
        self.folder = folder
        self.thresholds = thresholds or {'default': 0.75}
        self.guardrails = []
        self.keyword_index = {}
        self.load_all_guardrails()
    
    def _determine_category(self, filename):
        """Determine the category of a guardrail file based on filename"""
        file_lower = filename.lower()
        
        categories = {
            'drugs': ['drug', 'narcotic', 'substance', 'drugs.txt'],
            'weapon': ['weapon', 'firearm', 'gun', 'explosive', 'weaponry'],
            'selfharm': ['selfharm', 'suicide', 'self-harm', 'self_harm', 'harm.txt'],
            'discrimination': ['discrimin', 'racism', 'bias', 'equality'],
            'medical': ['medical', 'health', 'treatment'],
            'eu_ai_act': ['eu_ai', 'act', 'prohibited', 'biometric', 'surveillance', 'facial', 'recognition', 'social scoring'],
            'prohibited': ['prohibited']
        }
        
        for category, patterns in categories.items():
            if any(pattern in file_lower for pattern in patterns):
                return category
        
        return 'default'
        
    def load_all_guardrails(self):
        """Load ALL guardrail files and split into sentences for better matching"""
        print(f"Loading guardrails from: {self.folder}")
        
        for filename in os.listdir(self.folder):
            filepath = os.path.join(self.folder, filename)
            
            if os.path.isdir(filepath):
                continue
            
            category = self._determine_category(filename)
            
            print(f"Loading: {filename} (category: {category})")
            
            try:
                if filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        for article_num, article_data in data.items():
                            if isinstance(article_data, dict):
                                self._add_guardrail(
                                    name=f"Article_{article_num}",
                                    category=category,
                                    data=article_data,
                                    filename=filename
                                )
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # For EU AI Act files, split into sentences for better matching
                    if category == 'eu_ai_act':
                        # Split by newlines if available
                        if '\n' in content:
                            sentences = [s.strip() for s in content.split('\n') if s.strip() and len(s.strip()) > 10]
                        else:
                            sentences = split_into_sentences(content)
                        
                        if sentences and len(sentences) > 1:
                            print(f"   Splitting into {len(sentences)} individual guardrails")
                            for i, sentence in enumerate(sentences[:50]):  # Limit to first 50 for performance
                                self._add_guardrail(
                                    name=f"{filename}_rule_{i+1}",
                                    category=category,
                                    data={'text': sentence, 'name': filename},
                                    filename=filename
                                )
                        else:
                            self._add_guardrail(
                                name=filename,
                                category=category,
                                data={'text': content, 'name': filename},
                                filename=filename
                            )
                    else:
                        self._add_guardrail(
                            name=filename,
                            category=category,
                            data={'text': content, 'name': filename},
                            filename=filename
                        )
                        
            except Exception as e:
                print(f"   Warning: Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.guardrails)} guardrail entries")
        
    def _add_guardrail(self, name, category, data, filename):
        """Add a guardrail with embeddings and keyword indexing"""
        
        if isinstance(data, dict):
            text_parts = []
            
            if data.get('title'):
                text_parts.append(data.get('title'))
            
            prohibitions = data.get('prohibits', [])
            if not isinstance(prohibitions, list):
                prohibitions = [prohibitions] if prohibitions else []
            
            for prohibition in prohibitions[:5]:
                text_parts.append(f"PROHIBITED: {prohibition}")
            
            combined_text = " | ".join(text_parts) if text_parts else name
            prohibitions = prohibitions
        else:
            combined_text = data.get('text', '')
            prohibitions = []
        
        embedding = np.array(get_embedding(combined_text[:2000]))
        keywords = self._extract_keywords(name, combined_text, category)
        
        guardrail = {
            'name': name,
            'category': category,
            'filename': filename,
            'embedding': embedding,
            'keywords': keywords,
            'prohibitions': prohibitions,
            'data': data
        }
        
        self.guardrails.append(guardrail)
        
        for keyword in keywords:
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = []
            self.keyword_index[keyword].append(len(self.guardrails) - 1)
    
    def _extract_keywords(self, name, text, category):
        """Extract important keywords from guardrail for fast matching"""
        keywords = set()
        
        name_clean = name.lower().replace('_', ' ').replace('.txt', '').replace('.json', '')
        keywords.add(name_clean)
        
        category_keywords = {
            'eu_ai_act': ['facial', 'biometric', 'recognition', 'surveillance', 'public', 'street', 'ai', 'act'],
            'selfharm': ['kill', 'suicide', 'harm', 'hurt', 'self'],
            'drugs': ['drug', 'narcotic', 'substance'],
            'weapon': ['weapon', 'gun', 'firearm', 'explosive']
        }
        
        if category in category_keywords:
            for kw in category_keywords[category]:
                keywords.add(kw)
        
        text_lower = text.lower()
        for term in ['facial', 'biometric', 'recognition', 'surveillance']:
            if term in text_lower:
                keywords.add(term)
        
        return list(keywords)[:15]
    
    def _get_threshold(self, category):
        """Get threshold for a specific category"""
        if category == 'selfharm':
            return self.thresholds.get(category, 0.45)
        elif category == 'eu_ai_act':
            return self.thresholds.get(category, 0.35)
        return self.thresholds.get(category, self.thresholds.get('default', 0.7))
    
    def check_violation(self, text):
        """Check if user query violates any guardrail"""
        print(f"\nChecking for guardrail violations...")
        print(f"   Query: {text[:100]}...")
        
        if not text or len(text.strip()) < 3:
            return None
        
        query_lower = text.lower()
        
        candidate_indices = set()
        
        # Special handling for biometric/facial queries
        if any(term in query_lower for term in ['facial', 'biometric', 'recognition', 'public street']):
            for i, guardrail in enumerate(self.guardrails):
                if guardrail['category'] == 'eu_ai_act':
                    guardrail_text = guardrail['data'].get('text', '').lower()
                    if any(term in guardrail_text for term in ['facial', 'biometric', 'recognition']):
                        candidate_indices.add(i)
        
        for keyword, indices in self.keyword_index.items():
            if keyword in query_lower:
                for idx in indices:
                    candidate_indices.add(idx)
        
        if not candidate_indices:
            return None
        
        query_emb = np.array(get_embedding(text))
        violations = []
        
        for idx in candidate_indices:
            guardrail = self.guardrails[idx]
            
            similarity = np.dot(query_emb, guardrail['embedding']) / (
                np.linalg.norm(query_emb) * np.linalg.norm(guardrail['embedding'])
            )
            
            threshold = self._get_threshold(guardrail['category'])
            
            print(f"   {guardrail['name']} ({guardrail['category']}): {similarity:.3f} (threshold: {threshold})")
            
            if similarity >= threshold:
                matched_content = guardrail['data'].get('text', 'Prohibited content')
                violations.append({
                    'guardrail': guardrail,
                    'similarity': similarity,
                    'matched_content': matched_content
                })
        
        if violations:
            violations.sort(key=lambda x: x['similarity'], reverse=True)
            best = violations[0]
            return self._format_violation(best)
        
        return None
    
    def _format_violation(self, violation):
        """Format violation message with clear guidance"""
        guardrail = violation['guardrail']
        similarity = violation['similarity']
        matched_content = violation['matched_content']
        
        if guardrail.get('category') == 'eu_ai_act':
            return f"""
EU AI ACT VIOLATION DETECTED

Your query relates to prohibited AI practices under the EU AI Act.

**Matched Prohibition:** `{guardrail.get('name', 'Unknown')}`
**Confidence Score:** {similarity:.2%}

**Relevant Regulation Text:**
> {matched_content}

The EU AI Act prohibits certain AI systems including biometric identification in public spaces, social scoring, and manipulative AI.

Please rephrase your query to comply with EU regulations.
"""
        else:
            return f"""
GUARDRAIL VIOLATION DETECTED

Your query has been flagged for potential policy violation.

**Category:** {guardrail.get('category', 'unknown')}
**Confidence Score:** {similarity:.2%}

**Matched Content:**
> {matched_content}

Please rephrase your query or ensure it complies with applicable guidelines.
"""

# ----------------------------------------
# Initialize guardrails
# ----------------------------------------
print("Initializing Guardrail Manager...")
guardrail_manager = GuardrailManager(
    folder="guardrails",
    thresholds={
        'default': 0.70,
        'prohibited': 0.30,
        'drugs': 0.3,
        'weapon': 0.4,
        'selfharm': 0.45,
        'discrimination': 0.4,
        'medical': 0.4,
        'eu_ai_act': 0.4
    }
)
print("Guardrail Manager ready!")