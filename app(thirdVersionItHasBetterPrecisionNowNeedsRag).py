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

@cl.on_chat_start
async def start_chat():
    init_db()

    # Initialize guardrail manager
    guardrail_manager = GuardrailManager(
        folder="guardrails",
        thresholds={
            'default': 0.75,
            'prohibited': 0.30,
            'drugs': 0.10,
            'weapon': 0.35,
            'selfharm': 0.10,
            'discrimination': 0.10,
            'eu_ai_act': 0.25
        }
    )
    cl.user_session.set("guardrails", guardrail_manager)

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant that respects all applicable laws and regulations."}]
    )

    cl.user_session.set("collection", None)
    
    # Send welcome message
    await cl.Message(
        content="Welcome! I'm your AI assistant.\n\nYou can:\n- **Chat normally** by just typing your message\n- **Upload a document** for Q&A by clicking the paperclip icon 📎 and selecting a file\n- **Ask questions** about your uploaded document\n\nLet's get started! What would you like to know?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Check if this is a file upload request
        if message.elements and len(message.elements) > 0:
            # User uploaded a file with the message
            for element in message.elements:
                if isinstance(element, cl.File):
                    await process_uploaded_file(element)
            # Continue to process the message
            return
        
        collection = cl.user_session.get("collection")
        message_history = cl.user_session.get("message_history")

        # Guardrail check
        guardrail_manager = cl.user_session.get("guardrails")
        
        violation_result = guardrail_manager.check_violation(message.content)
        
        if violation_result:
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
        f"✅ Processing `{file.name}` complete.\n"
        f"Indexed **{len(chunks)} chunks**.\n\n"
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
# GuardrailManager class (same as before)
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
            
            # Determine category based on filename
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
                            elif isinstance(article_data, list):
                                self._add_guardrail(
                                    name=f"Article_{article_num}",
                                    category=category,
                                    data={'requires': article_data, 'prohibits': []},
                                    filename=filename
                                )
                    elif isinstance(data, list):
                        for i, article in enumerate(data):
                            if isinstance(article, dict):
                                self._add_guardrail(
                                    name=f"Article_{i+1}",
                                    category=category,
                                    data=article,
                                    filename=filename
                                )
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # For EU AI Act files, split into sentences for better matching
                    if category == 'eu_ai_act' or any(term in filename.lower() for term in ['eu_ai', 'act', 'prohibited']):
                        # Split by newlines if available (for list format)
                        if '\n' in content:
                            sentences = [s.strip() for s in content.split('\n') if s.strip() and len(s.strip()) > 10]
                        else:
                            # Split by sentences
                            sentences = re.split(r'[.!?]+', content)
                            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
                        
                        if sentences and len(sentences) > 1:
                            print(f"   Splitting into {len(sentences)} individual guardrails")
                            for i, sentence in enumerate(sentences):
                                self._add_guardrail(
                                    name=f"{filename}_rule_{i+1}",
                                    category=category,
                                    data={'text': sentence, 'name': filename},
                                    filename=filename
                                )
                        else:
                            # If only one sentence or no sentences, add as single guardrail
                            self._add_guardrail(
                                name=filename,
                                category=category,
                                data={'text': content, 'name': filename},
                                filename=filename
                            )
                    else:
                        # For other categories, keep as single guardrail
                        self._add_guardrail(
                            name=filename,
                            category=category,
                            data={'text': content, 'name': filename},
                            filename=filename
                        )
                        
            except Exception as e:
                print(f"   Warning: Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.guardrails)} guardrail entries")
        print(f"   Categories: {set(g['category'] for g in self.guardrails)}")
        
    def _add_guardrail(self, name, category, data, filename):
        """Add a guardrail with embeddings and keyword indexing"""
        
        if isinstance(data, dict):
            text_parts = []
            
            if data.get('title'):
                text_parts.append(data.get('title'))
            
            if data.get('description'):
                text_parts.append(data.get('description'))
            
            prohibitions = data.get('prohibits', [])
            if not isinstance(prohibitions, list):
                prohibitions = [prohibitions] if prohibitions else []
            
            for prohibition in prohibitions[:5]:
                text_parts.append(f"PROHIBITED: {prohibition}")
            
            requirements = data.get('requires', [])
            if not isinstance(requirements, list):
                requirements = [requirements] if requirements else []
            
            for requirement in requirements[:3]:
                text_parts.append(f"REQUIRED: {requirement}")
            
            exceptions = data.get('exceptions', [])
            if not isinstance(exceptions, list):
                exceptions = [exceptions] if exceptions else []
            
            for exception in exceptions[:2]:
                text_parts.append(f"EXCEPTION: {exception}")
            
            keywords = data.get('keywords', [])
            if not isinstance(keywords, list):
                keywords = [keywords] if keywords else []
            
            for keyword in keywords[:5]:
                text_parts.append(keyword)
            
            combined_text = " | ".join(text_parts) if text_parts else name
            
            prohibitions = prohibitions if isinstance(prohibitions, list) else [prohibitions] if prohibitions else []
            requirements = requirements if isinstance(requirements, list) else [requirements] if requirements else []
        else:
            combined_text = data.get('text', '')
            prohibitions = []
            requirements = []
        
        # Generate embedding
        embedding = np.array(get_embedding(combined_text[:2000]))
        
        # Extract keywords for fast pre-filtering
        keywords = self._extract_keywords(name, combined_text, category)
        
        # Store guardrail
        guardrail = {
            'name': name,
            'category': category,
            'filename': filename,
            'embedding': embedding,
            'keywords': keywords,
            'prohibitions': prohibitions,
            'requirements': requirements,
            'data': data
        }
        
        self.guardrails.append(guardrail)
        
        # Add to keyword index
        for keyword in keywords:
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = []
            self.keyword_index[keyword].append(len(self.guardrails) - 1)
    
    def _extract_keywords(self, name, text, category):
        """Extract important keywords from guardrail for fast matching"""
        keywords = set()
        
        # Add from filename
        name_clean = name.lower().replace('_', ' ').replace('.txt', '').replace('.json', '')
        keywords.add(name_clean)
        
        # Add category-specific keywords
        category_keywords = {
            'eu_ai_act': [
                'ai', 'act', 'regulation', 'eu', 'european', 'artificial', 'intelligence',
                'prohibited', 'high-risk', 'biometric', 'surveillance', 'scoring',
                'manipulation', 'social', 'credit', 'real-time', 'remote', 'identification',
                'facial', 'recognition', 'public', 'space', 'tracking', 'street'
            ],
            'drugs': ['drug', 'narcotic', 'substance', 'pharmaceutical', 'medication', 'controlled'],
            'weapon': ['weapon', 'firearm', 'gun', 'explosive', 'weaponry', 'armament', 'missile'],
            'selfharm': ['selfharm', 'suicide', 'self-harm', 'self_harm', 'kill', 'harm', 'hurt'],
            'discrimination': ['discrimin', 'racism', 'bias', 'equality', 'prejudice', 'sexism'],
            'medical': ['medical', 'health', 'treatment', 'diagnosis', 'therapy', 'clinical'],
            'prohibited': ['prohibited', 'forbidden', 'banned', 'illegal', 'not allowed']
        }
        
        if category in category_keywords:
            for kw in category_keywords[category]:
                keywords.add(kw)
        
        # Add common prohibited terms
        prohibited_terms = [
            'drug', 'weapon', 'violence', 'harm', 'kill', 'suicide', 'discrimination',
            'racism', 'hate', 'illegal', 'prohibited', 'biometric', 'surveillance',
            'scoring', 'manipulation', 'ai', 'act', 'regulation', 'credit', 'social',
            'facial', 'recognition', 'identification', 'public', 'space', 'tracking'
        ]
        
        text_lower = text.lower()
        for term in prohibited_terms:
            if term in text_lower:
                keywords.add(term)
        
        # Add any capitalized words that might be important
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for word in words[:5]:
            keywords.add(word.lower())
        
        return list(keywords)[:15]
    
    def _get_threshold(self, category):
        """Get threshold for a specific category"""
        # Higher threshold for self-harm to reduce false positives
        if category == 'selfharm':
            return self.thresholds.get(category, 0.45)
        elif category == 'eu_ai_act':
            return self.thresholds.get(category, 0.35)
        return self.thresholds.get(category, self.thresholds.get('default', 0.7))
    
    def check_violation(self, text):
        """Check if user query violates any guardrail with high precision"""
        print(f"\nChecking for guardrail violations...")
        print(f"   Query: {text[:100]}...")
        
        if not text or len(text.strip()) < 3:
            return None
        
        query_lower = text.lower()
        
        candidate_indices = set()
        
        # Special handling for EU AI Act biometric/facial queries
        if any(term in query_lower for term in ['facial', 'biometric', 'recognition', 'public street', 'public space']):
            # Add all EU AI Act guardrails that contain these terms
            for i, guardrail in enumerate(self.guardrails):
                if guardrail['category'] == 'eu_ai_act':
                    guardrail_text = guardrail['data'].get('text', '').lower()
                    if any(term in guardrail_text for term in ['facial', 'biometric', 'recognition', 'surveillance', 'public']):
                        candidate_indices.add(i)
                        print(f"   Added EU AI Act guardrail: {guardrail['name']}")
        
        # Check other keywords
        for keyword, indices in self.keyword_index.items():
            if keyword in query_lower:
                for idx in indices:
                    candidate_indices.add(idx)
        
        # If still no candidates, check guardrails with highest priority
        if not candidate_indices:
            priority_categories = ['eu_ai_act', 'prohibited', 'drugs', 'weapon', 'selfharm']
            for i, guardrail in enumerate(self.guardrails):
                if guardrail['category'] in priority_categories:
                    candidate_indices.add(i)
        
        print(f"   Candidate guardrails after keyword filter: {len(candidate_indices)}")
        
        if not candidate_indices:
            print(f"   No keyword matches found")
            return None
        
        query_emb = np.array(get_embedding(text))
        
        violations = []
        
        for idx in candidate_indices:
            guardrail = self.guardrails[idx]
            
            similarity = np.dot(query_emb, guardrail['embedding']) / (
                np.linalg.norm(query_emb) * np.linalg.norm(guardrail['embedding'])
            )
            
            # Boost similarity for EU AI Act biometric matches
            if guardrail['category'] == 'eu_ai_act':
                guardrail_text = guardrail['data'].get('text', '').lower()
                # Check if both query and guardrail contain biometric/facial terms
                query_has_terms = any(term in query_lower for term in ['facial', 'biometric', 'recognition', 'public'])
                guardrail_has_terms = any(term in guardrail_text for term in ['facial', 'biometric', 'recognition', 'public'])
                
                if query_has_terms and guardrail_has_terms:
                    similarity += 0.3  # Significant boost for matching terms
                    print(f"   Boosted similarity for {guardrail['name']} due to term match")
            
            threshold = self._get_threshold(guardrail['category'])
            
            print(f"   {guardrail['name']} ({guardrail['category']}): {similarity:.3f} (threshold: {threshold})")
            
            if similarity >= threshold:
                matched_content = None
                
                # For EU AI Act, automatically flag with detailed message
                if guardrail['category'] == 'eu_ai_act':
                    matched_content = guardrail['data'].get('text', 'Prohibited AI practice')
                    violations.append({
                        'guardrail': guardrail,
                        'similarity': similarity,
                        'matched_content': matched_content,
                        'matched_type': 'prohibition'
                    })
                else:
                    # Check exact prohibition matches for other categories
                    for prohibition in guardrail.get('prohibitions', []):
                        if isinstance(prohibition, str):
                            prohibition_lower = prohibition.lower()
                            key_phrase = ' '.join(prohibition_lower.split()[:5])
                            
                            if key_phrase in query_lower or any(word in query_lower for word in key_phrase.split()[:3]):
                                matched_content = prohibition
                                break
                    
                    if matched_content:
                        violations.append({
                            'guardrail': guardrail,
                            'similarity': similarity,
                            'matched_content': matched_content,
                            'matched_type': 'prohibition'
                        })
                    elif similarity >= threshold + 0.1:
                        # High similarity without exact match, still flag
                        matched_content = self._get_default_matched_content(guardrail['category'])
                        violations.append({
                            'guardrail': guardrail,
                            'similarity': similarity,
                            'matched_content': matched_content,
                            'matched_type': 'similar'
                        })
        
        if violations:
            # Sort by similarity and take best
            violations.sort(key=lambda x: x['similarity'], reverse=True)
            best = violations[0]
            
            print(f"   VIOLATION DETECTED: {best['guardrail']['name']} (score: {best['similarity']:.3f})")
            
            return self._format_violation(best)
        
        print(f"   No violations detected")
        return None
    
    def _get_default_matched_content(self, category):
        """Get default matched content based on category when no specific requirements exist"""
        default_messages = {
            'drugs': 'Content related to prohibited substances or drug-related activities',
            'weapon': 'Content related to weapons or prohibited items',
            'selfharm': 'Content related to self-harm or harm to others',
            'discrimination': 'Content that may promote discrimination or bias',
            'eu_ai_act': 'Content that may violate EU AI Act regulations regarding prohibited AI practices',
            'prohibited': 'Content that is prohibited by regulations',
            'medical': 'Content related to medical topics that may require professional oversight',
            'default': 'Content that may violate applicable regulations'
        }
        return default_messages.get(category, default_messages['default'])
    
    def _format_violation(self, violation):
        """Format violation message with clear guidance"""
        guardrail = violation['guardrail']
        similarity = violation['similarity']
        matched_content = violation['matched_content']
        
        # Category-specific messages
        category_messages = {
            'drugs': 'Your query relates to prohibited substances or drug-related activities.',
            'weapon': 'Your query relates to weapons or prohibited items.',
            'selfharm': 'Your query raises concerns about self-harm or harm to others.',
            'discrimination': 'Your query may promote discrimination or bias.',
            'eu_ai_act': 'Your query may violate EU AI Act regulations regarding prohibited AI practices.',
            'medical': 'Your query relates to medical topics that require professional oversight.',
            'prohibited': 'Your query may violate prohibited content guidelines.',
            'default': 'Your query may violate applicable regulations.'
        }
        
        message = category_messages.get(guardrail.get('category', 'default'), category_messages['default'])
        
        # Special handling for EU AI Act to provide more context
        if guardrail.get('category') == 'eu_ai_act':
            violation_msg = f"""
EU AI ACT VIOLATION DETECTED

{message}

The EU AI Act explicitly PROHIBITS the following AI practices:
• Biometric identification in publicly accessible spaces for law enforcement purposes
• Real-time remote biometric surveillance systems
• Social scoring systems that lead to detrimental treatment
• AI systems that manipulate human behavior to circumvent free will
• Emotion recognition systems in the workplace and educational institutions

Your query appears to involve prohibited AI practices.

Matched Document: `{guardrail.get('name', 'Unknown')}`
Confidence Score: {similarity:.2%}

Relevant Prohibition:
> {matched_content}

This is a violation of the EU AI Act. Please rephrase your query to comply with AI regulations.
"""
        else:
            violation_msg = f"""
GUARDRAIL VIOLATION DETECTED

{message}

Matched Document: `{guardrail.get('name', 'Unknown')}`
Confidence Score: {similarity:.2%}

Relevant Content:
> {matched_content}

Please rephrase your query or ensure it complies with applicable laws and regulations.

If you believe this is an error, please clarify your intent or contact support.
"""
    
        return violation_msg

# ----------------------------------------
# Initialize guardrails
# ----------------------------------------
print("Initializing Guardrail Manager...")
guardrail_manager = GuardrailManager(
    folder="guardrails",
    thresholds={
        'default': 0.70,
        'prohibited': 0.30,
        'drugs': 0.25,
        'weapon': 0.35,
        'selfharm': 0.45,  # Increased threshold to reduce false positives
        'discrimination': 0.30,
        'medical': 0.35,
        'eu_ai_act': 0.35
    }
)
print("Guardrail Manager ready!")