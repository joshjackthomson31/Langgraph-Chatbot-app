import streamlit as st
import base64
import re
import json
import os
import requests
from html import unescape
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import PyPDF2
from typing import Optional
import uuid

# RAG imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Try to import pdfplumber for better PDF extraction
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Try to import OCR libraries for scanned PDFs
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# =============================================================================
# CONSTANTS
# =============================================================================
CONVERSATIONS_FILE = "conversations.json"
MAX_CONTEXT_MESSAGES = 20  # Keep last N messages for context

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================
CONVERSATION_PROMPT = """You are a helpful, friendly AI assistant similar to ChatGPT.

MEMORY RULES:
- Remember everything the user tells you (name, job, favorites, etc.)
- When user asks "my favorite", "my name", "what do I do", recall from conversation history
- If user asks about something they mentioned before, use that context

RESPONSE RULES:
- Be concise and direct - don't be verbose
- DO NOT repeat the user's personal info in every response (like "You work at Google, Alex")
- Only mention their personal info if directly relevant to their question
- Use markdown formatting when helpful
- Be natural and conversational
"""

def get_current_date():
    """Get current date formatted nicely."""
    return datetime.now().strftime("%B %d, %Y")

SEARCH_PROMPT = """You answer questions using the provided search results.

CRITICAL RULES:
- Use ONLY the search results provided
- Give a clear, direct answer - be concise
- If search results seem conflicting or unclear, mention that the info may need verification
- For very recent events (last few days), add a note that details may still be developing
- If search results don't have complete information (e.g., only 2 out of 5 requested items), clearly state what's available and what's missing
- Do NOT fill in gaps with guesses - only report what's actually in the search results
- If you're not confident about accuracy, say "Based on search results..." to indicate uncertainty
- Do NOT make up information - only use what's in the search results
- Be confident and direct when results are clear
"""

RAG_PROMPT = """You answer questions based ONLY on the provided document context.

CRITICAL RULES:
- Use ONLY information that is EXPLICITLY stated in the document context
- NEVER make up or guess values - only use exact values from the document
- If a grade is mentioned, use the EXACT grade shown (e.g., "B-" not "A+")
- If a number like CGPA is mentioned, use the EXACT number from the document
- Be concise and direct
- If the answer is NOT in the document, say "I couldn't find that in the uploaded document"
- Quote the relevant text when answering to prove your answer is from the document
- DO NOT use any prior knowledge or conversation context - ONLY the document
"""

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    /* Main app background - ChatGPT style dark */
    .stApp { 
        background: #212121;
    }
    
    /* Hide default header */
    [data-testid="stHeader"] {
        background: #212121 !important;
        border-bottom: 1px solid #333 !important;
    }
    
    /* Sidebar - conversation list */
    [data-testid="stSidebar"] { 
        background: #171717 !important;
        border-right: 1px solid #333 !important;
    }
    [data-testid="stSidebar"] * { color: #ececec !important; }
    
    /* Sidebar header */
    .sidebar-header {
        padding: 1rem;
        border-bottom: 1px solid #333;
        margin-bottom: 1rem;
    }
    .sidebar-header h2 { 
        color: white !important; 
        margin: 0; 
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Main chat area */
    .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 1.5rem 0 !important;
        border-bottom: 1px solid #333 !important;
    }
    
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] code { 
        color: #ececec !important; 
    }
    
    /* Code blocks */
    [data-testid="stChatMessage"] pre {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
    
    /* Chat avatars */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background: #10a37f !important;
        border-radius: 4px !important;
        width: 30px !important;
        height: 30px !important;
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        background: #212121 !important;
        border-top: 1px solid #333 !important;
        padding: 1rem 0 !important;
    }
    [data-testid="stChatInput"] > div {
        background: #1a1a1a !important;
        border: 1px solid #555 !important;
        border-radius: 12px !important;
        max-width: 800px;
        margin: 0 auto;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] [contenteditable],
    [data-testid="stChatInput"] * {
        background: transparent !important;
        background-color: transparent !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: none !important;
        caret-color: #ffffff !important;
        font-size: 1rem !important;
    }
    [data-testid="stChatInput"] button {
        background: transparent !important;
        color: #ececec !important;
    }
    [data-testid="stChatInput"] button * {
        color: #ececec !important;
        -webkit-text-fill-color: #ececec !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #666 !important;
        -webkit-text-fill-color: #666 !important;
    }
    /* Force text visibility in all input areas */
    [data-baseweb="textarea"] textarea,
    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: transparent !important;
    }
    /* Override any conflicting styles */
    .stChatInput textarea,
    .stChatInputContainer textarea,
    div[data-testid="stChatInput"] div textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: #1a1a1a !important;
    }
    
    /* Bottom area */
    [data-testid="stBottom"] {
        background: #212121 !important;
    }
    
    /* Buttons in sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background: #212121 !important;
        border: 1px solid #444 !important;
        color: white !important;
        border-radius: 8px !important;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #2a2a2a !important;
        border-color: #555 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #2a2a2a !important;
        border: 1px dashed #444 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"] label {
        color: #ececec !important;
    }
    
    /* Welcome message */
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
    }
    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: #ececec;
        margin-bottom: 2rem;
    }
    .welcome-subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 3rem;
    }
    
    /* Spinner */
    .stSpinner > div { color: #ececec !important; }
    
    /* Hide footer */
    footer { display: none !important; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #212121; }
    ::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #555; }
    
    /* Success/error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        background: #303030 !important;
        color: #ececec !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONVERSATION PERSISTENCE
# =============================================================================
def load_conversations():
    """Load conversations from file."""
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_conversations(conversations):
    """Save conversations to file."""
    try:
        with open(CONVERSATIONS_FILE, 'w') as f:
            json.dump(conversations, f, indent=2)
    except:
        pass

def get_conversation_title(messages):
    """Generate a title from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content[:30] + "..." if len(content) > 30 else content
    return "New conversation"

def messages_to_dict(messages):
    """Convert LangChain messages to dict for storage."""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result

def dict_to_messages(message_dicts):
    """Convert dict messages back to LangChain messages."""
    result = []
    for msg in message_dicts:
        if msg["role"] == "user":
            result.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            result.append(AIMessage(content=msg["content"]))
    return result

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_attachment" not in st.session_state:
    st.session_state.current_attachment = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "last_route" not in st.session_state:
    st.session_state.last_route = None  # Track previous route for follow-ups

if "last_resolved_topic" not in st.session_state:
    st.session_state.last_resolved_topic = None  # Track the last resolved query for chained follow-ups

if "document_is_mine" not in st.session_state:
    st.session_state.document_is_mine = False  # Track if user claims document ownership

if "document_person" not in st.session_state:
    st.session_state.document_person = None  # Track the person mentioned in uploaded document

# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================
def extract_pdf_text(pdf_file) -> str:
    """Extract text from PDF using pdfplumber or PyPDF2."""
    pdf_file.seek(0)
    text = ""
    
    # Try pdfplumber first (best for text-based PDFs)
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            pass
    
    # Try PyPDF2 as fallback
    if not text.strip():
        pdf_file.seek(0)
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except:
            pass
    
    return text.strip()

def extract_pdf_with_ocr(pdf_file) -> str:
    """Extract text from scanned PDF using OCR."""
    if not HAS_OCR:
        return ""
    
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    text = ""
    
    try:
        # Convert PDF pages to images
        images = convert_from_bytes(pdf_bytes, dpi=200)
        
        # OCR each page
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            if page_text.strip():
                text += f"\n--- Page {i+1} ---\n{page_text}"
        
        return text.strip()
    except Exception as e:
        st.warning(f"OCR error: {str(e)[:100]}")
        return ""

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return attachment info."""
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    
    try:
        if file_type == "application/pdf":
            # First try normal text extraction
            text = extract_pdf_text(uploaded_file)
            
            # If no text found, try OCR for scanned PDFs
            if not text or len(text.strip()) < 50:
                if HAS_OCR:
                    st.info("📷 Scanned PDF detected. Running OCR...")
                    uploaded_file.seek(0)
                    text = extract_pdf_with_ocr(uploaded_file)
                    
            if text and len(text.strip()) > 50:
                return {"type": "document", "name": file_name, "data": text}
            else:
                # Scanned PDF or extraction failed
                st.warning(f"⚠️ Could not extract text from {file_name}. It may be a scanned PDF.")
                return None
        
        elif file_type.startswith("image/"):
            image_data = base64.b64encode(uploaded_file.read()).decode("utf-8")
            return {"type": "image", "name": file_name, "data": image_data, "mime_type": file_type}
        
        elif file_type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            if text.strip():
                return {"type": "document", "name": file_name, "data": text}
            else:
                st.warning("⚠️ The text file appears to be empty.")
                return None
        
        else:
            st.warning(f"⚠️ Unsupported file type: {file_type}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)[:100]}")
        return None

# =============================================================================
# RAG FUNCTIONS
# =============================================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(text: str) -> FAISS:
    """Create vector store from document text."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(documents, get_embeddings())

def retrieve_context(vector_store: FAISS, query: str, k: int = 5) -> str:
    """Retrieve relevant chunks from vector store."""
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([f"[Chunk {i+1}]:\n{doc.page_content}" for i, doc in enumerate(docs)])

# =============================================================================
# WEB SEARCH FUNCTIONS
# =============================================================================

def tavily_search(query: str) -> str:
    """Search using Tavily API (free tier: 1000 searches/month). Best for recent events."""
    try:
        api_key = st.secrets.get("TAVILY_API_KEY")
        if not api_key:
            return ""
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Use "advanced" search for list/comprehensive queries
        is_list_query = any(word in query.lower() for word in ["list", "last", "all", "every", "each", "top", "best", "worst", "most", "least"])
        
        # For stats/data/list queries, enhance with year context
        search_query = query
        current_year = datetime.now().year
        
        # Add year range for "last X" queries
        # E.g., "last 5 [items]" -> adds years to help search find recent data
        if "last" in query.lower() and re.search(r'last\s+\d+', query.lower()):
            num_match = re.search(r'last\s+(\d+)', query.lower())
            if num_match:
                num_years = int(num_match.group(1))
                years = [str(current_year - i) for i in range(num_years)]
                search_query = f"{query} {' '.join(years)}"
        
        # For stats/data queries, add "latest" to get more recent data
        data_keywords = ["stats", "statistics", "records", "current", "latest", "recent", "now", "today"]
        if any(kw in query.lower() for kw in data_keywords):
            if str(current_year) not in search_query and "latest" not in query.lower():
                search_query = f"{search_query} {current_year} latest"
        
        payload = {
            "api_key": api_key,
            "query": search_query,
            "search_depth": "advanced" if is_list_query else "basic",  # Advanced for lists
            "include_answer": True,
            "include_raw_content": False,
            "max_results": 10 if is_list_query else 5  # More results for lists
        }
        
        response = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            headers=headers,
            timeout=20
        )
        
        if response.status_code != 200:
            return ""
        
        data = response.json()
        results = []
        
        # Tavily provides a direct answer - great!
        if data.get("answer"):
            results.append(f"**Direct Answer:**\n{data['answer']}")
        
        # Include more search results for list queries
        max_items = 8 if is_list_query else 3
        for item in data.get("results", [])[:max_items]:
            title = item.get("title", "")
            content = item.get("content", "")[:600]
            if title and content:
                results.append(f"**{title}**\n{content}")
        
        return "\n\n".join(results)
    except Exception as e:
        return ""

def serper_search(query: str) -> str:
    """Search using Serper API (Google Search results). Free tier: 2,500 searches/month."""
    try:
        api_key = st.secrets.get("SERPER_API_KEY")
        if not api_key:
            return ""
        
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
        # Determine if this is a list query for more results
        is_list_query = any(word in query.lower() for word in ["list", "last", "all", "every", "each", "top", "best", "worst", "most", "least"])
        
        payload = {
            "q": query,
            "num": 10 if is_list_query else 5
        }
        
        response = requests.post(
            "https://google.serper.dev/search",
            json=payload,
            headers=headers,
            timeout=15
        )
        
        if response.status_code != 200:
            return ""
        
        data = response.json()
        results = []
        
        # Get answer box if available (direct answer from Google)
        if data.get("answerBox"):
            answer_box = data["answerBox"]
            if answer_box.get("answer"):
                results.append(f"**Direct Answer:**\n{answer_box['answer']}")
            elif answer_box.get("snippet"):
                results.append(f"**Direct Answer:**\n{answer_box['snippet']}")
        
        # Get knowledge graph if available
        if data.get("knowledgeGraph"):
            kg = data["knowledgeGraph"]
            title = kg.get("title", "")
            description = kg.get("description", "")
            if title and description:
                results.append(f"**{title}**\n{description}")
        
        # Get organic search results
        for item in data.get("organic", [])[:8 if is_list_query else 4]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title and snippet:
                results.append(f"**{title}**\n{snippet}")
        
        return "\n\n".join(results)
    except Exception as e:
        return ""

def searxng_search(query: str) -> str:
    """Search using public SearXNG instances (completely free, no API key)."""
    try:
        # List of public SearXNG instances
        instances = [
            "https://searx.be",
            "https://search.sapti.me", 
            "https://searx.tiekoetter.com",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        for instance in instances:
            try:
                url = f"{instance}/search"
                params = {
                    "q": query,
                    "format": "json",
                    "categories": "general",
                }
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for item in data.get("results", [])[:5]:
                        title = item.get("title", "")
                        content = item.get("content", "")[:400]
                        if title and content:
                            results.append(f"**{title}**\n{content}")
                    
                    if results:
                        return "\n\n".join(results)
            except:
                continue
        
        return ""
    except:
        return ""

def web_search(query: str, max_results: int = 8) -> str:
    """Search the web using DuckDuckGo."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return ""
        
        titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', response.text)
        raw_snippets = re.findall(r'class="result__snippet"[^>]*>(.+?)</a>', response.text, re.DOTALL)
        
        results = []
        for i, (title, raw) in enumerate(zip(titles[:max_results], raw_snippets[:max_results]), 1):
            snippet = unescape(re.sub(r'<[^>]+>', '', raw)).strip()
            if title.strip() and snippet:
                results.append(f"**{title.strip()}**\n{snippet}")
        
        return "\n\n".join(results)
    except Exception as e:
        return ""

def duckduckgo_api_search(query: str) -> str:
    """Fallback: Use DuckDuckGo Instant Answer API."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_html=1"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return ""
        
        data = response.json()
        results = []
        
        # Abstract (main answer)
        if data.get("Abstract"):
            results.append(f"**{data.get('Heading', 'Answer')}**\n{data['Abstract']}")
        
        # Related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])
        
        return "\n\n".join(results)
    except:
        return ""

def wikipedia_search(query: str) -> str:
    """Search Wikipedia for factual information."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        search_url = "https://en.wikipedia.org/w/api.php"
        
        # Clean up query for better Wikipedia search
        clean_query = query.replace("?", "").replace('"', "").strip()
        
        # Remove common question prefixes that don't help search
        prefixes_to_remove = [
            "who is the", "who was the", "who won the", "who holds the",
            "what is the", "what was the", "what are the",
            "tell me about", "list the", "list all",
            "how many", "when did", "when was",
        ]
        query_lower = clean_query.lower()
        for prefix in prefixes_to_remove:
            if query_lower.startswith(prefix):
                clean_query = clean_query[len(prefix):].strip()
                break
        
        params = {"action": "query", "list": "search", "srsearch": clean_query, "srlimit": 3, "format": "json"}
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return ""
        
        results = []
        for item in response.json().get("query", {}).get("search", []):
            title = item.get("title", "")
            content_params = {"action": "query", "titles": title, "prop": "extracts", "explaintext": True, "format": "json"}
            content_resp = requests.get(search_url, params=content_params, headers=headers, timeout=10)
            
            if content_resp.status_code == 200:
                pages = content_resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "")[:2000]
                    if extract:
                        results.append(f"**Wikipedia: {title}**\n{extract}")
        
        return "\n\n".join(results)
    except:
        return ""

def search_internet(query: str) -> str:
    """Search internet using multiple sources with fallbacks."""
    all_results = []
    
    # Extract year if present (for year-specific queries)
    year_match = re.search(r'\b(20[12]\d)\b', query)
    query_year = year_match.group(1) if year_match else None
    
    # General query cleaning - works for ALL queries
    search_query = query.replace("?", "").replace('"', "").strip()
    
    # Remove question prefixes that don't help search engines
    prefixes = [
        "who is the", "who was the", "who won the", "who holds the", "who are the",
        "what is the", "what was the", "what are the", "what does",
        "when did the", "when was the", "when is the",
        "where is the", "where was the",
        "how many", "how much", "how old is",
        "tell me about", "list the", "list all", "list last",
        "can you tell me", "do you know",
    ]
    query_lower = search_query.lower()
    for prefix in prefixes:
        if query_lower.startswith(prefix):
            search_query = search_query[len(prefix):].strip()
            break
    
    # For year-specific queries, also try with year at end (Wikipedia format)
    alternate_query = None
    if query_year and search_query.startswith(query_year):
        alternate_query = search_query.replace(query_year, "").strip() + " " + query_year
    
    # === PRIORITY 1: Tavily (best for recent events, has direct answers) ===
    tavily = tavily_search(query)  # Use original query for best results
    if tavily:
        all_results.append(tavily)
    
    # === PRIORITY 2: Serper (Google Search results, very accurate) ===
    if not all_results or len(all_results) < 2:
        serper = serper_search(query)
        if serper:
            all_results.append(serper)
    
    # === PRIORITY 3: SearXNG (free, good for current events) ===
    if not all_results:
        searx = searxng_search(search_query)
        if searx:
            all_results.append(searx)
    
    # === PRIORITY 3: DuckDuckGo HTML search ===
    if not all_results:
        ddg = web_search(search_query)
        if ddg:
            all_results.append(ddg)
    
    # Try alternate query if we have one and still no results
    if alternate_query and not all_results:
        ddg_alt = web_search(alternate_query)
        if ddg_alt:
            all_results.append(ddg_alt)
    
    # === PRIORITY 4: Wikipedia (best for historical/factual info) ===
    wiki = wikipedia_search(search_query)
    if wiki:
        all_results.append(wiki)
    
    # Try alternate query on Wikipedia
    if alternate_query and len(all_results) < 2:
        wiki_alt = wikipedia_search(alternate_query)
        if wiki_alt:
            all_results.append(wiki_alt)
    
    # === Fallbacks ===
    if not all_results:
        ddg_api = duckduckgo_api_search(search_query)
        if ddg_api:
            all_results.append(ddg_api)
    
    if not all_results:
        wiki_retry = wikipedia_search(query)
        if wiki_retry:
            all_results.append(wiki_retry)
    
    if not all_results:
        return ""
    
    return "\n\n---\n\n".join(all_results)

# =============================================================================
# ROUTING LOGIC
# =============================================================================
def determine_route(query: str, has_document: bool, conversation_history: list = None) -> str:
    """Determine which route to take."""
    query_lower = query.lower().strip()
    
    # ==========================================================================
    # FIRST: Check for personal DECLARATIONS (user sharing info, not asking)
    # These should ALWAYS go to conversation, even if document is uploaded
    # ==========================================================================
    personal_declarations = [
        "my name is", "i'm ", "i am ", "i work at", "i work for",
        "i like ", "i love ", "is my favorite", "is my favourite",
        "call me ", "i live in", "hi,", "hello,", "hey,"
    ]
    if any(p in query_lower for p in personal_declarations):
        return "conversation"
    
    # Simple greetings - also always conversation
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
                 'bye', 'goodbye', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no',
                 'how are you', "what's up", 'sup', 'yo', 'good']
    if query_lower.rstrip('!?.') in greetings or query_lower.rstrip('!?.').split(',')[0].strip() in greetings:
        return "conversation"
    
    # ==========================================================================
    # DOCUMENT OWNERSHIP DECLARATION
    # If user says "this is my document/transcript/resume", remember it
    # ==========================================================================
    ownership_patterns = [
        r'\b(?:this|the)\s+(?:is|are)\s+my\b',  # ownership declaration
        r'\bmy\s+(?:document|pdf|file|transcript|resume|report|record|academic)\b',  # "my [document type]"
        r'\b(?:document|pdf|file|transcript|resume|report|record)\s+(?:is|are)\s+mine\b',  # "document is mine"
        r'\bi\s+uploaded\s+my\b',  # "I uploaded my [document]"
        r'\b(?:uploaded|document|file)\s+(?:contains|has|shows)\s+(?:my|the\s+details)\b',  # "document contains my..."
        r'\bmy\s+(?:details|info|information)\b',  # "my details"
    ]
    if has_document and any(re.search(p, query_lower) for p in ownership_patterns):
        st.session_state.document_is_mine = True
        # Route to RAG so it can acknowledge using the document
        return "rag"
    
    # ==========================================================================
    # "MY" / "I" / "ME" QUERIES ROUTING - Personal queries about the user
    # If document is claimed as user's → RAG (check document)
    # Otherwise → Conversation (check what user declared)
    # ==========================================================================
    # Check for self-referential queries
    is_self_query = re.search(r'\b(my|me)\b', query_lower) or re.search(r'\bI\b', query)
    
    if is_self_query:
        # If user has claimed document ownership, check document for personal queries
        if has_document and st.session_state.document_is_mine:
            return "rag"
        # Otherwise, check conversation history for user's declared info
        return "conversation"
    
    # ==========================================================================
    # SECOND: Document queries require EXPLICIT document references
    # ==========================================================================
    # Follow-up indicators - short queries with reference words (generic)
    followup_patterns = [r'\bits\b', r'\bthat\b', r'\bthis\b', r'\bthere\b', r'\bit\b', r'\bthem\b']
    # Generic implicit follow-up patterns (no hardcoded categories)
    implicit_followup_patterns = [
        r'^(?:give me|tell me|show me|list|what about|how about)\s+the\s+\w+',
        r'^(?:and|also|what about)\s+',
        r'^top\s+\d+',
        r'^in\s+[A-Za-z]',  # "In X?"
        r'^for\s+[A-Za-z]',  # "For X?"
        r'^about\s+[A-Za-z]',  # "About X?"
        r'^the\s+\w+\s*\??$',  # "The X?" pattern
    ]
    is_short_query = len(query.split()) <= 8
    has_followup_ref = any(re.search(p, query_lower) for p in followup_patterns)
    has_implicit_followup = any(re.search(p, query_lower) for p in implicit_followup_patterns)
    is_followup = is_short_query and (has_followup_ref or has_implicit_followup)
    
    # ==========================================================================
    # RAG route - ONLY when query EXPLICITLY references the document
    # Document is NOT assumed to be user's data - requires explicit reference
    # ==========================================================================
    if has_document:
        # Check if query explicitly mentions the document
        doc_keywords = ["pdf", "document", "file", "uploaded", "attached"]
        is_doc_query = any(word in query_lower for word in doc_keywords)
        
        # Document-related patterns (generic - no domain-specific terms)
        document_patterns = [
            r'\b(?:this|the|uploaded|attached)\s+(?:pdf|document|file|doc)\b',
            r'\b(?:pdf|document|file|doc)\s+(?:about|contains?|says?|mentions?)\b',
            r'\babout\s+(?:this|the)\s+(?:pdf|document|file|doc)\b',
            r'\bwhat\s+is\s+(?:this|the)\b.*\b(?:pdf|document|file|doc)\b',
            r'\bwhat\s+does\s+(?:this|the)\b.*\b(?:pdf|document|file|doc)\b',
            r'\bsummar(?:y|ize)\b',
            r'\bextract\b',
            r'\bcontents?\s+of\b',
            r'\bwhat\s+is\s+(?:in\s+)?(?:this|the)\s+(?:pdf|document|file)\b',
            r'\bmentioned\s+in\b', r'\bstated\s+in\b', r'\baccording\s+to\b',
            r'\bin\s+the\s+(?:pdf|document|file)\b',
            r'\bas\s+(?:mentioned|stated|shown)\b',
        ]
        matches_doc_pattern = any(re.search(p, query_lower) for p in document_patterns)
        
        # If query explicitly references document → RAG
        if is_doc_query or matches_doc_pattern:
            return "rag"
        
        # =======================================================================
        # DEFAULT TO RAG: When document is uploaded, the chatbot should "know"
        # the document content. Check document first for any query that's NOT
        # clearly external (general knowledge questions).
        # This handles queries about entities mentioned in the document
        # =======================================================================
        
        # Patterns that indicate CLEARLY EXTERNAL queries (general knowledge)
        # These should go to web search instead of checking document
        external_query_patterns = [
            r'\bwho\s+won\b',                     # "who won X" - competition results
            r'\bwho\s+is\s+the\s+\w+\s+of\b',    # "who is the president of" - world leaders
            r'\bcapital\s+of\b',                  # "capital of X" - geography
            r'\bwhere\s+is\s+\w+\s+located\b',   # "where is X located" - locations
            r'\bweather\s+(?:in|for|at)\b',      # weather queries
            r'\blatest\s+news\b',                 # news queries
            r'\bcurrent\s+(?:president|prime\s+minister|ceo|leader)\b',  # world leaders
            r'\bhow\s+(?:tall|old)\s+is\b',      # celebrity facts
            r'\bwhat\s+is\s+the\s+(?:population|area|size)\s+of\b',  # geography facts
            r'\bwhen\s+(?:did|was|is)\s+\w+\s+(?:born|founded|established|released)\b',  # dates
            r'^list\s+(?!the\s+(?:course|subject|grade))',  # "list his X" but NOT "list the courses"
            r'\bgrand\s+slam\b',                  # sports achievements
            r'\bcareer\b',                        # career achievements
            r'\bachievements?\b',                 # achievements
            r'\btitles?\b',                       # titles won
            r'\bawards?\b',                       # awards
            r'\brecords?\b',                      # records
        ]
        is_clearly_external = any(re.search(p, query_lower) for p in external_query_patterns)
        
        # If clearly external, fall through to web search
        if is_clearly_external:
            pass  # Let it fall through to search
        else:
            # =======================================================================
            # SHORT FOLLOW-UPS: Check if this is a follow-up to web search
            # Short follow-up patterns should continue the previous route
            # =======================================================================
            last_route = st.session_state.get("last_route")
            
            # Short follow-up patterns that should continue previous route
            short_followup_patterns = [
                r'^in\s+\d{4}\??$',           # "In [year]?"
                r'^in\s+[A-Za-z]+\??$',       # "In [word]?"
                r'^for\s+[A-Za-z]+\??$',      # "For [word]?"
                r'^the\s+[\w\s]+\??$',        # "The US Open?" or "The 200m?"
                r'^and\s+',                   # "And what about..."
                r'^what\s+about\s+',          # "What about 2024?"
            ]
            is_short_followup = len(query.split()) <= 5 and any(re.search(p, query_lower) for p in short_followup_patterns)
            
            if is_short_followup and last_route == "search":
                # This is a follow-up to web search, continue with search
                return "search"
            
            # Default to RAG for other queries
            return "rag"
    
    # User ASKING about their favorite (e.g., "info about my favorite [thing]")
    if "my favorite" in query_lower or "my favourite" in query_lower:
        # Check if they're asking for info, not declaring
        asking_keywords = ["tell me", "give me", "list", "what", "how many", "about", "show", "info"]
        if any(kw in query_lower for kw in asking_keywords):
            if conversation_history:
                for msg in reversed(conversation_history):
                    if isinstance(msg, HumanMessage) and ("favorite" in msg.content.lower() or "favourite" in msg.content.lower()):
                        return "search_with_context"
        return "conversation"  # Either declaring or we don't know their favorite yet
    
    # Pronoun references about previously mentioned things - MUST go through context resolution
    pronoun_patterns = ["his ", "her ", "their ", "about him", "about her", "about them",
                        " he ", " she ", "does he", "does she", "list his", "list her"]
    # Also check if query starts with these
    starts_with_pronoun = query_lower.startswith(("his ", "her ", "list his", "list her", "how many", "does he", "does she"))
    
    if any(p in query_lower for p in pronoun_patterns) or starts_with_pronoun:
        if conversation_history:
            return "search_with_context"
        return "conversation"
    
    # References to previous topic - GENERIC follow-up detection
    # Any query (regardless of how it starts) that contains unresolved references
    reference_words = ["that", "this", "it", "its", "there", "those", "these", "them"]
    
    # Check if query contains reference words that need context
    has_reference = any(
        f" {word} " in f" {query_lower} " or 
        query_lower.endswith(f" {word}") or
        query_lower.endswith(f" {word}.")
        for word in reference_words
    )
    
    # If short query has unresolved references, it's likely a follow-up
    is_short_query = len(query.split()) <= 15
    
    if has_reference and is_short_query:
        if conversation_history:
            return "search_with_context"
        return "search"  # Fall back to regular search
    
    # Search route - factual/current events questions
    factual_indicators = [
        "who is", "who was", "who won", "who will", "who are", "who has",
        "what is", "what was", "what are", "what does",
        "when did", "when was", "when is", "when will",
        "where is", "where was", "where are",
        "how does", "how did", "how many", "how much", "how old",
        "why did", "why is", "why are",
        "which ", "list ", "tell me about", "define ",
        "current ", "latest ", "president", "capital",
        "population", "founded", "invented", "record", "best",
        "ceo", "company", "country", "city",
        "news", "weather", "price", "stock", "result", "election"
    ]
    if any(ind in query_lower for ind in factual_indicators):
        return "search"
    
    # Year-based questions should search
    if re.search(r'\b20[12]\d\b', query):  # Contains a year (2010-2029)
        return "search"
    
    if "?" in query and len(query.split()) > 3:
        return "search"
    
    if len(query.split()) > 8:
        return "search"
    
    return "conversation"

# =============================================================================
# LLM WITH STREAMING
# =============================================================================
@st.cache_resource
def get_llm(model="llama-3.3-70b-versatile"):
    return ChatGroq(
        model=model,
        temperature=0.7,
        api_key=st.secrets.get("GROQ_API_KEY") or None,
        streaming=True
    )

@st.cache_resource
def get_fallback_llm():
    """Smaller model for when rate limited."""
    return ChatGroq(
        model="llama-3.1-8b-instant",  # Smaller, faster model
        temperature=0.7,
        api_key=st.secrets.get("GROQ_API_KEY") or None,
        streaming=True
    )

def stream_response(messages, system_prompt, use_fallback=False):
    """Stream response from LLM with rate limit handling."""
    llm = get_fallback_llm() if use_fallback else get_llm()
    system_msg = SystemMessage(content=system_prompt)
    
    # Limit context to prevent token overflow
    context_messages = messages[-MAX_CONTEXT_MESSAGES:]
    
    try:
        for chunk in llm.stream([system_msg] + context_messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        error_str = str(e)
        if "rate_limit" in error_str.lower() or "429" in error_str:
            # Rate limited - silently try with smaller model
            if not use_fallback:
                for chunk in stream_response(messages, system_prompt, use_fallback=True):
                    yield chunk
            else:
                yield f"\n\n⚠️ **Rate limit reached.** Please wait a few minutes and try again."
        else:
            yield f"\n\n❌ Error: {error_str[:200]}"

# =============================================================================
# SIDEBAR - CONVERSATION LIST
# =============================================================================
with st.sidebar:
    # Header with logo
    st.markdown('''
    <div class="sidebar-header">
        <h2>🤖 AI Chatbot</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, key="new_chat"):
        new_id = str(uuid.uuid4())
        st.session_state.current_conversation_id = new_id
        st.session_state.conversation_history = []
        st.session_state.vector_store = None
        st.session_state.current_attachment = None
        st.session_state.last_resolved_topic = None  # Clear for new conversation
        st.session_state.last_route = None
        st.rerun()
    
    st.markdown("---")
    
    # Conversation list
    st.markdown("##### 💬 Conversations")
    
    # Sort conversations by most recent
    sorted_convs = sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1].get("updated_at", ""),
        reverse=True
    )
    
    for conv_id, conv_data in sorted_convs:
        title = conv_data.get("title", "New conversation")
        
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                f"💬 {title}", 
                key=f"conv_{conv_id}",
                use_container_width=True,
            ):
                st.session_state.current_conversation_id = conv_id
                st.session_state.conversation_history = dict_to_messages(
                    conv_data.get("messages", [])
                )
                st.session_state.last_resolved_topic = None  # Clear when switching conversations
                st.session_state.last_route = None
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"del_{conv_id}"):
                del st.session_state.conversations[conv_id]
                save_conversations(st.session_state.conversations)
                if st.session_state.current_conversation_id == conv_id:
                    st.session_state.current_conversation_id = None
                    st.session_state.conversation_history = []
                st.rerun()
    
    st.markdown("---")
    
    # File upload section
    st.markdown("##### 📎 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        help="Upload a document to ask questions about it",
        label_visibility="collapsed",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        # Only process if it's a new file (different name or size)
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if st.session_state.get("last_uploaded_file") != file_key:
            st.session_state.last_uploaded_file = file_key
            st.session_state.document_is_mine = False  # Reset ownership for new document
            
            with st.spinner(f"Processing {uploaded_file.name}..."):
                attachment = process_uploaded_file(uploaded_file)
                
            if attachment:
                st.session_state.current_attachment = attachment
                
                if attachment["type"] == "document":
                    with st.spinner("Creating searchable index..."):
                        st.session_state.vector_store = create_vector_store(attachment["data"])
                    st.success(f"✅ {attachment['name']} indexed!")
                else:
                    st.success(f"✅ {attachment['name']} uploaded!")
            else:
                st.error("❌ Failed to process file. Try a different file.")
    
    # Show current document status (persistent indicator)
    if st.session_state.vector_store is not None:
        doc_name = st.session_state.current_attachment.get('name', 'Unknown')
        ownership_status = " (yours)" if st.session_state.document_is_mine else ""
        st.info(f"📚 Document ready: {doc_name}{ownership_status}")
        if st.button("🗑️ Clear Document", use_container_width=True):
            st.session_state.current_attachment = None
            st.session_state.vector_store = None
            st.session_state.last_uploaded_file = None
            st.session_state.document_is_mine = False  # Reset ownership
            st.rerun()
    elif st.session_state.current_attachment:
        # Check if it's an image (images don't need indexing)
        if st.session_state.current_attachment.get('type') == 'image':
            st.success(f"🖼️ Image loaded: {st.session_state.current_attachment['name']}")
        else:
            st.warning(f"⚠️ {st.session_state.current_attachment['name']} (could not index)")
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.current_attachment = None
            st.session_state.vector_store = None
            st.session_state.last_uploaded_file = None
            st.rerun()

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

# Welcome screen when no conversation is active
if not st.session_state.conversation_history:
    st.markdown('''
    <div class="welcome-container">
        <div class="welcome-title">How can I help you today?</div>
        <div class="welcome-subtitle">Ask me anything - I can search the web, analyze documents, or just chat!</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Suggestion cards
    col1, col2 = st.columns(2)
    
    suggestions = [
        ("💡 Explain a concept", "What is machine learning?"),
        ("🌍 Search the web", "What are the latest AI news?"),
        ("📝 Help me write", "Write a professional email"),
        ("🧮 Solve a problem", "How do I sort a list in Python?"),
    ]
    
    for i, (title, prompt) in enumerate(suggestions):
        with col1 if i % 2 == 0 else col2:
            if st.button(title, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

# Display conversation history
for message in st.session_state.conversation_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

def resolve_query_with_context(query: str, messages: list) -> str:
    """Resolve pronouns and references in query using conversation history."""
    query_lower = query.lower().strip()
    original_query = query
    
    # ==========================================================================
    # IMPORTANT: Do NOT resolve document queries as follow-ups
    # "What is this pdf about?" should NOT be resolved with web search context
    # ==========================================================================
    doc_keywords = ["pdf", "document", "file", "uploaded", "attached", "paper"]
    if any(word in query_lower for word in doc_keywords):
        # This is a document-related query - do NOT treat as follow-up
        # Return original query unchanged
        return query
    
    # Find the last mentioned person/entity for pronoun resolution
    last_person = None
    last_favorite = {}
    
    # Use the last RESOLVED topic from session state (handles chained follow-ups)
    # This ensures chained follow-ups use the previously resolved query, not the original
    last_topic = st.session_state.get("last_resolved_topic", None)
    
    # IMPORTANT: Exclude the current query (last message) when looking for context
    # Otherwise we'd extract context from the current query itself!
    messages_to_check = messages[:-1] if messages else []
    
    for msg in messages_to_check:
        content = msg.content
        content_lower = content.lower()
        
        # Track from both HumanMessage AND AIMessage
        
        if isinstance(msg, HumanMessage):
            # Only use message history for last_topic if we don't have a resolved one
            # This prevents overwriting the resolved topic with short follow-up queries
            if last_topic is None and len(content.split()) > 3:
                last_topic = content
            
            # Track favorite things and extract person name
            if "favorite" in content_lower or "favourite" in content_lower:
                match1 = re.search(r'(\w[\w\s]+?)\s+is\s+my\s+(?:favorite|favourite)\s+(\w+)', content, re.I)
                match2 = re.search(r'my\s+(?:favorite|favourite)\s+(\w+)\s+is\s+(.+?)(?:\.|$)', content, re.I)
                
                if match1:
                    person_name = match1.group(1).strip()
                    category = match1.group(2).lower()
                    last_favorite[category] = person_name
                    last_person = person_name
                elif match2:
                    category = match2.group(1).lower()
                    person_name = match2.group(2).strip()
                    last_favorite[category] = person_name
                    last_person = person_name
    
    # ==========================================================================
    # SEPARATE PERSON TRACKING BY CONTEXT
    # Use session state to track document_person and search_person separately
    # This prevents web search entities from overwriting document people
    # ==========================================================================
    
    # Get stored context-specific persons
    document_person = st.session_state.get("document_person")
    
    # Determine current context: is this a document-related query?
    doc_indicators = ["document", "file", "pdf", "transcript", "uploaded", "this", "the file"]
    is_document_query = any(ind in query_lower for ind in doc_indicators)
    
    # Also check if we're following up on document discussion
    last_route = st.session_state.get("last_route")
    if last_route == "rag" and not is_document_query:
        # Short follow-up after document query - likely still about document
        if len(query.split()) <= 6:
            is_document_query = True
    
    # For document queries, prefer document_person
    if is_document_query and document_person:
        last_person = document_person
    
    # If no document_person yet, try to extract from recent RAG responses
    if not document_person:
        for msg in reversed(messages_to_check):
            if isinstance(msg, AIMessage):
                content = msg.content
                # Look for possessive pattern: "X's" where X is multi-word capitalized
                poss_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\'s\b', content)
                if poss_match:
                    potential = poss_match.group(1).strip()
                    if 2 <= len(potential.split()) <= 3:
                        st.session_state.document_person = potential
                        if is_document_query or last_route == "rag":
                            last_person = potential
                        break
    
    # ==========================================================================
    # GENERIC FOLLOW-UP RESOLUTION
    # Works for ANY topic - no hardcoded categories
    # ==========================================================================
    
    is_short_query = len(query.split()) <= 6 and last_topic
    
    if is_short_query:
        last_topic_lower = last_topic.lower()
        # Clean the last topic (remove trailing punctuation issues and normalize to single ?)
        last_topic_clean = re.sub(r'[\?>]+$', '', last_topic.strip()) + '?'
        
        # PATTERN 0: Event/measurement substitution (e.g., "200m?" or "In 200m?" replacing "100m")
        # Detects patterns like "100m", "200m", "400m" in both query and last topic
        event_match = re.search(r'^(?:in\s+)?(?:the\s+)?(\d+m)\s*[\?>]*$', query_lower)
        if event_match:
            new_event = event_match.group(1)  # e.g., "200m"
            # Look for similar event pattern in last topic (e.g., "the 100m")
            old_event_match = re.search(r'\b(?:the\s+)?(\d+m)\b', last_topic_clean, re.I)
            if old_event_match:
                old_event = old_event_match.group(1)  # e.g., "100m"
                result = re.sub(rf'\b{re.escape(old_event)}\b', new_event, last_topic_clean, flags=re.I)
                if result != last_topic_clean:
                    return result
        
        # PATTERN 1: "In X?" / "For X?" / "About X?" / "Of X?"
        # Replaces the corresponding part in the previous question
        prep_match = re.search(r'^(?:in|for|about|of|at|with)\s+(.+?)\s*[\?>]*$', query_lower)
        if prep_match:
            new_value = prep_match.group(1).strip()
            # Remove any trailing punctuation from the new value
            new_value = re.sub(r'[\?>]+$', '', new_value).strip()
            
            # Skip if this was already handled by PATTERN 0 (event substitution)
            if re.match(r'^\d+m$', new_value):
                pass  # Already handled above
            else:
                # Find the same preposition pattern in last topic and replace
                for prep in ['in', 'for', 'about', 'of', 'at', 'with']:
                    # Pattern to find "prep [value]" before the final ? (supports letters AND numbers)
                    pattern = rf'\b{prep}\s+([A-Za-z0-9][A-Za-z0-9\s\-]*?)\s*\?$'
                    match = re.search(pattern, last_topic_clean, re.I)
                    if match:
                        old_value = match.group(1).strip()
                        # Replace the old value with new value (preserve case for numbers)
                        if new_value.isdigit():
                            replacement_value = new_value
                        else:
                            replacement_value = new_value.title()
                        result = re.sub(rf'\b{prep}\s+{re.escape(old_value)}\s*\?$', 
                                       f'{prep} {replacement_value}?', last_topic_clean, flags=re.I)
                        if result != last_topic_clean:
                            return result
        
        # PATTERN 2: "The X?" - substitutes a noun phrase
        # E.g., "The X?" substitutes a noun phrase from the previous question
        the_match = re.search(r'^(?:the|and\s+the|and)\s+(.+?)\s*[\?>]*$', query_lower)
        if the_match:
            new_value = the_match.group(1).strip()
            new_value = re.sub(r'[\?>]+$', '', new_value).strip()
            
            # FIRST: Try to find "the [multi-word phrase] in/at/for/of/from"
            # E.g., "the [noun phrase] in [year]" - replace the noun phrase
            pattern_multiword = r'\bthe\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(in|at|for|of|from)\b'
            match_multiword = re.search(pattern_multiword, last_topic_clean)
            if match_multiword:
                old_value = match_multiword.group(1).strip()
                prep = match_multiword.group(2)
                # Preserve capitalization style of new value
                new_value_titled = ' '.join(w.title() for w in new_value.split())
                result = re.sub(rf'\bthe\s+{re.escape(old_value)}\s+{prep}\b', 
                               f'the {new_value_titled} {prep}', last_topic_clean, count=1, flags=re.I)
                if result != last_topic_clean:
                    return result
            
            # SECOND: Try single word pattern with preposition
            pattern = r'\bthe\s+(\S+)(\s+(?:in|at|for|of|from)\b)'
            match = re.search(pattern, last_topic_clean, re.I)
            if match:
                old_value = match.group(1).strip()
                suffix = match.group(2)  # e.g., " in"
                # Replace "the [old_value]" with "the [new_value]"
                result = re.sub(rf'\bthe\s+{re.escape(old_value)}\b', 
                               f'the {new_value}', last_topic_clean, count=1, flags=re.I)
                if result != last_topic_clean:
                    return result
            
            # THIRD: Try matching "the X?" at the end (no preposition)
            pattern2 = r'\bthe\s+(\S+)\s*\?$'
            match2 = re.search(pattern2, last_topic_clean, re.I)
            if match2:
                old_value = match2.group(1).strip()
                result = re.sub(rf'\bthe\s+{re.escape(old_value)}\s*\?$', 
                               f'the {new_value}?', last_topic_clean, count=1, flags=re.I)
                if result != last_topic_clean:
                    return result
        
        # PATTERN 3: Standalone value - just a noun/number
        # E.g., standalone noun/value as a short follow-up
        standalone_match = re.search(r'^([A-Za-z0-9][A-Za-z0-9\s\-\*]*?)\s*[\?>]*$', query_lower)
        if standalone_match and len(query.split()) <= 3:
            new_value = standalone_match.group(1).strip()
            new_value = re.sub(r'[\?>]+$', '', new_value).strip()
            
            # Try to find a similar pattern to replace in last_topic (generic patterns only)
            patterns_to_try = [
                (r'\bin\s+([A-Z][A-Za-z\s]+?)\s*\?$', f'in {new_value.title()}?'),
                (r'\bof\s+([A-Z][A-Za-z\s]+?)\s*\?$', f'of {new_value.title()}?'),
                (r'\bfor\s+([A-Z][A-Za-z\s]+?)\s*\?$', f'for {new_value.title()}?'),
            ]
            
            for pattern, replacement in patterns_to_try:
                if re.search(pattern, last_topic_clean, re.I):
                    result = re.sub(pattern, replacement, last_topic_clean, flags=re.I)
                    if result != last_topic_clean:
                        return result
    
    # ==========================================================================
    # REFERENCE WORD FOLLOW-UPS
    # Handles: "that one", "this", "it", "there", etc.
    # ==========================================================================
    
    # Reference words that indicate a follow-up needing context
    followup_indicators = [
        r'\bthat\b',           # "that one", "that thing"
        r'\bthis\b',           # "this one", "this"
        r'\bthose\b',          # "those results"
        r'\bthese\b',          # "these items"
        r'\bthere\b',          # "what happened there"
        r'(?<![a-z])it(?![a-z])',  # "when was it", "details about it" - standalone "it" only
        r'(?<![a-z])its(?![a-z])',  # "what is its score" - standalone "its" only
        r'\bthem\b',           # "tell me about them"
        r'\bthe same\b',       # "the same thing"
        r'(?<!\w)about it(?!\w)',  # "tell me more about it" - but not "about its history"
        r'\bmore details\b',   # "give me more details"
        r'\bmore info\b',      # "more info"
        r'\belaborate\b',      # "can you elaborate"
    ]
    
    # Generic implicit follow-up patterns (no hardcoded categories)
    # These indicate SHORT follow-ups that need context, NOT complete questions
    implicit_followup_patterns = [
        r'^(?:what about|how about)\s+',  # "what about 2024?", "how about Python?"
        r'^(?:and|also)\s+',  # "and what about...", "also..."
        r'^top\s+\d+',  # "top 3", "top 5"
        r'^in\s+\d{4}\??$',  # "In [year]?" - year only
        r'^for\s+[A-Za-z]+\??$',  # "For [word]?" - single word
        r'^about\s+[A-Za-z]+\??$',  # "About X?" - single word
    ]
    
    # Check if query contains follow-up indicators
    is_short_query = len(query.split()) <= 15
    has_followup_indicator = any(re.search(p, query_lower) for p in followup_indicators)
    has_implicit_followup = any(re.search(p, query_lower) for p in implicit_followup_patterns)
    
    is_followup = is_short_query and (has_followup_indicator or has_implicit_followup)
    
    # For follow-up questions, add context from last topic
    if is_followup and last_topic:
        # Extract key terms from last topic (generic - works for any topic)
        stopwords = {'who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'was', 'were', 
                    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but',
                    'did', 'does', 'do', 'get', 'got', 'that', 'this', 'can', 'could', 'would',
                    'should', 'will', 'shall', 'may', 'might', 'must', 'have', 'has', 'had',
                    'be', 'been', 'being', 'am', 'your', 'my', 'me', 'you', 'i', 'we', 'they'}
        
        # Get important words from last topic
        topic_words = re.findall(r'\b[A-Za-z0-9]+\b', last_topic)
        
        # Keep words not in stopwords
        key_terms = [w for w in topic_words if w.lower() not in stopwords and len(w) > 1]
        
        # Build context string
        context_str = ' '.join(key_terms[:6])
        
        # Replace reference words with context
        enhanced_query = query
        
        # Replace common reference words
        enhanced_query = re.sub(r'\bits\b', f"{context_str}'s", enhanced_query, flags=re.I)
        enhanced_query = re.sub(r'\bit\b', context_str, enhanced_query, flags=re.I)
        enhanced_query = re.sub(r'\bthere\b', f'in {context_str}', enhanced_query, flags=re.I)
        enhanced_query = re.sub(r'\bin that\b', f'in {context_str}', enhanced_query, flags=re.I)
        enhanced_query = re.sub(r'\bof that\b', f'of {context_str}', enhanced_query, flags=re.I)
        enhanced_query = re.sub(r'\bthem\b', context_str, enhanced_query, flags=re.I)
        enhanced_query = re.sub(r'\b(those|these)\b', context_str, enhanced_query, flags=re.I)
        enhanced_query = re.sub(r'\b(that|this)\s+(\w+)\b', f'{context_str} \\2', enhanced_query, flags=re.I)
        
        # If no replacement happened, append context
        if enhanced_query == query:
            enhanced_query = f"{query} {context_str}"
        
        query = enhanced_query.strip()
    
    # Resolve "my favorite X" ONLY when ASKING about it, not when DECLARING it
    # "List info about my favorite X" → resolves to the declared favorite
    # "Y is my favorite X" → keep as-is (declaration, not a question)
    is_declaration = re.search(r'\bis\s+my\s+(?:favorite|favourite)\b', query_lower)
    if not is_declaration and ("my favorite" in query_lower or "my favourite" in query_lower):
        for category, name in last_favorite.items():
            query = re.sub(rf'my\s+(?:favorite|favourite)\s+{category}', name, query, flags=re.I)
    
    # Resolve pronouns like "his", "her" to last mentioned person
    # IMPORTANT: Choose the RIGHT person based on LAST ROUTE (context)
    has_pronoun = any(p in query_lower for p in ["his ", "her ", "him ", " he ", " she ", "his?", "her?"])
    starts_with_pronoun = query_lower.startswith(("list his", "list her", "his ", "her "))
    
    if has_pronoun or starts_with_pronoun:
        # Determine which person to use based on last_route
        document_person = st.session_state.get("document_person")
        last_route = st.session_state.get("last_route")
        
        # Use document_person if last route was RAG (document context)
        # Use last_person if last route was search (web context)
        person_to_use = None
        if last_route == "rag" and document_person:
            person_to_use = document_person
        elif last_person:
            person_to_use = last_person
        
        if person_to_use:
            query = re.sub(r'\bhis\b', f"{person_to_use}'s", query, flags=re.I)
            query = re.sub(r'\bher\b', f"{person_to_use}'s", query, flags=re.I)
            query = re.sub(r'\bhim\b', person_to_use, query, flags=re.I)
            query = re.sub(r'\bhe\b', person_to_use, query, flags=re.I)
            query = re.sub(r'\bshe\b', person_to_use, query, flags=re.I)
    
    return query

# =============================================================================
# PROCESS PENDING PROMPT OR NEW INPUT
# =============================================================================
prompt = st.session_state.pending_prompt or st.chat_input("Message AI Chatbot...")

if prompt:
    # Clear pending prompt
    st.session_state.pending_prompt = None
    
    # Create new conversation if none exists
    if not st.session_state.current_conversation_id:
        st.session_state.current_conversation_id = str(uuid.uuid4())
    
    # Add user message
    user_message = HumanMessage(content=prompt)
    st.session_state.conversation_history.append(user_message)
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Resolve any references FIRST (before routing)
    # This resolves short follow-ups by incorporating context from previous queries
    resolved_prompt = resolve_query_with_context(prompt, st.session_state.conversation_history)
    
    # Determine route using the RESOLVED prompt for better routing decisions
    has_document = st.session_state.vector_store is not None
    
    # Check if this is a SHORT PATTERN follow-up (In X?, The Y?, standalone word)
    # vs a PRONOUN resolution (his/her → the referenced person)
    # Only use last_route for pattern follow-ups, not pronoun resolutions
    prompt_was_resolved = (resolved_prompt != prompt)
    prompt_lower = prompt.lower().strip()
    
    # Pronoun-based queries should be routed fresh (not use last_route)
    # These are NEW queries about a previously mentioned entity
    pronoun_patterns = [r'\bhis\b', r'\bher\b', r'\bhim\b', r'\bhe\b', r'\bshe\b', r'\btheir\b', r'\bthem\b']
    is_pronoun_query = any(re.search(p, prompt_lower) for p in pronoun_patterns)
    
    # Short pattern follow-ups: "In X?", "The Y?", "Python?", etc.
    # These should use the same route as the previous query
    is_short_pattern_followup = (
        prompt_was_resolved and 
        len(prompt.split()) <= 4 and 
        not is_pronoun_query and
        st.session_state.get("last_route")
    )
    
    if is_short_pattern_followup:
        route = st.session_state.last_route
    else:
        route = determine_route(resolved_prompt, has_document, st.session_state.conversation_history)
    
    # Save the route for follow-up detection
    st.session_state.last_route = route
    
    # Update last_resolved_topic for chained follow-ups
    # This ensures subsequent follow-ups use the resolved query as context
    if len(resolved_prompt.split()) > 3:
        st.session_state.last_resolved_topic = resolved_prompt
    
    # Generate response with streaming
    with st.chat_message("assistant", avatar="🤖"):
        if route == "rag" and st.session_state.vector_store:
            # RAG: Get context and stream response using RESOLVED prompt
            # Show what we're searching for if query was resolved
            if resolved_prompt != prompt:
                st.caption(f"🔍 Searching for: {resolved_prompt}")
            
            context = retrieve_context(st.session_state.vector_store, resolved_prompt)
            rag_prompt = f"""DOCUMENT CONTEXT (use ONLY this information to answer):
{context}

USER QUESTION: {resolved_prompt}

IMPORTANT: Answer using ONLY the exact values from the document context above. 
Quote the specific text that contains the answer. Do not guess or make up values."""
            
            # For RAG, use only the current prompt to avoid confusion from conversation history
            messages = [HumanMessage(content=rag_prompt)]
            response = st.write_stream(stream_response(messages, RAG_PROMPT))
            
        elif route in ["search", "search_with_context"]:
            # Search: Always use resolved_prompt for the actual search
            # This ensures follow-ups search for the full resolved query
            search_query = resolved_prompt
            
            # Show what we're searching for
            if resolved_prompt != prompt:
                st.caption(f"🔍 Searching for: {resolved_prompt}")
            
            with st.spinner("🔍 Searching the web..."):
                search_results = search_internet(search_query)
            
            if search_results:
                current_date = get_current_date()
                # Use resolved_prompt in the question so LLM has full context
                display_question = resolved_prompt if resolved_prompt != prompt else prompt
                search_prompt = f"""TODAY IS {current_date}.

QUESTION: {display_question}

SEARCH RESULTS:
{search_results}

INSTRUCTIONS:
- Answer using the search results above
- If results are clear, give a confident direct answer
- If results seem conflicting or from unreliable sources, mention that info should be verified
- For very recent events, note that details may still be developing
- Be concise - but honest about uncertainty if it exists"""
                
                messages = [HumanMessage(content=search_prompt)]
                response = st.write_stream(stream_response(messages, SEARCH_PROMPT))
            else:
                # No search results - warn user and use LLM's knowledge
                st.caption("⚠️ Web search returned no results - using AI knowledge (may not include recent events)")
                
                fallback_prompt = f"""The user asked: "{prompt}"

I couldn't find current web search results. Please answer based on your training knowledge.
IMPORTANT: If this question is about events after your training cutoff (late 2023), 
you MUST say "I don't have information about events after late 2023" rather than guessing.
Be helpful for questions within your knowledge."""
                
                messages = [HumanMessage(content=fallback_prompt)]
                response = st.write_stream(stream_response(messages, CONVERSATION_PROMPT))
        else:
            # Conversation: Stream response directly with full context
            response = st.write_stream(
                stream_response(st.session_state.conversation_history, CONVERSATION_PROMPT)
            )
    
    # Add AI response to history
    st.session_state.conversation_history.append(AIMessage(content=response))
    
    # Save conversation
    conv_id = st.session_state.current_conversation_id
    st.session_state.conversations[conv_id] = {
        "title": get_conversation_title(messages_to_dict(st.session_state.conversation_history)),
        "messages": messages_to_dict(st.session_state.conversation_history),
        "updated_at": datetime.now().isoformat()
    }
    save_conversations(st.session_state.conversations)
