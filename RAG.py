import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from dotenv import load_dotenv

# --- THE BULLETPROOF 2026 IMPORTS ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Handle the Memory import which is often the most unstable
try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    try:
        from langchain_community.memory import ConversationBufferMemory
    except ImportError:
        class ConversationBufferMemory:
            def __init__(self, **kwargs): pass
            def load_memory_variables(self, *args): return {"chat_history": []}
            def save_context(self, *args): pass
            def clear(self): pass

load_dotenv()

# Verify API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")

# --- DATA LOADING FUNCTIONS ---

def load_malaysian_data(directory: str) -> List[Dict]:
    """
    Unified loader for local data. 
    Processes JSON (Super Database & Cases) and PDFs.
    """
    documents = []
    files = Path(directory).glob("*")
    
    # Global splitter settings for statutory continuity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,       
        chunk_overlap=250,     
        separators=["\n\nSection", "\nSection", "\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    for file in files:
        # Handle JSON (Super Database or Case Law)
        if file.suffix == '.json':
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"📂 Processing JSON: {file.name}")
                
                # Support both single dicts and lists of dicts
                if not isinstance(data, list):
                    data = [data]
                
                for entry in data:
                    # Flexible key check for Super Database (content) or Cases (opinion/text)
                    raw_text = entry.get("content") or entry.get("text") or entry.get("opinion") or ""
                    if not raw_text.strip():
                        continue
                        
                    source_name = entry.get("source") or entry.get("title") or file.stem
                    page_info = entry.get("page_number") or "N/A"
                    
                    # Split into searchable chunks
                    chunks = text_splitter.split_text(raw_text)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "id": f"{source_name}_p{page_info}_c{i}_{file.stem}",
                            "text": chunk,
                            "source": f"{source_name} (Page {page_info}, Chunk {i+1})"
                        })
                
            except Exception as e:
                print(f"❌ Error loading JSON {file.name}: {e}")

        # Handle PDFs (Fallback if you ever put them back)
        elif file.suffix == '.pdf':
            try:
                loader = PyPDFLoader(str(file))
                pages = loader.load()
                full_text = "\n\n".join([p.page_content for p in pages])
                chunks = text_splitter.split_text(full_text)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "id": f"pdf_{file.stem}_c{i}",
                        "text": chunk,
                        "source": f"{file.name} (Chunk {i+1})"
                    })
                print(f"✅ Loaded PDF: {file.name}")
            except Exception as e:
                print(f"⚠️ Error loading PDF {file.name}: {e}")

    return documents

# --- EMBEDDING & SEARCH FUNCTIONS ---

def create_embeddings(data: List[Dict], pkl_file="finaldata1.pkl", force_rebuild=False) -> None:
    """
    Generates FAISS embeddings and ensures the pkl file is created.
    """
    if not data:
        print("⚠️ No data found to process. Skipping embedding.")
        return

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(pkl_file) and not force_rebuild:
        with open(pkl_file, "rb") as f:
            combined_data = pickle.load(f)
    else:
        combined_data = []

    existing_ids = {entry["metadata"].get("id") for entry in combined_data if "metadata" in entry}
    
    texts_to_embed = []
    metadatas_to_embed = []

    for entry in data:
        if entry["id"] not in existing_ids:
            texts_to_embed.append(entry["text"])
            metadatas_to_embed.append(entry)

    if texts_to_embed:
        print(f"🧠 Indexing {len(texts_to_embed)} new segments...")
        embeddings = embeddings_model.embed_documents(texts_to_embed)
        
        for i, emb in enumerate(embeddings):
            combined_data.append({
                "embedding": np.array(emb).reshape(1, -1),
                "metadata": metadatas_to_embed[i]
            })

        with open(pkl_file, "wb") as f:
            pickle.dump(combined_data, f)
        print(f"✅ Successfully saved {len(combined_data)} total segments to {pkl_file}.")
    else:
        # Ensure the file exists even if no new data was added
        if not os.path.exists(pkl_file):
            with open(pkl_file, "wb") as f:
                pickle.dump(combined_data, f)
        print("✅ Database is already up to date.")

def load_faiss_index(pkl_file="finaldata1.pkl") -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Loads the search index and metadata into memory.
    """
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Database file {pkl_file} missing. Please run 'Reload Database'.")

    with open(pkl_file, "rb") as f:
        combined_data = pickle.load(f)

    embeddings = np.vstack([entry["embedding"] for entry in combined_data])
    metadata = [entry["metadata"] for entry in combined_data]

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, metadata

# --- THE BRAIN (GROQ PIPELINE) ---

def create_rag_pipeline(index, metadata):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def search_docs(query: str, top_k=7):
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        q_embed = np.array([embed_model.embed_query(query)])
        distances, indices = index.search(q_embed, top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(metadata) and idx != -1: 
                results.append(metadata[idx])
        return results

    def query_rag(query: str) -> str:
        docs = search_docs(query)
        if not docs:
            return "I could not find any relevant Malaysian legal data in the current database."

        context_text = "\n\n".join([
            f"SOURCE: {d['source']}\nCONTENT: {d['text']}" 
            for d in docs
        ])

        system_prompt = f"""
        You are an expert Malaysian Legal Consultant specializing in Copyright and IP Law.
        
        INSTRUCTIONS:
        1. Answer based ONLY on the context provided.
        2. Always cite the specific Section, Act, or Page Number.
        3. Use professional, authoritative Malaysian legal terminology.

        CONTEXT FROM DATABASE:
        {context_text}

        USER QUESTION:
        {query}
        """

        return llm.invoke(system_prompt).content

    return query_rag, memory, {}

def summarize_text(text):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    prompt = f"Perform a forensic summary highlighting key sections and obligations:\n\n{text}"
    return llm.invoke(prompt).content

if __name__ == "__main__":
    raw_data = load_malaysian_data("data")
    create_embeddings(raw_data)
    print("Local Database Engine Online.")
