import streamlit as st
import base64
import os
import json
import time
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --- IMPORTS FROM YOUR NEW BACKEND ---
from RAG import (
    load_malaysian_data,
    create_embeddings,
    load_faiss_index,
    create_rag_pipeline,
    summarize_text
)

# Load API Key
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ GROQ_API_KEY not found in .env file!")
    st.stop()

# --- HELPER: IMAGE LOADING ---
def img_to_base64(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

img_path = "LOGO.webp"
img_base64 = img_to_base64(img_path)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Copyright Forensics Tool", 
    page_icon="⚖️", 
    layout="wide"
)

# --- PROFESSIONAL VOID DARK MODE CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;600;700&display=swap');

    /* Global Body Setup - Void Black */
    .stApp {{
        background-color: #0A0A0A;
        font-family: 'Public Sans', sans-serif;
        color: #FFFFFF;
    }}

    /* Minimalist Header - Dark Glass */
    .fixed-title {{
        position: fixed;
        top: 0; left: 0; width: 100%;
        background: rgba(10, 10, 10, 0.95);
        border-bottom: 2px solid #003366; /* Malaysian Royal Blue Accent */
        padding: 15px;
        text-align: center;
        z-index: 999;
        font-size: 22px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: 0.5px;
    }}

    /* Sidebar Styling - Royal Blue */
    [data-testid="stSidebar"] {{
        background-color: #003366 !important; 
        color: white !important;
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}

    /* Sidebar Cards */
    .sidebar-card {{
        background: rgba(255, 255, 255, 0.08);
        border-left: 4px solid #FFCC00; /* Gold Accent */
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }}

    /* Chat Bubbles - High Contrast Dark Mode */
    .chat-container {{
        max-width: 850px;
        margin: auto;
    }}

    .user-bubble {{
        background: #1A1A1A;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 15px;
        border: 1px solid #333333;
        color: #FFFFFF;
        font-size: 15px;
        border-right: 4px solid #CC0000; /* Crimson Accent */
    }}

    .bot-bubble {{
        background: #111111;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 15px;
        border: 1px solid #003366; /* Blue Accent */
        color: #E0E0E0;
        font-size: 15px;
        border-left: 4px solid #FFCC00; /* Gold Accent */
    }}

    /* Professional Buttons - Minimalist Slate */
    .stButton>button {{
        width: 100%;
        background-color: #1A1A1A;
        color: white !important;
        border: 1px solid #444444;
        padding: 10px;
        border-radius: 4px;
        font-weight: 600;
        transition: 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #FFFFFF;
        color: #000000 !important;
        border: 1px solid #FFFFFF;
    }}

    /* Form and Input Customization */
    .stTextInput input {{
        background-color: #1A1A1A !important;
        color: white !important;
        border: 1px solid #333333 !important;
    }}

    [data-testid="stFileUploader"] {{
        background-color: #111111;
        border: 1px dashed #444444;
        color: white;
    }}

    /* Educational Disclaimer at Bottom */
    .disclaimer {{
        font-size: 11px;
        color: #888888;
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #222222;
        line-height: 1.6;
    }}

    .block-container {{ padding-top: 80px !important; }}
</style>
""", unsafe_allow_html=True)

# --- HEADER & SIDEBAR ---
st.markdown('<div class="fixed-title">ASKLAW: MALAYSIA INTELLECTUAL PROPERTY RESEARCH</div>', unsafe_allow_html=True)

with st.sidebar:
    if img_base64:
        st.markdown(f'<div style="text-align:center;"><img src="data:image/webp;base64,{img_base64}" style="width:100px; margin-bottom: 20px;"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card"><b>SYSTEM STATUS</b><br><span style="font-size:12px;">Act 332 (Copyright) Indexed<br>Llama-3.3-70B Active</span></div>', unsafe_allow_html=True)
    
    mode = st.radio("Intelligence Mode:", ["RAG Statutory Search", "Expert Persona"], index=0)
    
    if st.button("🔄 Reload Database"):
        with st.spinner("Processing Malaysian Data..."):
            raw_data = load_malaysian_data("data")
            create_embeddings(raw_data, "finaldata1.pkl")
            st.rerun()

    st.markdown("---")
    # Duplicate disclaimer removed as per request.

# --- ENGINE INITIALIZATION ---
@st.cache_resource
def get_engine():
    if not os.path.exists("finaldata1.pkl"):
        create_embeddings(load_malaysian_data("data"), "finaldata1.pkl")
    index, metadata = load_faiss_index("finaldata1.pkl")
    rag_func, _, _ = create_rag_pipeline(index, metadata)
    return rag_func

def run_fine_tuned_simulation(query):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    prompt = f"SYSTEM: You are a Senior Malaysian IP Lawyer. Use a professional, academic, and authoritative tone. Focus exclusively on Malaysian statutes and case law.\nUSER: {query}"
    return llm.invoke(prompt).content

# --- CHAT DISPLAY LOGIC ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_placeholder = st.container()

with chat_placeholder:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="user-bubble"><b>CLIENT INQUIRY</b><br>{chat["query"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-bubble"><b>ASKLAW COUNSEL RESPONSE</b><br>{chat["response"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- USER INPUT FORM ---
st.write("") 
with st.form(key="query_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_query = st.text_input("Consultation Input:", placeholder="Enter your query regarding Malaysian Copyright Law...")
    with col2:
        submit_button = st.form_submit_button(label="CONSULT")
    
    if mode == "RAG Statutory Search":
        uploaded_file = st.file_uploader("Upload Case JSON for Analysis (Optional)", type="json")
    else:
        uploaded_file = None

# --- PROCESS SUBMISSION ---
if submit_button:
    if user_query.strip() or (mode == "RAG Statutory Search" and uploaded_file):
        with st.spinner("Analyzing Statutes..."):
            try:
                if mode == "RAG Statutory Search":
                    engine = get_engine()
                    if uploaded_file:
                        data = json.load(uploaded_file)
                        text = data[0].get("opinion", "") if isinstance(data, list) else data.get("opinion", "")
                        response = summarize_text(text)
                    else:
                        response = engine(user_query)
                else:
                    response = run_fine_tuned_simulation(user_query)
                
                st.session_state.chat_history.append({"query": user_query or "Document Forensic Analysis", "response": response})
                st.rerun()
            except Exception as e:
                st.error(f"Forensic Analysis Error: {e}")

# --- FORMAL EDUCATIONAL DISCLAIMER ---
st.markdown("""
<div class="disclaimer">
    <b>LEGAL & EDUCATIONAL DISCLAIMER</b><br>
    This <b>Copyright Forensics Tool</b> is developed for <b>educational purposes only</b> as part of an academic project. 
    The responses generated are based on AI processing of the Malaysian Copyright Act 1987 and do not constitute formal legal advice. 
    Users are advised to consult a qualified legal practitioner for official legal matters. 
    This application is <b>Legal Profession Act 1976 Compliant</b> in its capacity as a research tool.<br>
    <i>Powered by Groq LPU & Malaysian Copyright Act 1987.</i>
</div>
""", unsafe_allow_html=True)