import streamlit as st
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Must be called first
st.set_page_config(page_title="Industrial RAG", page_icon="🏭", layout="wide", initial_sidebar_state="auto")

import config
from services.embeddings import EmbeddingService
from services.retrieval import RetrievalService
from services.reranker import RerankerService
from services.generator import GenerationService

# Load Custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize Services in session state to avoid reloading
@st.cache_resource
def init_services(rerank_enabled: bool):
    try:
        embedding_service = EmbeddingService()
        retriever = RetrievalService(embedding_service=embedding_service)
        reranker = RerankerService(enabled=rerank_enabled)
        generator = GenerationService()
        return retriever, reranker, generator, None
    except Exception as e:
        logging.error(f"Error initializing services: {e}")
        return None, None, None, str(e)

# --- UI Layout ---

# Sidebar Settings
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Configure RAG pipeline parameters.")
    
    top_k = st.slider("Retrieval Count", 1, 20, config.TOP_K)
    # Reranking is disabled to keep the app lightweight
    rerank_toggle = False
    rerank_k = top_k
        
    temperature = st.slider("Temperature", 0.0, 1.0, config.TEMPERATURE)
    
    st.divider()
    st.markdown("**System Status**")
    
    if not config.GEMINI_API_KEY:
        st.error("❌ GEMINI_API_KEY missing")
    else:
        st.success("✅ Gemini API Key")
        
    if not config.GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY missing")
    else:
        st.success("✅ Groq API Key")
    
    if os.path.exists(config.QDRANT_PATH):
        st.success(f"✅ DB: `{config.QDRANT_PATH}`")
    else:
        st.error(f"❌ DB Missing: `{config.QDRANT_PATH}`")

# Initialize based on toggle state
retriever, reranker, generator, error_msg = init_services(rerank_toggle)

# Main Chat Interface
st.title("🏭 Earth Tekniks AI Chat Agent")
st.markdown("Ask questions about the Earth Tekniks Services")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Retrieved Sources"):
                for i, source in enumerate(message["sources"]):
                    score = source.get("rerank_score", source.get("score", 0))
                    st.markdown(f"**Source {i+1}** (Score: `{score:.4f}`)")
                    st.info(source.get("text", "No text content"))

# Chat input
if prompt := st.chat_input("Enter your query here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if error_msg:
            st.error(f"Services failed to initialize: {error_msg}")
            st.stop()
            
        try:
            # Contextualize query for better vector search on follow-ups
            search_query = prompt
            user_msgs = [m["content"] for m in st.session_state.messages[:-1] if m["role"] == "user"]
            if user_msgs and len(prompt.split()) < 6:
                search_query = f"{user_msgs[-1]} {prompt}"

            with st.spinner("🔍 Generating query embedding & retrieving..."):
                raw_docs = retriever.retrieve(search_query, top_k=top_k)
                
            if not raw_docs:
                st.warning("No relevant documents found in the database.")
                st.stop()
                
            # Passing retrieved documents directly to generator
            final_docs = raw_docs[:rerank_k]
                
            with st.spinner("💡 Generating answer..."):
                history = st.session_state.messages[:-1]
                answer = generator.generate(prompt, final_docs, chat_history=history, temperature=temperature)
                st.markdown(answer)
                
                with st.expander("View Retrieved Sources"):
                    for i, source in enumerate(final_docs):
                        score = source.get("rerank_score", source.get("score", 0))
                        st.markdown(f"**Source {i+1}** (Score: `{score:.4f}`)")
                        st.info(source.get("text", "No text content"))
                        
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": final_docs
                })
        except Exception as e:
            st.error(f"Error processing query: {e}")
            logging.error(f"Runtime error: {e}", exc_info=True)
