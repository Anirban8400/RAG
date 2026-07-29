import streamlit as st
import requests

# Point this to your FastAPI server URL
API_URL = "http://127.0.0.1:8000"

# --- Page Configuration ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("RAG Chatbot 🤖")

# --- Sidebar: Health Check & Settings ---
with st.sidebar:
    st.header("⚙️ System Status")
    
    # Hit the /health endpoint
    try:
        health_res = requests.get(f"{API_URL}/health")
        if health_res.status_code == 200:
            data = health_res.json()
            st.success("API Status: Online")
            st.metric("Vector Store Chunks", data.get("vector_store_chunks", 0))
        else:
            st.error("API Status: Error")
    except requests.exceptions.ConnectionError:
        st.error("API Status: Offline (Is FastAPI running?)")
        
    st.divider()
    st.header("🔍 Retrieval Settings")
    top_k = st.slider("Number of sources to retrieve (top_k)", min_value=1, max_value=10, value=3)

# --- Chat History Initialization ---
# Streamlit re-runs the whole script on every interaction, so we use session_state to save memory.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Existing Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if the AI provided them in a previous turn
        if msg.get("sources"):
            with st.expander("📚 View Sources"):
                for idx, source in enumerate(msg["sources"]):
                    st.markdown(f"**{idx+1}. {source.get('source_file', 'Unknown')} (Page {source.get('page', 'N/A')})**")
                    st.caption(f"Similarity Score: {source.get('similarity_score', 0.0)}")
                    st.text(source.get('snippet', ''))

# --- Chat Input & API Call ---
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # 1. Display user message in UI and add to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            
            # Prepare payload for backend
            # Note: We slice [:-1] so we only send PAST history, not the current prompt twice
            history_payload = [
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.messages[:-1]
            ]
            
            payload = {
                "text": prompt,
                "top_k": top_k,
                "chat_history": history_payload
            }
            
            try:
                # Hit the /ask_with_source endpoint
                response = requests.post(f"{API_URL}/ask_with_source", json=payload)
                
                if response.status_code == 200:
                    res_data = response.json()
                    answer = res_data.get("answer", "")
                    sources = res_data.get("sources", [])
                    
                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Sources inside an expander widget
                    if sources:
                        with st.expander("📚 View Sources"):
                            for idx, source in enumerate(sources):
                                st.markdown(f"**{idx+1}. {source.get('source_file', 'Unknown')} (Page {source.get('page', 'N/A')})**")
                                st.caption(f"Similarity Score: {source.get('similarity_score', 0.0)}")
                                st.text(source.get('snippet', ''))
                    
                    # Append AI response and sources to session state for future turns
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                else:
                    error_detail = response.json().get('detail', 'Unknown Error')
                    st.error(f"Backend Error: {error_detail}")
            
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Please ensure your FastAPI server is running.")