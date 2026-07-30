Markdown
# 🧠 Full-Stack RAG Bot (FastAPI + Streamlit)

A complete, end-to-end Retrieval-Augmented Generation (RAG) application. This project features a high-performance **FastAPI** backend for document ingestion and vector search, paired with an interactive **Streamlit** frontend for a seamless, conversational user experience.

---

## ✨ Features

### Backend (FastAPI & RAG Pipeline)
*   **Intelligent PDF Ingestion:** Automatically loads and chunks PDFs from a local directory using `PDFPlumber` and LangChain's `RecursiveCharacterTextSplitter`.
*   **Local Vector Database:** Uses **ChromaDB** and `SentenceTransformers` (`all-MiniLM-L6-v2`) for fast, local, and cost-free embeddings.
*   **Smart Startup:** Checks if the vector database is already populated to avoid redundant re-indexing on server restarts.
*   **Context-Aware LLM:** Connects to `deepseek-ai/DeepSeek-V4-Pro:novita` via the Hugging Face Inference API to generate highly accurate answers.
*   **Source Attribution:** Returns exact metadata (source file, page number, similarity score, and snippet) for full transparency.

### Frontend (Streamlit)
*   **Interactive Chat Interface:** A sleek, ChatGPT-like UI built with Streamlit's native chat elements.
*   **Conversational Memory:** Maintains chat history across interactions using Streamlit's `session_state`, allowing for multi-turn conversations.
*   **Expandable Source Citations:** Cleanly hides document sources behind an expandable widget (`st.expander`), keeping the chat UI uncluttered while allowing users to verify facts.
*   **Sidebar Controls & Health Check:** Real-time API status monitoring, vector store chunk counts, and a slider to adjust the number of retrieved sources (`top_k`).

---

## 🛠️ Tech Stack

*   **Frontend:** Streamlit, Requests
*   **Backend:** FastAPI, Uvicorn, Pydantic
*   **LLM Integration:** OpenAI Python SDK (routed via Hugging Face)
*   **Embeddings & Vector Store:** SentenceTransformers, ChromaDB
*   **Document Processing:** LangChain, PDFPlumber

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed and a Hugging Face API token.

### 2. Installation
Clone the repository and install the required dependencies for both the frontend and backend:

```bash
pip install fastapi uvicorn pydantic python-dotenv numpy chromadb openai langchain-community langchain-text-splitters sentence-transformers pdfplumber streamlit requests
3. Project Structure
Ensure your project directory is set up as follows:

Plaintext
├── backend.py                # Your FastAPI application file
├── frontend.py               # Your Streamlit application file
├── .env                      # Environment variables
├── data_pdf_folder/ # Drop your PDF files in this folder
└── data/
    └── vector_store/         # Auto-generated ChromaDB storage
4. Environment Variables
Create a .env file in the root directory and add your Hugging Face token:

Code snippet
HF_API_TOKEN=your_huggingface_api_token_here
🏃‍♂️ Running the Application
You will need to run the backend and frontend simultaneously in two separate terminal windows.

Step 1: Start the Backend (Terminal 1)
Run the FastAPI server. On the first run, it will download the embedding model, ingest your PDFs, and build the ChromaDB vector database.

Bash
uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
You can verify the backend is running by visiting the Swagger UI at http://127.0.0.1:8000/docs.

Step 2: Start the Frontend (Terminal 2)
In a new terminal window, launch the Streamlit chat interface:

Bash
streamlit run frontend.py
This will automatically open the chat interface in your default web browser at http://localhost:8501.

💡 How to Use
Check Status: Look at the sidebar in the Streamlit app to ensure the API is "Online" and that your Vector Store Chunks are greater than 0.

Adjust Retrieval: Use the slider in the sidebar to choose how many document chunks the bot should read before answering (default is 3).

Chat: Type a question related to your PDFs in the chat box.

Verify Sources: Click the "📚 View Sources" dropdown under the AI's response to see exactly which PDFs, pages, and text snippets were used to generate the answer.
