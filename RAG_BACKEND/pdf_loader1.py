import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import numpy as np
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# Paths & Directories
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "Computational_Psychology"
VECTOR_DB = BASE_DIR / "data" / "vector_store"

# --- 1. Document Processing & Splitting ---

def process_pdfs(pdf_directory: str):
    pdf_dir = Path(pdf_directory)
    pdf_documents = []
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in '{pdf_directory}'")

    for pdf_file in pdf_files:
        try:
            loader = PDFPlumberLoader(str(pdf_file))
            documents = loader.load()
            
            for doc in documents:
                doc.metadata["source"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"
            pdf_documents.extend(documents)
            print(f"Successfully loaded: {pdf_file.name}")
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")
    
    print(f"Total PDF pages loaded: {len(pdf_documents)}")
    return pdf_documents


def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    """Splits loaded documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


# --- 2. Embeddings & Vector Store ---

class EmbeddingMan:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)


class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = VECTOR_DB):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def clear_database(self):
        """Deletes the current collection and creates a fresh, empty one."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector database cleared.")

    def count(self) -> int:
        return self.collection.count()

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Document count does not match embedding count.")

        ids, metadatas, documents_text, embeddings_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text
        )


# --- 3. RAG Retrieval & Inference ---

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingMan):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.create_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for doc_id, document, metadata, distance in zip(
                results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0]
            ):
                similarity_score = 1 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': similarity_score
                    })
        return retrieved_docs


def rag_with_sources(
    query: str, 
    retriever: RAGRetriever, 
    client: OpenAI, 
    top_k: int = 5, 
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """Generates an answer and collects detailed source document metadata."""
    results = retriever.retrieve(query, top_k=top_k)
    
    # ✅ FIX: Fixed typo ('chat_historry' -> 'chat_history')
    if chat_history is None:
        chat_history = []

    if not results:
        return {
            "answer": "No relevant context found in uploaded documents.",
            "sources": []
        }

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately and concisely."}
    ]
    
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    context = "\n\n".join([doc['content'] for doc in results])

    prompt = f"""Context:
{context}

Question: {query}
Answer:"""
    
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Pro:novita",
        messages=messages,
    )

    # Extract source metadata cleanly
    sources = []
    for doc in results:
        meta = doc["metadata"]
        sources.append({
            "source_file": meta.get("source", "Unknown"),
            "page": meta.get("page", None),
            "similarity_score": round(doc["similarity_score"], 4),
            "snippet": doc["content"][:250] + ("..." if len(doc["content"]) > 250 else ""),
            "full_metadata": meta
        })

    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }


# --- 4. FastAPI Setup & State Management ---

rag_retriever: RAGRetriever = None
ai_client: OpenAI = None


def initialize_rag(force_reindex: bool = False):
    global rag_retriever, ai_client

    if rag_retriever is not None and ai_client is not None and not force_reindex:
        return rag_retriever, ai_client

    load_dotenv()
    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_API_TOKEN is missing from environment variables.")

    ai_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    embedding_manager = EmbeddingMan()
    vectorstore = VectorStore()

    if vectorstore.count() == 0 or force_reindex:
        print("Vector database empty. Ingesting initial PDFs...")
        all_docs = process_pdfs(str(PDF_DIR))
        chunks = split_docs(all_docs)
        if chunks:
            texts = [doc.page_content for doc in chunks]
            embeddings = embedding_manager.create_embeddings(texts)
            vectorstore.add_documents(chunks, embeddings)
    else:
        print(f"Vector store already loaded with {vectorstore.count()} chunks. Skipping initial ingestion.")

    rag_retriever = RAGRetriever(vectorstore, embedding_manager)
    return rag_retriever, ai_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager — initializes resources once on server start."""
    print("Initializing RAG pipeline for FastAPI...")
    initialize_rag()
    yield
    print("Shutting down RAG service...")


app = FastAPI(title="RAG PDF Search API", lifespan=lifespan)

# --- Pydantic Schemas ---

class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    text: str
    top_k: int
    chat_history: List[ChatMessage] = []

class SourceMetadata(BaseModel):
    source_file: str
    page: Optional[int] = None
    similarity_score: float
    snippet: str
    full_metadata: Dict[str, Any]

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]
    chat_history: List[ChatMessage] = []


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "vector_store_chunks": rag_retriever.vector_store.count() if rag_retriever else 0
    }


@app.post("/ask_with_source", response_model=AskResponse)
def ask_text(request: AskRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Query text must not be empty")

    try:
        hist_dicts = [msg.model_dump() for msg in request.chat_history]
        answer = rag_with_sources(request.text.strip(), rag_retriever, ai_client, request.top_k, hist_dicts)
        return AskResponse(**answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {exc}") from exc


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400, 
                detail=f"File '{file.filename}' is not a PDF. Only PDFs are allowed."
            )
    
    # 1. Clear database
    rag_retriever.vector_store.clear_database()
    
    # 2. Temporary save & process
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        saved_filenames = []
        for file in files:
            temp_path = temp_dir / file.filename
            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_filenames.append(file.filename)
        
        # 3. Process, chunk, embed
        all_docs = process_pdfs(str(temp_dir))
        chunks = split_docs(all_docs)
        
        if chunks:
            texts = [doc.page_content for doc in chunks]
            embeddings = rag_retriever.embedding_manager.create_embeddings(texts)
            rag_retriever.vector_store.add_documents(chunks, embeddings)
        
        return {
            "status": "success", 
            "message": f"Processed {len(files)} PDF(s) and reset memory.", 
            "files_processed": saved_filenames,
            "chunks_loaded": len(chunks) if chunks else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDFs: {str(e)}")
        
    finally:
        # 4. Clean up temporary files
        for temp_file in temp_dir.glob("*"):
            if temp_file.is_file():
                temp_file.unlink()


if __name__ == "__main__":
    uvicorn.run("pdf_loader1:app", host="127.0.0.1", port=8000, reload=True)