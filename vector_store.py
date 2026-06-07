import os
import uuid
import logging
import time
import re
from typing import List, Dict, Any, Union, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Duck-type langchain's Document structure to ensure compatibility
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def _detect_embedding_provider():
    """Detects which embedding provider to use based on available API keys."""
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    elif os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    else:
        return None

def _call_with_retry(func, *args, **kwargs):
    max_retries = 7
    delay = 5
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str or "limit" in err_str:
                sleep_time = delay
                if "retry_delay" in err_str or "seconds:" in err_str:
                    match = re.search(r'seconds:\s*(\d+)', err_str)
                    if match:
                        sleep_time = int(match.group(1)) + 2
                logger.warning(f"Gemini API rate limit hit. Sleeping for {sleep_time} seconds. (Error: {err_str[:150]})")
                time.sleep(sleep_time)
                delay = min(delay * 2, 60)
            else:
                raise e
    raise RuntimeError("Max retries exceeded for Gemini API call due to rate limiting.")

def _sanitize_metadata(metadata: dict) -> dict:
    import json
    sanitized = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        elif isinstance(v, list):
            try:
                str_list = [str(item) for item in v if item is not None]
                sanitized[k] = ", ".join(str_list) if str_list else ""
            except Exception:
                sanitized[k] = str(v)
        elif v is None:
            sanitized[k] = ""
        else:
            try:
                sanitized[k] = json.dumps(v)
            except Exception:
                sanitized[k] = str(v)
    return sanitized

class RAGVectorStore:
    def __init__(
        self,
        db_type: Optional[str] = None,
        index_name: str = "github_rag",
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        persist_directory: str = "./chroma_db",
        embedding_provider: Optional[str] = None
    ):
        """
        Initializes the RAG Vector Store.
        
        Args:
            db_type: "chroma" (local) or "pinecone". If None, falls back to env DB_TYPE, 
                     or "pinecone" if PINECONE_API_KEY is present, otherwise "chroma".
            index_name: Name of the vector index or collection.
            openai_api_key: OpenAI API key (default: OPENAI_API_KEY env var).
            gemini_api_key: Gemini API key (default: GEMINI_API_KEY env var).
            pinecone_api_key: Pinecone API key (default: PINECONE_API_KEY env var).
            persist_directory: Directory for Chroma persistence (only used for chroma).
            embedding_provider: "openai" or "gemini". Auto-detected if None.
        """
        self.index_name = index_name
        
        # Determine embedding provider
        self.openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        
        if embedding_provider:
            self.embedding_provider = embedding_provider.lower()
        elif self.openai_key:
            self.embedding_provider = "openai"
        elif self.gemini_key:
            self.embedding_provider = "gemini"
        else:
            self.embedding_provider = None
            logger.warning("No API key found (OPENAI_API_KEY or GEMINI_API_KEY). Embedding generation will fail.")
        
        # Set embedding dimension based on provider
        if self.embedding_provider == "openai":
            self.embedding_dim = 1536
        else:
            self.embedding_dim = 768  # Gemini embedding models (embedding-001, text-embedding-004, embedding-2)
            
        logger.info(f"Using '{self.embedding_provider}' as the embedding provider (dimension={self.embedding_dim}).")
        
        # Initialize embedding client
        self.openai_client = None
        self.genai = None
        self.gemini_embedding_model = "models/text-embedding-004"
        
        if self.embedding_provider == "openai":
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.openai_key)
        elif self.embedding_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            self.genai = genai
            try:
                available_models = [m.name for m in self.genai.list_models() if 'embedContent' in m.supported_generation_methods]
                if available_models:
                    if "models/text-embedding-004" in available_models:
                        self.gemini_embedding_model = "models/text-embedding-004"
                    else:
                        self.gemini_embedding_model = available_models[0]
                    logger.info(f"Selected Gemini embedding model: {self.gemini_embedding_model}")
            except Exception as e:
                logger.warning(f"Could not list models, defaulting to models/text-embedding-004: {e}")
        
        # Determine database type
        self.pinecone_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if db_type:
            self.db_type = db_type.lower()
        elif os.environ.get("DB_TYPE"):
            self.db_type = os.environ.get("DB_TYPE").lower()
        elif self.pinecone_key and self.pinecone_key != "your_pinecone_api_key_here":
            self.db_type = "pinecone"
        else:
            self.db_type = "chroma"
            
        logger.info(f"Using '{self.db_type}' as the vector database.")
        
        # Initialize DB client
        if self.db_type == "pinecone":
            self._init_pinecone()
        elif self.db_type == "chroma":
            self._init_chroma(persist_directory)
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}. Use 'chroma' or 'pinecone'.")

    def _init_pinecone(self):
        """Initializes Pinecone client and index."""
        from pinecone import Pinecone, ServerlessSpec
        
        if not self.pinecone_key:
            raise ValueError("PINECONE_API_KEY must be provided or set as environment variable for Pinecone.")
            
        self.pc = Pinecone(api_key=self.pinecone_key)
        
        # Create index if it does not exist
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: '{self.index_name}' (dim={self.embedding_dim})...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Pinecone index '{self.index_name}' initialized.")

    def _init_chroma(self, persist_directory: str):
        """Initializes Chroma local client."""
        import chromadb
        
        # Use PersistentClient to save to disk
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.index_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Chroma collection '{self.index_name}' initialized at '{persist_directory}'.")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts using the configured provider."""
        if not texts:
            return []
            
        logger.info(f"Generating embeddings for {len(texts)} chunks using '{self.embedding_provider}'...")
        
        if self.embedding_provider == "openai":
            return self._get_openai_embeddings(texts)
        elif self.embedding_provider == "gemini":
            return self._get_gemini_embeddings(texts)
        else:
            raise ValueError("No embedding provider configured. Set OPENAI_API_KEY or GEMINI_API_KEY.")

    def _get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates OpenAI text-embedding-3-small embeddings in batches."""
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.openai_client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            embeddings.extend([data.embedding for data in response.data])
            
        return embeddings

    def _get_gemini_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates Gemini embeddings in batches with retry and throttling."""
        embeddings = []
        # Use a smaller batch size to avoid hitting quota limit of 100 requests per minute too quickly
        batch_size = 20  
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            def call_embed():
                return self.genai.embed_content(
                    model=self.gemini_embedding_model,
                    content=batch,
                    task_type="retrieval_document"
                )
                
            result = _call_with_retry(call_embed)
            # embed_content returns a dict with "embedding" key
            # For batch input, it returns a list of embeddings
            if isinstance(result["embedding"][0], list):
                embeddings.extend(result["embedding"])
            else:
                # Single text input returns a flat list
                embeddings.append(result["embedding"])
            # Small sleep between batches to stay under limits
            time.sleep(1)
            
        return embeddings

    def _get_query_embedding(self, text: str) -> List[float]:
        """Generates a single query embedding using the configured provider."""
        if self.embedding_provider == "openai":
            response = self.openai_client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        elif self.embedding_provider == "gemini":
            def call_embed_query():
                return self.genai.embed_content(
                    model=self.gemini_embedding_model,
                    content=text,
                    task_type="retrieval_query"
                )
            result = _call_with_retry(call_embed_query)
            return result["embedding"]
        else:
            raise ValueError("No embedding provider configured.")

    def ingest_documents(self, documents: List[Union[Dict[str, Any], Any]]):
        """
        Chunks, embeds, and upserts a list of documents.
        
        Args:
            documents: A list of dicts (with 'content' and 'metadata' keys) or objects
                       (with 'content'/'page_content' and 'metadata' properties).
        """
        # Convert documents to a standardized list of Document objects
        standardized_docs = []
        for i, doc in enumerate(documents):
            # Check if dict
            if isinstance(doc, dict):
                content = doc.get("content") or doc.get("page_content", "")
                metadata = doc.get("metadata", {})
            else:
                content = getattr(doc, "content", None) or getattr(doc, "page_content", "")
                metadata = getattr(doc, "metadata", {})
                
            if not content:
                logger.warning(f"Document at index {i} has empty content. Skipping.")
                continue
                
            standardized_docs.append(Document(page_content=content, metadata=metadata))

        if not standardized_docs:
            logger.warning("No valid documents to ingest.")
            return

        # 1. Chunk documents using RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(standardized_docs)
        logger.info(f"Split {len(standardized_docs)} documents into {len(chunks)} chunks.")

        # Prepare payload
        ids = [str(uuid.uuid4()) for _ in chunks]
        contents = [chunk.page_content for chunk in chunks]
        # Sanitize metadatas (e.g. Chroma/Pinecone don't support lists/dicts/None)
        metadatas = [_sanitize_metadata(chunk.metadata) for chunk in chunks]

        # 2. Embed each chunk
        embeddings = self._get_embeddings(contents)

        # 3. Upsert into selected index
        logger.info(f"Upserting {len(chunks)} vectors to '{self.db_type}'...")
        if self.db_type == "pinecone":
            vectors = []
            for cid, emb, meta, text in zip(ids, embeddings, metadatas, contents):
                # Ensure the text content is stored in metadata for retrieval
                pinecone_meta = {**meta, "text": text}
                vectors.append((cid, emb, pinecone_meta))
                
            # Upsert in batches of 100
            for i in range(0, len(vectors), 100):
                self.index.upsert(vectors=vectors[i:i+100])
        else:
            # Chroma
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=contents
            )
            
        logger.info("Ingestion completed successfully.")

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant chunks matching the query text.
        
        Args:
            text: Query string.
            top_k: Number of results to return.
            
        Returns:
            A list of dicts: [{"content": str, "metadata": dict, "score": float}]
        """
        if not text:
            return []
            
        logger.info(f"Querying vector store for: '{text}' (top_k={top_k})")
        
        # Get query embedding
        query_embedding = self._get_query_embedding(text)
        
        results = []
        if self.db_type == "pinecone":
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            for match in response.matches:
                meta = match.metadata or {}
                content = meta.pop("text", "")
                results.append({
                    "content": content,
                    "metadata": meta,
                    "score": match.score
                })
        else:
            # Chroma query
            query_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            if query_results and "documents" in query_results and query_results["documents"]:
                docs = query_results["documents"][0]
                metas = query_results["metadatas"][0]
                distances = query_results["distances"][0] if "distances" in query_results else [0.0] * len(docs)
                
                for doc, meta, dist in zip(docs, metas, distances):
                    # Chroma distances are distance metrics (smaller is better). 
                    # We can return 1 - dist as a rough similarity score if distance is cosine distance.
                    score = 1.0 - dist if dist is not None else None
                    results.append({
                        "content": doc,
                        "metadata": meta,
                        "score": score
                    })
                    
        return results

# Module-level helper function for default store access
_default_store = None

def get_default_store() -> RAGVectorStore:
    """Returns a singleton instance of RAGVectorStore."""
    global _default_store
    if _default_store is None:
        _default_store = RAGVectorStore()
    return _default_store

def query(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Global query function using the default vector store."""
    return get_default_store().query(text, top_k=top_k)
