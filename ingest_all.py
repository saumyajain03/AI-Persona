import os
import sys
import json
import argparse
from dotenv import load_dotenv
from vector_store import RAGVectorStore

# Load environment variables from .env if present
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest local GitHub repository JSON documents and optional resume PDF into the RAG vector store.")
    parser.add_argument("--repos-dir", type=str, default="data/repos", help="Directory where repository JSON files are stored")
    parser.add_argument("--resume", type=str, default=None, help="Path to the resume PDF file (optional)")
    parser.add_argument("--db-type", type=str, default=None, help="Vector database type: 'chroma' or 'pinecone'")
    parser.add_argument("--index-name", type=str, default="github_rag", help="Name of the vector collection/index")
    return parser.parse_args()

def parse_resume(pdf_path):
    """Parses text from a PDF resume using pdfplumber."""
    import pdfplumber
    print(f"[Log] Parsing PDF resume: {pdf_path}")
    text_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"[Log] Resume PDF opened. Total pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                print(f"[Log] Page {i+1} parsed. Character count: {len(page_text) if page_text else 0}")
        
        full_text = "\n\n--- Page Break ---\n\n".join(text_content)
        if not full_text.strip():
            raise ValueError("No text could be extracted from the PDF resume.")
        return full_text
    except Exception as e:
        print(f"[Error] Failed to parse PDF resume at '{pdf_path}': {e}")
        raise e

def main():
    args = parse_args()
    
    documents = []
    
    # 1. Process GitHub repository files if the directory exists
    if os.path.exists(args.repos_dir):
        json_files = [f for f in os.listdir(args.repos_dir) if f.endswith(".json")]
        if json_files:
            print(f"[Log] Found {len(json_files)} repository documents in '{args.repos_dir}'. Loading...")
            for filename in json_files:
                filepath = os.path.join(args.repos_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        repo_data = json.load(f)
                        
                    metadata = repo_data.get("metadata", {})
                    readme = repo_data.get("readme_content", "")
                    
                    # Use README as the primary content to chunk, and inject repo details at the top to give context
                    repo_header = f"Repository: {metadata.get('name')}\nDescription: {metadata.get('description')}\nLanguage: {metadata.get('language')}\nStars: {metadata.get('stars')}\nTopics: {', '.join(metadata.get('topics', []))}\n\n"
                    content = repo_header + (readme or "No README content available.")
                    
                    # Tag with source: github
                    metadata["source"] = "github"
                    
                    documents.append({
                        "content": content,
                        "metadata": metadata
                    })
                    
                except Exception as e:
                    print(f"[Warning] Failed to read or parse '{filename}': {e}")
        else:
            print(f"[Warning] No JSON files found in '{args.repos_dir}'. Skipping GitHub ingestion.")
    else:
        print(f"[Info] Directory '{args.repos_dir}' does not exist. Skipping GitHub ingestion.")
        
    # 2. Process Resume PDF if provided
    if args.resume:
        if os.path.exists(args.resume):
            try:
                resume_text = parse_resume(args.resume)
                documents.append({
                    "content": resume_text,
                    "metadata": {
                        "source": "resume",
                        "filename": os.path.basename(args.resume)
                    }
                })
                print(f"[Log] Successfully loaded resume from '{args.resume}'")
            except Exception as e:
                print(f"[Error] Failed to process resume PDF: {e}")
        else:
            print(f"[Error] Resume PDF file not found at '{args.resume}'")
            sys.exit(1)
            
    if not documents:
        print("[Error] No documents (GitHub or resume) were loaded. Ingestion aborted.")
        sys.exit(1)
        
    print(f"[Log] Successfully loaded {len(documents)} total source documents. Initializing vector store...")
    
    try:
        # Initialize store
        store = RAGVectorStore(
            db_type=args.db_type,
            index_name=args.index_name
        )
        
        # Ingest documents
        store.ingest_documents(documents)
        
        print("\n[Log] Ingestion process finished successfully!")
        
    except Exception as e:
        print(f"[Fatal] Error during ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
