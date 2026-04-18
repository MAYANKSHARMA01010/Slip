import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Directory for the vector store
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    """
    Reads the retention_strategies.md, splits it, and saves it into a FAISS vector store.
    """
    kb_path = "knowledge_base/retention_strategies.md"
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Knowledge base file not found at {kb_path}")

    with open(kb_path, 'r', encoding='utf-8') as f:
        markdown_document = f.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    # Use a small, efficient local embedding model
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    print("Creating vector store...")
    db = FAISS.from_documents(md_header_splits, embeddings)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store saved to {DB_FAISS_PATH}")
    return db

def get_vector_db():
    """
    Loads or creates the FAISS vector store.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    if os.path.exists(DB_FAISS_PATH):
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return create_vector_db()

if __name__ == "__main__":
    create_vector_db()
