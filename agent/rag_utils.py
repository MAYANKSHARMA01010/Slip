import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

# We'll store our FAISS vector index here for quick retrieval during agent sessions.
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    """
    Parses our retention playbooks and embeds them into a searchable FAISS database.
    """
    kb_path = "knowledge_base/retention_strategies.md"
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"We couldn't find the playbook at {kb_path}. Please check the folder.")

    with open(kb_path, 'r', encoding='utf-8') as f:
        markdown_document = f.read()

    # We split the markdown by headers so the agent gets contextually relevant chunks.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    # Initializing a lightweight embedding model to convert text into searchable vectors.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Building the vector store and saving it locally to avoid re-embedding every time.
    db = FAISS.from_documents(md_header_splits, embeddings)
    
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    return db

def get_vector_db():
    """
    Retrieves the existing vector store or creates a new one if it's missing.
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
    # Allows for manual refresh of the knowledge base from the terminal.
    create_vector_db()
