import os
import glob
import re
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

# --- CONFIG ---
TRANSCRIPTS_DIR = "transcripts"
INDEX_NAME = "podcast-transcripts"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 20  # Number of lines per chunk (adjust as needed)
CHUNK_OVERLAP = 5

# --- HELPERS ---
def parse_transcript(file_path):
    """Parse a transcript file into a list of (timestamp, text) tuples with context."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # First pass: collect all lines with their timestamps
    all_lines = []
    current_timestamp = None
    for line in lines:
        line = line.strip()
        if re.match(r"^\d{1,2}:\d{2}$", line):
            current_timestamp = line
        elif line:
            all_lines.append((current_timestamp, line))
    
    # Second pass: create chunks with context
    chunks = []
    for i in range(0, len(all_lines), CHUNK_SIZE):
        # Get the main chunk
        chunk_lines = all_lines[i:i + CHUNK_SIZE]
        
        # Get context from previous chunk (if exists)
        prev_context = []
        if i > 0:
            prev_start = max(0, i - CHUNK_OVERLAP)
            prev_context = [line[1] for line in all_lines[prev_start:i]]
        
        # Get context from next chunk (if exists)
        next_context = []
        if i + CHUNK_SIZE < len(all_lines):
            next_end = min(len(all_lines), i + CHUNK_SIZE + CHUNK_OVERLAP)
            next_context = [line[1] for line in all_lines[i + CHUNK_SIZE:next_end]]
        
        # Combine everything
        chunk_text = []
        if prev_context:
            chunk_text.append("Previous context: " + " ".join(prev_context))
        chunk_text.extend([line[1] for line in chunk_lines])
        if next_context:
            chunk_text.append("Next context: " + " ".join(next_context))
        
        # Use the first timestamp in the chunk
        timestamp = chunk_lines[0][0] if chunk_lines else None
        chunks.append((timestamp, " ".join(chunk_text)))
    
    return chunks

def get_episode_metadata(filename):
    """Extract episode number and title from filename."""
    base = os.path.basename(filename)
    match = re.match(r"episode_(\d+)", base)
    episode_num = int(match.group(1)) if match else None
    title = base.replace(".txt", "").replace("_", " ").title()
    return episode_num, title

def delete_all_vectors(index):
    """Delete all vectors from the Pinecone index."""
    print("Deleting all existing vectors from the index...")
    index.delete(delete_all=True)
    print("Deletion complete.")

# --- MAIN ---
def main():
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # Create index if not exists
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
    
    # Delete existing vectors before uploading new ones
    delete_all_vectors(index)
    
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        text_key="text"
    )
    files = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt")))
    print(f"Found {len(files)} transcript files.")
    all_docs = []
    for file in tqdm(files, desc="Processing transcripts"):
        episode_num, title = get_episode_metadata(file)
        chunks = parse_transcript(file)
        for i, (timestamp, chunk_text) in enumerate(chunks):
            metadata = {
                "title": title,
                "chunk_index": i,
                "source_file": os.path.basename(file)
            }
            if timestamp is not None:
                metadata["timestamp"] = timestamp
            all_docs.append({"text": chunk_text, "metadata": metadata})
    print(f"Uploading {len(all_docs)} chunks to Pinecone...")
    # Upload in batches
    batch_size = 50
    for i in tqdm(range(0, len(all_docs), batch_size), desc="Uploading"):
        batch = all_docs[i:i+batch_size]
        texts = [doc["text"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        vectorstore.add_texts(texts, metadatas=metadatas)
    print("Upload complete!")

if __name__ == "__main__":
    main() 