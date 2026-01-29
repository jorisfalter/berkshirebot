"""
Re-index Berkshire Hathaway annual letters with hybrid search support.
Uses both dense (semantic) and sparse (BM25 keyword) vectors for better retrieval.
"""

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
import os
import pickle
import re
import io
import time

load_dotenv()

SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'openid'
]


def get_google_credentials():
    """Get or refresh Google credentials."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                        "redirect_uris": ["http://localhost:54049"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token"
                    }
                },
                SCOPES
            )
            creds = flow.run_local_server(port=54049)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds


def extract_year_from_filename(filename: str) -> str:
    """Extract year from various filename formats."""
    match = re.search(r'(19[7-9]\d|20[0-2]\d)', filename)
    if match:
        return match.group(1)
    return None


def standardize_source_name(original_source: str) -> str:
    """Convert any source name to standardized format."""
    year = extract_year_from_filename(original_source)
    if year:
        return f"Berkshire Hathaway Annual Letter {year}"
    return original_source.replace('.pdf', '').replace('pdf', '').strip()


def list_drive_files(service, folder_id):
    """List all files in a Google Drive folder recursively."""
    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token
        ).execute()

        for file in response.get('files', []):
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                files.extend(list_drive_files(service, file['id']))
            else:
                files.append(file)

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return files


def download_and_extract_text(service, file_id, file_name, mime_type):
    """Download a file and extract its text content."""
    try:
        if mime_type == 'application/vnd.google-apps.document':
            content = service.files().export(fileId=file_id, mimeType='text/plain').execute()
            return content.decode('utf-8')

        elif mime_type == 'application/pdf' or file_name.endswith('.pdf'):
            content = service.files().get_media(fileId=file_id).execute()
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text

        elif mime_type == 'text/plain' or file_name.endswith('.txt'):
            content = service.files().get_media(fileId=file_id).execute()
            return content.decode('utf-8')

        else:
            print(f"   Skipping unsupported file type: {file_name} ({mime_type})")
            return None

    except Exception as e:
        print(f"   Error processing {file_name}: {e}")
        return None


def reindex_berkshire():
    """Re-index with hybrid search (dense + sparse vectors)."""

    print("=" * 60)
    print("BERKSHIRE HATHAWAY RE-INDEXING (HYBRID SEARCH)")
    print("=" * 60)

    # Get Google credentials
    print("\n1. Authenticating with Google Drive...")
    creds = get_google_credentials()
    service = build('drive', 'v3', credentials=creds)

    # Initialize Pinecone
    print("\n2. Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "berkshire"

    # Delete existing index
    if index_name in pc.list_indexes().names():
        print(f"   Deleting existing index '{index_name}'...")
        pc.delete_index(index_name)
        while index_name in pc.list_indexes().names():
            time.sleep(2)

    # Create new index with hybrid search support (dotproduct metric required)
    print(f"   Creating new index with hybrid search support...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='dotproduct',  # Required for hybrid search
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(2)

    index = pc.Index(index_name)

    # Initialize embeddings
    print("\n3. Initializing embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # List files from Google Drive
    print("\n4. Listing files in Google Drive...")
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    files = list_drive_files(service, folder_id)
    print(f"   Found {len(files)} files")

    # Download and process documents
    print("\n5. Downloading and processing documents...")
    documents = []

    for file in files:
        file_name = file['name']
        file_id = file['id']
        mime_type = file['mimeType']

        print(f"   Processing: {file_name}")
        text = download_and_extract_text(service, file_id, file_name, mime_type)

        if text and len(text.strip()) > 100:
            standardized = standardize_source_name(file_name)
            doc = Document(
                page_content=text,
                metadata={
                    'source': standardized,
                    'original_filename': file_name,
                }
            )
            documents.append(doc)
            print(f"      -> {standardized} ({len(text):,} chars)")

    print(f"\n   Processed {len(documents)} documents")

    # Split documents
    print("\n6. Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"   Created {len(splits)} chunks")

    # Fit BM25 encoder on corpus
    print("\n7. Training BM25 encoder on corpus...")
    corpus = [doc.page_content for doc in splits]
    bm25 = BM25Encoder()
    bm25.fit(corpus)

    # Save BM25 encoder for later use in retrieval
    bm25.dump("bm25_encoder.json")
    print("   BM25 encoder saved to bm25_encoder.json")

    # Create and upload vectors
    print("\n8. Creating embeddings and uploading to Pinecone...")
    batch_size = 50
    total_batches = (len(splits) + batch_size - 1) // batch_size

    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"   Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        # Get texts and metadata
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        # Create dense embeddings
        dense_vectors = embeddings.embed_documents(texts)

        # Create sparse embeddings (BM25)
        sparse_vectors = bm25.encode_documents(texts)

        # Prepare vectors for upsert
        vectors = []
        for j, (text, dense, sparse, metadata) in enumerate(zip(texts, dense_vectors, sparse_vectors, metadatas)):
            vector_id = f"doc_{i + j}"
            vectors.append({
                "id": vector_id,
                "values": dense,
                "sparse_values": sparse,
                "metadata": {**metadata, "text": text}
            })

        # Upsert to Pinecone
        index.upsert(vectors=vectors)

    # Verify
    print("\n9. Verifying index...")
    time.sleep(3)
    stats = index.describe_index_stats()
    print(f"   Total vectors: {stats.total_vector_count}")

    print("\n" + "=" * 60)
    print("RE-INDEXING COMPLETE (HYBRID SEARCH ENABLED)")
    print("=" * 60)


def test_hybrid_retrieval():
    """Test hybrid retrieval."""
    print("\n\nTesting hybrid retrieval...")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("berkshire")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    bm25 = BM25Encoder().load("bm25_encoder.json")

    query = "swimming naked"
    print(f"\nQuery: '{query}'")

    # Create dense and sparse query vectors
    dense_query = embeddings.embed_query(query)
    sparse_query = bm25.encode_queries([query])[0]

    # Hybrid search with alpha=0.5 (equal weight to semantic and keyword)
    results = index.query(
        vector=dense_query,
        sparse_vector=sparse_query,
        top_k=5,
        include_metadata=True
    )

    print("\nResults:")
    for i, match in enumerate(results['matches']):
        score = match['score']
        source = match['metadata'].get('source', 'Unknown')
        text = match['metadata'].get('text', '')[:200]
        has_quote = 'swimming' in text.lower() and 'naked' in text.lower()
        marker = ' ✓ CONTAINS QUOTE' if has_quote else ''
        print(f"{i+1}. Score: {score:.4f} | {source}{marker}")
        print(f"   {text}...")
        print()


if __name__ == "__main__":
    reindex_berkshire()
    test_hybrid_retrieval()
