from langchain_community.document_loaders import GoogleDriveLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

# Google Drive API scopes
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',  # For reading Drive files
    'https://www.googleapis.com/auth/userinfo.email',  # For getting user email
    'https://www.googleapis.com/auth/userinfo.profile',  # For basic profile info
    'openid'  # Add this scope
]

def get_google_credentials():
    creds = None
    # Check if we have stored credentials
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials don't exist or are invalid, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Create flow using environment variables instead of credentials.json
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
        
        # Save credentials for future use
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def create_rag_database():
    print("Starting RAG database creation...")
    # Get Google Drive credentials
    creds = get_google_credentials()
    
    # Initialize Pinecone with new syntax
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
    
    index_name = "googledrive-docs"
    
    # Delete index if it exists
    if index_name in pc.list_indexes().names():
        print(f"Deleting existing index '{index_name}'...")
        pc.delete_index(index_name)
    
    # Create new index with correct dimension
    print(f"Creating new index '{index_name}' with dimension 1536...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # Changed to match OpenAI's embedding dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    
    # Get the Pinecone index
    index = pc.Index(index_name)
    
    # Create embeddings using OpenAI instead of Llama
    embeddings = OpenAIEmbeddings()  # Uses text-embedding-ada-002 model
    
    # Create vector store using the new PineconeVectorStore
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    
    # Create Google Drive service
    service = build('drive', 'v3', credentials=creds)
    
    # Initialize Google Drive loader with service
    loader = GoogleDriveLoader(
        folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
        recursive=True,
        service=service  # Pass the service instead of credentials
    )
    
    print("Loading documents from Google Drive...")
    documents = loader.load()
    
    print(f"Processing {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    print("Creating embeddings and storing in Pinecone...")
    vectorstore.add_documents(splits)
    
    print("Database creation complete!")
    
    return vectorstore

def query_database(query: str, vectorstore):
    # Perform similarity search
    results = vectorstore.similarity_search(query, k=3)
    return results

def setup_qa_chain(vectorstore):
    # Initialize OpenAI LLM instead of Llama
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo"  # or "gpt-4" for better but more expensive results
    )
    
    # Create a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

def main():
    # Create or load the vector database
    vectorstore = create_rag_database()
    
    # Setup the QA chain with local LLM
    qa_chain = setup_qa_chain(vectorstore)
    
    # Example query
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        # Get AI-generated answer
        answer = qa_chain.run(query)
        print("\nAI Answer:")
        print(answer)
        print("\n" + "-" * 50)
        
        # Show unique source document titles
        results = query_database(query, vectorstore)
        print("\nSource documents:")
        seen_sources = set()  # Track unique sources
        
        for doc in results:
            source = doc.metadata.get('source', 'Unknown source')
            title = doc.metadata.get('title', source.split('/')[-1])  # Get title or last part of URL
            
            # Only show each source once
            if source not in seen_sources:
                seen_sources.add(source)
                print(f"\nTitle: {title}")
                print("-" * 50)

if __name__ == "__main__":
    main() 