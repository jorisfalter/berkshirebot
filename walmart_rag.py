from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Load environment variables
load_dotenv()

def create_rag_database():
    print("Starting RAG database creation...")
    
    # Initialize Pinecone
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
    
    index_name = "walmart-docs"  # Changed name to reflect content
    
    # Delete index if it exists
    if index_name in pc.list_indexes().names():
        print(f"Deleting existing index '{index_name}'...")
        pc.delete_index(index_name)
    
    # Create new index
    print(f"Creating new index '{index_name}' with dimension 1536...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    
    # Get the Pinecone index
    index = pc.Index(index_name)
    
    # Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    
    # Initialize PDF loader
    loader = PyPDFLoader("Walmart 2025 Annual Report.pdf")  # Update with your PDF path
    
    print("Loading PDF document...")
    documents = loader.load()
    
    print(f"Processing document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # Larger chunks to keep more context
        chunk_overlap=200,    # More overlap to avoid missing cross-chunk information
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]  # More granular splitting
    )
    splits = text_splitter.split_documents(documents)
    
    print(f"Created {len(splits)} splits, now storing in Pinecone...")
    
    # Process in smaller batches
    batch_size = 50
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {len(splits)//batch_size + 1}...")
        vectorstore.add_documents(batch)
    
    print("Database creation complete!")
    
    return vectorstore

def query_database(query: str, vectorstore):
    # Perform similarity search
    results = vectorstore.similarity_search(query, k=3)
    return results

def setup_qa_chain(vectorstore):
    # Initialize OpenAI LLM with more precise settings
    llm = ChatOpenAI(
        temperature=0.1,  # Lower temperature for more factual responses
        model="gpt-4"     # Use GPT-4 for better comprehension
    )
    
    # Create a QA chain with specific prompt
    prompt_template = """
    You are a financial analyst assistant analyzing Walmart's annual report. 
    Use the following pieces of context to answer the question.
    For questions about auditors, financial statements, or accounting matters, pay special attention to the audit report and financial statement sections.
    Always include relevant quotes from the source document to support your answer.
    If you find multiple mentions of the same information, include all of them for verification.
    
    Context: {context}
    
    Question: {question}
    
    Answer: Let me help you with that based on the annual report:
    """
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),  # Increase from 5 to 8
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            ),
        }
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