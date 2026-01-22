import gradio as gr
from berkshire_rag import setup_qa_chain
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Pinecone and get existing index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("berkshire")

# Create vector store from existing index
# Must use same embedding model as indexing (text-embedding-3-small)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    text_key="text"
)

# Setup QA chain
qa_chain = setup_qa_chain(vectorstore)

def answer_query(message, history):
    try:
        response = qa_chain.invoke({"query": message})
        result = response.get('result', str(response))
        
        # Log source documents for debugging
        if 'source_documents' in response:
            print(f"\n=== RETRIEVAL DEBUG ===")
            print(f"Query: '{message}'")
            print(f"Retrieved {len(response['source_documents'])} source documents")
            for i, doc in enumerate(response['source_documents'][:5]):
                content_lower = doc.page_content.lower()
                has_swimming = "swimming" in content_lower and "naked" in content_lower
                has_tide = "tide" in content_lower and "goes" in content_lower and "out" in content_lower
                marker = "✓" if (has_swimming or has_tide) else " "
                print(f"{marker} Source {i+1}: {doc.metadata.get('source', 'Unknown')}")
                print(f"   Preview: {doc.page_content[:200]}...")
                if has_swimming or has_tide:
                    print(f"   ✓ CONTAINS QUOTE!")
            print(f"=== END DEBUG ===\n")
        
        return result
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

# Create the Gradio interface
demo = gr.ChatInterface(
    answer_query,
    title="Berkshire Hathaway AI Assistant",
    description="Ask questions about Berkshire Hathaway's documents.",
    examples=[
        "When did they write about swimming naked?",
        "What is the meaning of float in Berkshire's reports?",
        "Who is Ajit Jain?"
    ]
)

# Launch the interface
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Launching Gradio on 0.0.0.0:{port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        quiet=False
    ) 