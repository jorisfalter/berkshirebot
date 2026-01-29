import gradio as gr
from berkshire_rag import setup_qa_chain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup conversational QA chain with hybrid search
print("Initializing conversational QA chain with hybrid search...")
qa_chain = setup_qa_chain()
print("Ready!")

def answer_query(message, history):
    try:
        # Use 'question' key for ConversationalRetrievalChain
        response = qa_chain.invoke({"question": message})
        result = response.get('answer', str(response))

        # Log source documents for debugging
        if 'source_documents' in response:
            print(f"\n=== RETRIEVAL DEBUG ===")
            print(f"Query: '{message}'")
            print(f"Retrieved {len(response['source_documents'])} source documents")
            for i, doc in enumerate(response['source_documents'][:5]):
                score = doc.metadata.get('score', 'N/A')
                source = doc.metadata.get('source', 'Unknown')
                print(f"  {i+1}. {source} (score: {score})")
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
    description="Ask questions about Berkshire Hathaway's annual letters. Uses hybrid search (semantic + keyword) for better retrieval.",
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
