import gradio as gr
from walmart_rag import setup_qa_chain
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the RAG system
print("Initializing RAG system...")

# Initialize Pinecone and get existing index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("walmart-docs")

# Create vector store from existing index
vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(),
    text_key="text"
)

# Setup QA chain
qa_chain = setup_qa_chain(vectorstore)

def answer_query(message, history):
    # Use the run method instead of invoke for string output
    response = qa_chain.run(message)
    return response

# Create the Gradio interface
demo = gr.ChatInterface(
    answer_query,
    title="Walmart Annual Report AI Assistant",
    description="Ask questions about Walmart's 2025 Annual Report",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),  # Built-in theme
    examples=[
        "Who is the auditor?",
        "What was the revenue in 2024?",
        "What are the main risks mentioned in the report?",
    ]
)

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=False)  # share=False for local only 