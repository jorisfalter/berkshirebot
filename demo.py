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
vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    text_key="text"
)

# Setup QA chain
qa_chain = setup_qa_chain(vectorstore)

def answer_query(message, history):
    response = qa_chain.invoke({"query": message})
    return response.get('result', str(response))

# Create the Gradio interface
demo = gr.ChatInterface(
    answer_query,
    title="Berkshire Hathaway AI Assistant",
    description="Ask questions about Berkshire Hathaway's documents.",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    examples=[
        "When did they write about swimming naked?",
        "What is the meaning of float in Berkshire's reports?",
        "Who is Ajit Jain?"
    ]
)

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=False)  # share=False for local only 