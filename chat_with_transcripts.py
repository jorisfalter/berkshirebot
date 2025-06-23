import os
import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pinecone import Pinecone
from dotenv import load_dotenv

# --- CONFIG ---
INDEX_NAME = "podcast-transcripts"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"

def initialize_agent():
    """Initialize the agent with tools and memory."""
    load_dotenv()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)
    
    # Initialize vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        text_key="text"
    )
    
    # Initialize chat model
    llm = ChatOpenAI(
        model_name=CHAT_MODEL,
        temperature=0.7
    )
    
    # Create tools
    search_tool = Tool(
        name="search_transcripts",
        description="Search through podcast transcripts for relevant information. Use this to find specific details or quotes.",
        func=vectorstore.similarity_search
    )
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant that answers questions about podcast transcripts.
        Follow these steps for each question:
        1. Break down the question into smaller parts
        2. Search for relevant information using the search tool
        3. Verify the information by cross-referencing different sources
        4. Synthesize a comprehensive answer
        5. Always cite your sources
        
        If you're not sure about something, say so and explain what you do know.
        If you need more information, use the search tool again with different queries."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, [search_tool], prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[search_tool],
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

def format_sources(docs):
    """Format the source documents for display."""
    sources = []
    for doc in docs:
        metadata = doc.metadata
        source = f"From: {metadata['title']}"
        if metadata.get('timestamp'):
            source += f" (at {metadata['timestamp']})"
        sources.append(source)
    return "\n".join(sources)

def chat(message, history, agent):
    """Process a chat message and return the response."""
    # Get response from agent
    result = agent.invoke({"input": message})
    
    # Format the response
    response = result["output"]
    
    return response

def create_interface():
    """Create and launch the Gradio interface."""
    agent = initialize_agent()
    
    # Create the chat interface
    interface = gr.ChatInterface(
        fn=lambda message, history: chat(message, history, agent),
        title="Podcast Transcript Chat",
        description="""Ask questions about the podcast transcripts. The AI will:
        1. Break down your question
        2. Search for relevant information
        3. Verify and cross-reference the information
        4. Provide a comprehensive answer with sources""",
        examples=[
            "What are the main topics discussed in the podcast?",
            "Can you summarize the key points from the latest episode?",
            "What was the most interesting insight shared?",
            "How do the guests' opinions differ on this topic?",
        ],
        theme=gr.themes.Soft()
    )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True) 