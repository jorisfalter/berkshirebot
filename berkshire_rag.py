from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from typing import List, Any, Dict
from pydantic import Field
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever using both semantic (dense) and keyword (sparse/BM25) search."""

    index: Any = Field(description="Pinecone index")
    embeddings: Any = Field(description="OpenAI embeddings")
    bm25: Any = Field(description="BM25 encoder")
    search_kwargs: Dict = Field(default_factory=lambda: {"k": 10})
    alpha: float = Field(default=0.5, description="Weight for dense vs sparse (1.0 = all dense, 0.0 = all sparse)")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        k = self.search_kwargs.get("k", 10)

        # Create dense (semantic) query vector
        dense_query = self.embeddings.embed_query(query)

        # Create sparse (BM25/keyword) query vector
        sparse_query = self.bm25.encode_queries([query])[0]

        # Hybrid search
        results = self.index.query(
            vector=dense_query,
            sparse_vector=sparse_query,
            top_k=k,
            include_metadata=True
        )

        logger.info(f"Hybrid search for '{query[:50]}...' returned {len(results['matches'])} results")

        # Convert to LangChain documents
        docs = []
        for match in results['matches']:
            text = match['metadata'].get('text', '')
            source = match['metadata'].get('source', 'Unknown')
            score = match['score']

            # Format with source citation
            formatted_content = f"[Source: {source}]\n{text}"

            doc = Document(
                page_content=formatted_content,
                metadata={**match['metadata'], 'score': score}
            )
            docs.append(doc)

            # Log if we found a notable match
            if 'swimming' in text.lower() and 'naked' in text.lower():
                logger.info(f"  ✓ Found 'swimming naked' in {source} (score: {score:.4f})")

        return docs


def setup_qa_chain(index=None, embeddings=None, bm25=None):
    """Set up conversational QA chain with hybrid retrieval and memory."""

    # Initialize components if not provided
    if index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("berkshire")

    if embeddings is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if bm25 is None:
        bm25 = BM25Encoder().load("bm25_encoder.json")

    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o"
    )

    # Create hybrid retriever
    retriever = HybridRetriever(
        index=index,
        embeddings=embeddings,
        bm25=bm25,
        search_kwargs={"k": 30}
    )

    # Create memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Custom condense question prompt that preserves keywords
    condense_prompt = PromptTemplate(
        template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

IMPORTANT: Preserve ALL specific phrases, names, quotes, and keywords from the original question exactly as written.
Do NOT paraphrase unique phrases like "swimming naked" into generic terms like "the concept of".
Keep the exact wording of any quoted or distinctive phrases.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question (preserve exact phrases):""",
        input_variables=["chat_history", "question"]
    )

    # Create conversational chain with memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template="""You are a financial analyst assistant analyzing Berkshire Hathaway's Chairman's Letters.
Use the following context and chat history to answer the question.

Each piece of context starts with [Source: document name] - always cite this when referencing information.

IMPORTANT RULES:
- Only cite information that is EXPLICITLY stated in the context
- Do NOT make up or infer quotes - only use exact text from the context
- If asked about a specific phrase, only cite sources where that EXACT phrase appears
- Always include the exact quote from each source in quotation marks
- You can refer to previous answers in the conversation

If you cannot find the information in the provided context, say so explicitly.

Context: {context}

Question: {question}

Answer (include exact quotes from each source):""",
                input_variables=["context", "question"]
            ),
        }
    )

    return qa_chain


# For backwards compatibility with demo.py
def setup_qa_chain_from_vectorstore(vectorstore):
    """Legacy function - now uses hybrid search instead."""
    return setup_qa_chain()
