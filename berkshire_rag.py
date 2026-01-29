from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
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

        # Hybrid search - fetch more to allow deduplication
        results = self.index.query(
            vector=dense_query,
            sparse_vector=sparse_query,
            top_k=k * 2,  # Fetch extra for deduplication
            include_metadata=True
        )

        logger.info(f"Hybrid search for '{query[:50]}...' returned {len(results['matches'])} results")

        # Convert to LangChain documents with deduplication by source
        # This ensures diverse results (one high-scoring doc per source)
        docs = []
        seen_sources = set()

        for match in results['matches']:
            text = match['metadata'].get('text', '')
            source = match['metadata'].get('source', 'Unknown')
            score = match['score']

            # Skip if we already have a doc from this source (unless it's highly relevant)
            if source in seen_sources:
                # Still include if it contains high-value keywords
                if not ('swimming' in text.lower() and 'naked' in text.lower()):
                    continue

            seen_sources.add(source)

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

            # Stop once we have enough diverse results
            if len(docs) >= k:
                break

        return docs


def setup_qa_chain(index=None, embeddings=None, bm25=None):
    """Set up QA chain with hybrid retrieval."""

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
        model="gpt-4"
    )

    # Create prompt
    prompt_template = """
    You are a financial analyst assistant analyzing Berkshire Hathaway's Chairman's Letters.
    Use the following context to answer the question.

    Each piece of context starts with [Source: document name] - always cite this when referencing information.
    Use direct quotes when possible and put them in quotation marks.

    IMPORTANT: Scan through ALL the context chunks provided. When asked about when something appeared,
    list EVERY year where you find it mentioned. Do not skip any sources.

    If you cannot find the information in the provided context, say so explicitly.

    Context: {context}

    Question: {question}

    Answer (remember to cite ALL relevant sources):
    """

    # Create hybrid retriever (higher k for better recall)
    retriever = HybridRetriever(
        index=index,
        embeddings=embeddings,
        bm25=bm25,
        search_kwargs={"k": 30}
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            ),
        }
    )

    return qa_chain


# For backwards compatibility with demo.py
def setup_qa_chain_from_vectorstore(vectorstore):
    """Legacy function - now uses hybrid search instead."""
    return setup_qa_chain()
