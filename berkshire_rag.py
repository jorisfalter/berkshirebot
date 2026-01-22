from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Any, Dict
from pydantic import Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataRetriever(BaseRetriever):
    """Custom retriever that formats documents with metadata for better citations"""
    
    vectorstore: Any = Field(description="The vector store to retrieve from")
    search_kwargs: Dict = Field(default_factory=lambda: {"k": 8})
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        k = self.search_kwargs.get("k", 8)
        
        logger.info(f"Retrieving documents for query: '{query[:100]}...'")
        
        # Use MMR for better diversity, but fallback to regular search if it fails
        try:
            # MMR: fetch more candidates, then select diverse subset
            fetch_k = min(k * 5, 100)  # Fetch 5x more candidates
            docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=0.7  # Favor relevance (0.0 = max diversity, 1.0 = max relevance)
            )
            logger.info(f"MMR search returned {len(docs)} docs")
        except Exception as e:
            logger.warning(f"MMR search failed: {e}, falling back to similarity search")
            # Fallback to regular similarity search
            try:
                docs = self.vectorstore.similarity_search(query, k=k * 2)
                # Take top k by relevance (they're already sorted)
                docs = docs[:k]
                logger.info(f"Similarity search returned {len(docs)} docs")
            except Exception as e2:
                logger.error(f"Similarity search also failed: {e2}")
                docs = []
        
        if docs:
            logger.info(f"First doc preview: {docs[0].page_content[:150]}...")
        else:
            logger.warning("No documents retrieved!")
        
        # Collect all unique docs (in case of duplicates)
        seen = set()
        unique_docs = []
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        
        logger.info(f"Returning {len(unique_docs)} unique documents")
        
        # Format each document to include metadata in the content
        formatted_docs = []
        for doc in unique_docs[:k]:
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown Source')
            lines_from = metadata.get('loc.lines.from', '')
            lines_to = metadata.get('loc.lines.to', '')
            
            # Create formatted content with metadata - prioritize source over title
            formatted_content = f"[Source: {source}"
            if lines_from and lines_to:
                formatted_content += f", Lines: {lines_from}-{lines_to}"
            formatted_content += f"]\n{doc.page_content}"
            
            # Create new document with formatted content
            formatted_doc = Document(
                page_content=formatted_content,
                metadata=metadata
            )
            formatted_docs.append(formatted_doc)
        
        return formatted_docs

def setup_qa_chain(vectorstore):
    # Initialize OpenAI LLM with more precise settings
    llm = ChatOpenAI(
        temperature=0.1,  # Lower temperature for more factual responses
        model="gpt-4"     # Use GPT-4 for better comprehension
    )
    
    # Create a QA chain with specific prompt for Berkshire Hathaway
    prompt_template = """
    You are a financial analyst assistant analyzing Berkshire Hathaway's Chairman's Letters and documents. 
    Use the following pieces of context to answer the question.
    
    Each piece of context is formatted with source metadata at the beginning in brackets, followed by the content.
    When referencing information, always cite the specific source and line numbers from the brackets.
    
    Format your citations like: "According to Chairman's Letter - 1989.pdf (lines 777-790), Buffett states..."
    Use direct quotes when possible and put them in quotation marks.
    If information comes from multiple sources, mention all relevant sources.
    
    For questions about auditors, financial statements, or accounting matters, pay special attention to the audit report and financial statement sections.
    
    IMPORTANT: Look for related concepts, metaphors, and famous quotes in the context. Financial metaphors and aphorisms should be connected to their full context.
    
    Search thoroughly through all provided context before concluding information is not available.
    If you cannot find the information in the provided context, say so explicitly.
    
    Context: {context}
    
    Question: {question}
    
    Answer: Based on the Berkshire Hathaway documents:
    """
    
    # Use our custom retriever with higher k for better recall
    # MMR will fetch 5x this amount as candidates and select diverse subset
    custom_retriever = MetadataRetriever(
        vectorstore=vectorstore, 
        search_kwargs={"k": 30}  # Retrieve more documents for better coverage
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=custom_retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            ),
        }
    )
    return qa_chain 