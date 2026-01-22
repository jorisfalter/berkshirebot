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
        
        logger.info(f"Retrieving documents for query: '{query}'")
        logger.info(f"Vectorstore type: {type(self.vectorstore)}")
        
        # Use simple similarity search - most reliable
        try:
            # Search for more documents to ensure we get good results
            docs = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Similarity search returned {len(docs)} documents")
            
            if not docs:
                logger.error("No documents returned from similarity search!")
                # Try with a simpler query
                simple_query = query.split()[0] if query.split() else query
                logger.info(f"Trying simpler query: '{simple_query}'")
                docs = self.vectorstore.similarity_search(simple_query, k=k)
                logger.info(f"Simple query returned {len(docs)} documents")
        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            docs = []
        
        if docs:
            logger.info(f"First doc preview: {docs[0].page_content[:200]}...")
            logger.info(f"First doc metadata: {docs[0].metadata}")
        else:
            logger.error("No documents retrieved! This is a critical error.")
        
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
    # Test the vectorstore first
    try:
        test_results = vectorstore.similarity_search("test", k=1)
        logger.info(f"Vectorstore test successful, returned {len(test_results)} docs")
    except Exception as e:
        logger.error(f"Vectorstore test failed: {e}", exc_info=True)
    
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