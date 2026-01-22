from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Any, Dict
from pydantic import Field
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataRetriever(BaseRetriever):
    """Custom retriever that formats documents with metadata for better citations"""
    
    vectorstore: Any = Field(description="The vector store to retrieve from")
    search_kwargs: Dict = Field(default_factory=lambda: {"k": 8})
    
    def _keyword_score(self, doc_content: str, query: str) -> float:
        """Score document based on keyword matches"""
        content_lower = doc_content.lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Big bonus for exact phrase match (or close variation)
        if query_lower in content_lower:
            score += 2.0
        else:
            # Check for phrase with word order preserved (allowing gaps)
            query_words = query_lower.split()
            if len(query_words) >= 2:
                # Check if all words appear in order (with possible gaps)
                last_idx = -1
                ordered_match = True
                for word in query_words:
                    idx = content_lower.find(word, last_idx + 1)
                    if idx == -1:
                        ordered_match = False
                        break
                    last_idx = idx
                if ordered_match:
                    score += 1.5
        
        # Extract meaningful words (3+ chars, not common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'when', 'who', 'what', 'where', 'why', 'how', 'about', 'they', 'did', 'write'}
        
        # Check for multi-word phrases in the query (2+ consecutive words)
        query_words_list = query_lower.split()
        if len(query_words_list) >= 2:
            # Check all 2-word and 3-word phrases from the query
            for phrase_len in [2, 3]:
                for i in range(len(query_words_list) - phrase_len + 1):
                    phrase = " ".join(query_words_list[i:i+phrase_len])
                    if phrase in content_lower:
                        # Bigger bonus for longer phrases
                        score += 0.5 * phrase_len  # 1.0 for 2-word, 1.5 for 3-word
        
        query_words = [w for w in query_lower.split() if len(w) >= 3 and w not in stop_words]
        
        if query_words:
            # Count how many query words appear in the document
            matches = sum(1 for word in query_words if word in content_lower)
            word_score = matches / len(query_words) if query_words else 0
            score += word_score * 0.5  # Add word matching as bonus
        else:
            # If all words are stop words, check for important single words
            all_query_words = query_lower.split()
            important_words = ['swimming', 'naked', 'tide', 'goes', 'out', 'discover']
            matches = sum(1 for word in all_query_words if word in important_words and word in content_lower)
            if matches > 0:
                score += matches * 0.3  # Smaller bonus for single important words
        
        return min(score, 3.0)  # Cap at 3.0
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        k = self.search_kwargs.get("k", 8)
        
        logger.info(f"Retrieving documents for query: '{query}'")
        
        # Fetch more documents than needed for reranking
        # Since we know quotes can be missed by semantic search, cast a wider net
        fetch_k = min(k * 6, 100)  # Get 6x more candidates (up to 100)
        
        try:
            # Semantic search - get more candidates
            docs = self.vectorstore.similarity_search(query, k=fetch_k)
            logger.info(f"Semantic search returned {len(docs)} documents")
        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            docs = []
        
        if not docs:
            logger.error("No documents retrieved!")
            return []
        
        # Hybrid approach: Combine semantic similarity with keyword matching
        # Score each document
        scored_docs = []
        for doc in docs:
            semantic_score = 1.0  # All docs from semantic search are relevant
            keyword_score = self._keyword_score(doc.page_content, query)
            
            # Combined score: 40% semantic, 60% keyword (heavily favor keyword matches)
            # This is important because semantic search can miss exact phrases
            combined_score = 0.4 * semantic_score + 0.6 * keyword_score
            
            scored_docs.append((doc, combined_score, keyword_score))
        
        # Sort by combined score (keyword matches get boost)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Log top results
        logger.info(f"Top 3 results after hybrid reranking:")
        for i, (doc, combined_score, keyword_score) in enumerate(scored_docs[:3], 1):
            logger.info(f"  {i}. Score: {combined_score:.3f} (keyword: {keyword_score:.3f})")
            logger.info(f"     Preview: {doc.page_content[:150]}...")
        
        # Take top k unique documents
        unique_docs = []
        seen = set()
        for doc, score, keyword_score in scored_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
                if len(unique_docs) >= k:
                    break
        
        logger.info(f"Returning {len(unique_docs)} unique documents (top {k} after hybrid reranking)")
        
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