from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from typing import List, Any, Dict
from langchain.schema import Document
from pydantic import Field

class MetadataRetriever(BaseRetriever):
    """Custom retriever that formats documents with metadata for better citations"""
    
    vectorstore: Any = Field(description="The vector store to retrieve from")
    search_kwargs: Dict = Field(default_factory=lambda: {"k": 8})
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Get the original documents
        docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        
        # Format each document to include metadata in the content
        formatted_docs = []
        for doc in docs:
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
    If you cannot find the information in the provided context, say so explicitly.
    
    Context: {context}
    
    Question: {question}
    
    Answer: Based on the Berkshire Hathaway documents:
    """
    
    # Use our custom retriever
    custom_retriever = MetadataRetriever(
        vectorstore=vectorstore, 
        search_kwargs={"k": 15}
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