from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from berkshire_rag import setup_qa_chain
from dotenv import load_dotenv
import os
import re

def find_mentioned_sources(answer, all_sources):
    """Find sources that are mentioned in the AI's answer"""
    mentioned_sources = []
    answer_lower = answer.lower()
    
    for doc in all_sources:
        source = doc.metadata.get('source', '')
        
        # Check if source filename or year is mentioned in the answer
        if source:
            source_name = source.lower()
            # Remove .pdf extension for matching
            source_name_clean = source_name.replace('.pdf', '')
            
            if source_name_clean in answer_lower or source_name in answer_lower:
                mentioned_sources.append(doc)
                continue
        
        # Also check for year patterns (like "2007") in the answer
        lines_from = doc.metadata.get('loc.lines.from', '')
        lines_to = doc.metadata.get('loc.lines.to', '')
        
        # If the answer mentions specific line numbers that match this doc
        if lines_from and lines_to:
            if f"lines {int(float(lines_from))}" in answer_lower or f"lines {lines_from}" in answer_lower:
                mentioned_sources.append(doc)
    
    return mentioned_sources

def highlight_quote_in_content(content, query):
    """Try to find and highlight relevant quotes in the content"""
    # Remove the metadata header from content for searching
    if content.startswith('[Source:'):
        # Find the end of the metadata section
        metadata_end = content.find(']\n')
        if metadata_end != -1:
            clean_content = content[metadata_end + 2:]
        else:
            clean_content = content
    else:
        clean_content = content
    
    # Look for key phrases from the query in the content
    query_words = query.lower().split()
    
    # Try to find sentences that contain multiple query words
    sentences = re.split(r'[.!?]+', clean_content)
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        word_matches = sum(1 for word in query_words if word in sentence_lower)
        if word_matches >= 2:  # If at least 2 query words are found
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        return clean_content[:400] + "\n\n*** RELEVANT QUOTES ***\n" + "\n".join(relevant_sentences[:2])
    else:
        return clean_content[:1000]

def print_top_similar_chunks(vectorstore, phrase):
    print(f"\nTop 10 most similar chunks for: '{phrase}'\n" + "="*50)
    results = vectorstore.similarity_search(phrase, k=10)
    for i, doc in enumerate(results, 1):
        metadata = doc.metadata
        title = metadata.get('pdf.info.Title', 'Unknown Document')
        source = metadata.get('source', 'Unknown Source')
        lines_from = metadata.get('loc.lines.from', '')
        lines_to = metadata.get('loc.lines.to', '')
        print(f"\nResult {i}:")
        print(f"Source: {source}")
        if title and title != 'Unknown Document' and title not in source:
            print(f"Document Title: {title}")
        if lines_from and lines_to:
            print(f"Lines: {lines_from}-{lines_to}")
        print(f"Content: {doc.page_content[:1000]}")
        print("-" * 40)

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Pinecone
    print("Initializing connection to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("berkshire")
    
    # Create vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        text_key="text"
    )
    
    # Setup QA chain
    qa_chain = setup_qa_chain(vectorstore)
    
    print("\nBerkshire Hathaway AI Assistant")
    print("Type 'quit' or 'exit' to end the session")
    print("-" * 50)
    
    while True:
        # Get user input
        query = input("\nYour question: ").strip()
        
        # Special command: !search <phrase>
        if query.startswith('!search '):
            phrase = query[len('!search '):].strip()
            print_top_similar_chunks(vectorstore, phrase)
            continue
        
        # Check for exit command
        if query.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not query:
            continue
            
        try:
            # Get response from the QA chain
            response = qa_chain.invoke({"query": query})
            
            # Extract just the result from the response
            answer = response.get('result', str(response))
            print("\nAnswer:", answer)
            
            # Show source documents if they were returned
            if 'source_documents' in response:
                print("\n" + "="*50)
                print("SOURCE DOCUMENTS:")
                print("="*50)
                
                # Show only top 3 most relevant sources
                source_docs = response['source_documents']
                for i, doc in enumerate(source_docs[:3], 1):
                    print(f"\nSource {i}:")
                    
                    # Extract key metadata for better display
                    metadata = doc.metadata
                    title = metadata.get('pdf.info.Title', 'Unknown Document')
                    source = metadata.get('source', 'Unknown Source')
                    lines_from = metadata.get('loc.lines.from', '')
                    lines_to = metadata.get('loc.lines.to', '')
                    
                    # Display formatted source info - prioritize source over title
                    print(f"Source: {source}")
                    if title and title != 'Unknown Document' and title not in source:
                        print(f"Document Title: {title}")
                    if lines_from and lines_to:
                        print(f"Lines: {lines_from}-{lines_to}")
                    
                    print(f"Content: {highlight_quote_in_content(doc.page_content, query)}")
                    print("-" * 40)
                # Find mentioned sources
                mentioned_sources = find_mentioned_sources(answer, source_docs)
                if mentioned_sources:
                    print("\n" + "="*50)
                    print("RELATED SOURCES MENTIONED IN ANSWER:")
                    print("="*50)
                    for i, doc in enumerate(mentioned_sources, 1):
                        print(f"\nSource {i}:")
                        metadata = doc.metadata
                        title = metadata.get('pdf.info.Title', 'Unknown Document')
                        source = metadata.get('source', 'Unknown Source')
                        lines_from = metadata.get('loc.lines.from', '')
                        lines_to = metadata.get('loc.lines.to', '')
                        print(f"Source: {source}")
                        if title and title != 'Unknown Document' and title not in source:
                            print(f"Document Title: {title}")
                        if lines_from and lines_to:
                            print(f"Lines: {lines_from}-{lines_to}")
                        print(f"Content: {highlight_quote_in_content(doc.page_content, query)}")
                        print("-" * 40)
            else:
                # Fallback: Get source documents for this query
                print("\n" + "="*50)
                print("RELATED SOURCES:")
                print("="*50)
                
                # Perform similarity search to show sources
                source_docs = vectorstore.similarity_search(query, k=3)
                for i, doc in enumerate(source_docs, 1):
                    print(f"\nSource {i}:")
                    metadata = doc.metadata
                    title = metadata.get('pdf.info.Title', 'Unknown Document')
                    source = metadata.get('source', 'Unknown Source')
                    lines_from = metadata.get('loc.lines.from', '')
                    lines_to = metadata.get('loc.lines.to', '')
                    print(f"Source: {source}")
                    if title and title != 'Unknown Document' and title not in source:
                        print(f"Document Title: {title}")
                    if lines_from and lines_to:
                        print(f"Lines: {lines_from}-{lines_to}")
                    print(f"Content: {highlight_quote_in_content(doc.page_content, query)}")
                    print("-" * 30)
                # Find mentioned sources in fallback case
                mentioned_sources = find_mentioned_sources(answer, source_docs)
                if mentioned_sources:
                    print("\n" + "="*50)
                    print("RELATED SOURCES MENTIONED IN ANSWER:")
                    print("="*50)
                    for i, doc in enumerate(mentioned_sources, 1):
                        print(f"\nSource {i}:")
                        metadata = doc.metadata
                        title = metadata.get('pdf.info.Title', 'Unknown Document')
                        source = metadata.get('source', 'Unknown Source')
                        lines_from = metadata.get('loc.lines.from', '')
                        lines_to = metadata.get('loc.lines.to', '')
                        print(f"Source: {source}")
                        if title and title != 'Unknown Document' and title not in source:
                            print(f"Document Title: {title}")
                        if lines_from and lines_to:
                            print(f"Lines: {lines_from}-{lines_to}")
                        print(f"Content: {highlight_quote_in_content(doc.page_content, query)}")
                        print("-" * 40)
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 