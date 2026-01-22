#!/usr/bin/env python3
"""Test the hybrid search to see if it finds the quote"""

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Import the retriever logic
import sys
sys.path.insert(0, os.path.dirname(__file__))

from berkshire_rag import MetadataRetriever

print("=" * 70)
print("Testing Hybrid Search")
print("=" * 70)

# Setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("berkshire")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    text_key="text"
)

# Create retriever
retriever = MetadataRetriever(
    vectorstore=vectorstore,
    search_kwargs={"k": 10}
)

# Test queries
test_queries = [
    "swimming naked",
    "tide goes out",
    "when did they write about swimming naked",
    "what about the tide going out",
]

for query in test_queries:
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print("=" * 70)
    
    try:
        docs = retriever._get_relevant_documents(query)
        
        print(f"\nRetrieved {len(docs)} documents:")
        
        found_quote = False
        for i, doc in enumerate(docs[:5], 1):  # Show top 5
            content_lower = doc.page_content.lower()
            source = doc.metadata.get('source', 'Unknown')
            lines = f"{doc.metadata.get('loc.lines.from', '?')}-{doc.metadata.get('loc.lines.to', '?')}"
            
            # Check if it contains the quote
            has_swimming = "swimming" in content_lower and "naked" in content_lower
            has_tide = "tide" in content_lower and "goes" in content_lower and "out" in content_lower
            
            marker = "✓" if (has_swimming or has_tide) else " "
            if has_swimming or has_tide:
                found_quote = True
            
            print(f"\n{marker} Result {i}:")
            print(f"   Source: {source}")
            print(f"   Lines: {lines}")
            print(f"   Preview: {doc.page_content[:200]}...")
            
            if has_swimming:
                # Find the context
                idx = content_lower.find("swimming")
                start = max(0, idx - 50)
                end = min(len(doc.page_content), idx + 100)
                print(f"   ✓ Contains 'swimming naked': ...{doc.page_content[start:end]}...")
            
            if has_tide:
                idx = content_lower.find("tide")
                start = max(0, idx - 50)
                end = min(len(doc.page_content), idx + 100)
                print(f"   ✓ Contains 'tide goes out': ...{doc.page_content[start:end]}...")
        
        if found_quote:
            print(f"\n✓ SUCCESS: Found quote in results!")
        else:
            print(f"\n✗ FAILED: Quote not in top results")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("Test Complete")
print("=" * 70)
