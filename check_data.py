#!/usr/bin/env python3
"""Minimal script to check if data exists in Pinecone"""

import os
import sys
from dotenv import load_dotenv

# Try to use existing venv if available
if os.path.exists('venv/bin/activate'):
    # We'll let the user run this with their venv
    pass

load_dotenv()

try:
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import OpenAIEmbeddings
except ImportError as e:
    print(f"ERROR: Missing dependencies. Please run:")
    print(f"  source venv/bin/activate  # or your venv")
    print(f"  pip install pinecone-client langchain-pinecone langchain-openai python-dotenv")
    sys.exit(1)

print("=" * 70)
print("Pinecone Data Verification")
print("=" * 70)

# Connect
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("berkshire")

# Stats
stats = index.describe_index_stats()
print(f"\n✓ Index 'berkshire' found")
print(f"  Total vectors: {stats.total_vector_count:,}")
print(f"  Dimension: {stats.dimension}")

if stats.total_vector_count == 0:
    print("\n✗ ERROR: Index is EMPTY! No data found.")
    sys.exit(1)

# Test retrieval
print(f"\n{'='*70}")
print("Testing Retrieval")
print(f"{'='*70}")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    text_key="text"
)

test_queries = [
    ("swimming naked", ["swimming", "naked"]),
    ("tide goes out", ["tide", "goes", "out"]),
    ("float", ["float"]),
    ("Berkshire", ["berkshire"]),
]

for query, keywords in test_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 70)
    try:
        results = vectorstore.similarity_search(query, k=5)
        print(f"  Retrieved: {len(results)} documents")
        
        if not results:
            print(f"  ✗ NO RESULTS")
            continue
            
        # Check each result
        for i, doc in enumerate(results[:3], 1):
            content_lower = doc.page_content.lower()
            found_keywords = [kw for kw in keywords if kw.lower() in content_lower]
            
            print(f"\n  Result {i}:")
            print(f"    Keywords found: {found_keywords if found_keywords else 'NONE'}")
            print(f"    Preview: {doc.page_content[:150]}...")
            print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
            
            # Show context around keywords if found
            if found_keywords:
                for keyword in found_keywords:
                    idx = content_lower.find(keyword.lower())
                    if idx != -1:
                        start = max(0, idx - 50)
                        end = min(len(doc.page_content), idx + len(keyword) + 50)
                        context = doc.page_content[start:end]
                        print(f"    Context for '{keyword}': ...{context}...")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("Test Complete")
print(f"{'='*70}")
