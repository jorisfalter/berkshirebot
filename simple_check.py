#!/usr/bin/env python3
"""Simple check - uses same imports as cli_query.py"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("Pinecone Data Check")
print("=" * 70)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("berkshire")

stats = index.describe_index_stats()
print(f"\nIndex Stats:")
print(f"  Total vectors: {stats.total_vector_count:,}")
print(f"  Dimension: {stats.dimension}")

if stats.total_vector_count == 0:
    print("\n✗ INDEX IS EMPTY!")
    sys.exit(1)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    text_key="text"
)

print(f"\n{'='*70}")
print("Testing Queries")
print(f"{'='*70}")

queries = [
    "swimming naked",
    "swimming", 
    "naked",
    "tide goes out",
    "tide",
    "float"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    try:
        results = vectorstore.similarity_search(query, k=3)
        print(f"  Found: {len(results)} results")
        
        if results:
            content = results[0].page_content.lower()
            query_words = query.lower().split()
            found = [w for w in query_words if w in content]
            
            print(f"  Top result preview: {results[0].page_content[:200]}...")
            print(f"  Query words in result: {found if found else 'NONE'}")
            print(f"  Source: {results[0].metadata.get('source', 'Unknown')}")
        else:
            print(f"  ✗ NO RESULTS")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

print(f"\n{'='*70}")
