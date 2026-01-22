#!/usr/bin/env python3
"""Quick test using existing setup"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

print("Testing Pinecone 'berkshire' index...")
print("=" * 60)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("berkshire")

# Get stats
stats = index.describe_index_stats()
print(f"\nIndex Stats:")
print(f"  Total vectors: {stats.total_vector_count}")
print(f"  Dimension: {stats.dimension}")

# Test retrieval
vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    text_key="text"
)

queries = ["swimming naked", "swimming", "naked", "tide goes out", "float"]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = vectorstore.similarity_search(query, k=3)
    print(f"  Results: {len(results)}")
    if results:
        content = results[0].page_content.lower()
        print(f"  First result (first 200 chars): {results[0].page_content[:200]}...")
        # Check if query words are in result
        query_words = query.lower().split()
        found = [w for w in query_words if w in content]
        print(f"  Query words found in result: {found if found else 'NONE'}")
