#!/usr/bin/env python3
"""Test script to directly query Pinecone and verify data exists"""

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 60)
print("Pinecone Database Test")
print("=" * 60)

# Initialize Pinecone
print("\n1. Connecting to Pinecone...")
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    print("✓ Connected to Pinecone")
except Exception as e:
    print(f"✗ Failed to connect: {e}")
    exit(1)

# Check if index exists
print("\n2. Checking 'berkshire' index...")
try:
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    print(f"Available indexes: {index_names}")
    
    if "berkshire" not in index_names:
        print("✗ 'berkshire' index not found!")
        exit(1)
    
    index = pc.Index("berkshire")
    print("✓ 'berkshire' index found")
except Exception as e:
    print(f"✗ Error checking index: {e}")
    exit(1)

# Get index stats
print("\n3. Getting index statistics...")
try:
    stats = index.describe_index_stats()
    print(f"Total vectors: {stats.total_vector_count}")
    print(f"Dimension: {stats.dimension}")
    print(f"Index fullness: {stats.index_fullness}")
    if hasattr(stats, 'namespaces') and stats.namespaces:
        print(f"Namespaces: {list(stats.namespaces.keys())}")
except Exception as e:
    print(f"✗ Error getting stats: {e}")

# Test with different embedding models
print("\n4. Testing retrieval with text-embedding-3-small...")
try:
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        text_key="text"
    )
    
    # Test queries
    test_queries = [
        "swimming naked",
        "swimming",
        "naked",
        "tide goes out",
        "tide",
        "float",
        "Berkshire Hathaway"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        try:
            results = vectorstore.similarity_search(query, k=5)
            print(f"  Found {len(results)} results")
            
            if results:
                print(f"  Top result preview:")
                print(f"    {results[0].page_content[:200]}...")
                print(f"    Metadata: {results[0].metadata}")
                
                # Check if query terms appear in results
                query_lower = query.lower()
                found_terms = []
                for result in results[:3]:
                    content_lower = result.page_content.lower()
                    for word in query_lower.split():
                        if word in content_lower:
                            found_terms.append(word)
                
                if found_terms:
                    print(f"  ✓ Found query terms in results: {set(found_terms)}")
                else:
                    print(f"  ⚠ Query terms not found in top results")
            else:
                print(f"  ✗ No results found!")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            
except Exception as e:
    print(f"✗ Error with text-embedding-3-small: {e}")
    import traceback
    traceback.print_exc()

# Test with ada-002 as fallback
print("\n5. Testing retrieval with text-embedding-ada-002 (fallback)...")
try:
    vectorstore_ada = PineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(),  # Defaults to ada-002
        text_key="text"
    )
    
    results = vectorstore_ada.similarity_search("swimming naked", k=3)
    print(f"Found {len(results)} results with ada-002")
    if results:
        print(f"Top result: {results[0].page_content[:200]}...")
except Exception as e:
    print(f"✗ Error with ada-002: {e}")

# Try to fetch some raw vectors to see what's stored
print("\n6. Sampling raw vectors from index...")
try:
    # Query with a dummy vector to see what we get back
    stats = index.describe_index_stats()
    if stats.total_vector_count > 0:
        # Use query to get some sample data
        sample_query = vectorstore.similarity_search("test", k=1)
        if sample_query:
            print("✓ Can retrieve vectors from index")
            print(f"Sample metadata keys: {list(sample_query[0].metadata.keys())}")
        else:
            print("⚠ Index appears empty or query returned no results")
    else:
        print("✗ Index is empty (0 vectors)")
except Exception as e:
    print(f"✗ Error sampling vectors: {e}")

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
