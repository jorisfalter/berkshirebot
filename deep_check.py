#!/usr/bin/env python3
"""Deep check - search for the actual quote"""

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("berkshire")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    text_key="text"
)

# Try the actual famous quote
quotes_to_test = [
    "only when the tide goes out do you discover who's been swimming naked",
    "only when the tide goes out",
    "discover who's been swimming naked",
    "who's been swimming naked",
    "tide goes out do you discover",
    "swimming naked when the tide",
]

print("=" * 70)
print("Searching for the actual Buffett quote")
print("=" * 70)

for quote in quotes_to_test:
    print(f"\nQuery: '{quote}'")
    print("-" * 70)
    results = vectorstore.similarity_search(quote, k=5)
    print(f"Found {len(results)} results")
    
    for i, doc in enumerate(results, 1):
        content_lower = doc.page_content.lower()
        # Check if any part of the quote is in the content
        quote_words = quote.lower().split()
        found_words = [w for w in quote_words if w in content_lower]
        
        print(f"\n  Result {i}:")
        print(f"    Found words: {found_words}")
        print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"    Lines: {doc.metadata.get('loc.lines.from', '')}-{doc.metadata.get('loc.lines.to', '')}")
        print(f"    Content: {doc.page_content[:300]}...")
        
        # If we found the phrase, show more context
        if len(found_words) >= 3:  # Found at least 3 words
            # Find where the words appear
            for word in found_words[:3]:
                idx = content_lower.find(word)
                if idx != -1:
                    start = max(0, idx - 100)
                    end = min(len(doc.page_content), idx + len(word) + 100)
                    context = doc.page_content[start:end]
                    print(f"    Context for '{word}': ...{context}...")

# Also try with ada-002 to see if that's what was used for indexing
print("\n" + "=" * 70)
print("Testing with text-embedding-ada-002 (in case that's what indexed it)")
print("=" * 70)

vectorstore_ada = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(),  # Defaults to ada-002
    text_key="text"
)

results_ada = vectorstore_ada.similarity_search("swimming naked", k=5)
print(f"\nWith ada-002, found {len(results_ada)} results for 'swimming naked'")

for i, doc in enumerate(results_ada[:3], 1):
    content_lower = doc.page_content.lower()
    found = "swimming" in content_lower or "naked" in content_lower
    print(f"\n  Result {i}:")
    print(f"    Contains 'swimming' or 'naked': {found}")
    print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"    Content: {doc.page_content[:200]}...")
