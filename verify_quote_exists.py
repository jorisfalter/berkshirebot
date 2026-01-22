#!/usr/bin/env python3
"""Verify if the quote actually exists in the source documents"""

import os
import glob
from pathlib import Path

# The famous Buffett quote
QUOTE = "only when the tide goes out do you discover who's been swimming naked"
QUOTE_VARIATIONS = [
    "swimming naked",
    "tide goes out",
    "tide going out",
    "only when the tide",
    "discover who's been swimming",
    "who's been swimming naked",
]

print("=" * 70)
print("Verifying if quote exists in source documents")
print("=" * 70)
print(f"\nLooking for: '{QUOTE}'")
print(f"And variations: {QUOTE_VARIATIONS}")

# Check for PDF files in current directory
pdf_files = glob.glob("*.pdf")
print(f"\nFound {len(pdf_files)} PDF files in current directory")

if pdf_files:
    print("PDF files found:")
    for pdf in pdf_files[:10]:  # Show first 10
        print(f"  - {pdf}")

# Try to search PDFs using PyPDF2 if available
try:
    import PyPDF2
    PDF_AVAILABLE = True
    print("\n✓ PyPDF2 available - can search PDFs")
except ImportError:
    PDF_AVAILABLE = False
    print("\n⚠ PyPDF2 not available - install with: pip install PyPDF2")

# Search through PDFs
if PDF_AVAILABLE and pdf_files:
    print("\n" + "=" * 70)
    print("Searching PDF files...")
    print("=" * 70)
    
    found_in_files = []
    
    for pdf_file in pdf_files:
        try:
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages[:50]):  # Limit to first 50 pages for speed
                    text_content += page.extract_text() + "\n"
                
                text_lower = text_content.lower()
                
                # Check for quote and variations
                found_terms = []
                for term in [QUOTE] + QUOTE_VARIATIONS:
                    if term.lower() in text_lower:
                        found_terms.append(term)
                        # Find context
                        idx = text_lower.find(term.lower())
                        start = max(0, idx - 150)
                        end = min(len(text_content), idx + len(term) + 150)
                        context = text_content[start:end]
                        
                        print(f"\n✓ FOUND '{term}' in {pdf_file}")
                        print(f"  Context: ...{context}...")
                        print(f"  Page: ~{page_num + 1}")
                
                if found_terms:
                    found_in_files.append((pdf_file, found_terms))
                    
        except Exception as e:
            print(f"⚠ Error reading {pdf_file}: {e}")
    
    if found_in_files:
        print(f"\n{'='*70}")
        print(f"SUMMARY: Quote found in {len(found_in_files)} file(s)")
        for pdf, terms in found_in_files:
            print(f"  - {pdf}: {terms}")
    else:
        print(f"\n{'='*70}")
        print("SUMMARY: Quote NOT FOUND in any PDF files")
        print("This means either:")
        print("  1. The quote doesn't exist in these documents")
        print("  2. The documents weren't indexed")
        print("  3. The documents are in Google Drive (not local)")

# Check if documents are in Google Drive
print(f"\n{'='*70}")
print("Checking Google Drive setup...")
print("=" * 70)

if os.getenv("GOOGLE_DRIVE_FOLDER_ID"):
    print(f"✓ GOOGLE_DRIVE_FOLDER_ID is set")
    print(f"  Documents are likely in Google Drive, not local files")
    print(f"  To verify, you would need to:")
    print(f"  1. Run google_drive_rag.py to see what documents were loaded")
    print(f"  2. Or check Google Drive folder directly")
else:
    print("⚠ GOOGLE_DRIVE_FOLDER_ID not set")
    print("  Documents might be in Google Drive or need to be downloaded")

# Check Pinecone to see what sources are indexed
print(f"\n{'='*70}")
print("Checking what sources are in Pinecone index...")
print("=" * 70)

try:
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import OpenAIEmbeddings
    from dotenv import load_dotenv
    
    load_dotenv()
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("berkshire")
    
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        text_key="text"
    )
    
    # Get a sample of documents to see what sources exist
    sample_results = vectorstore.similarity_search("Berkshire", k=20)
    
    sources = set()
    for doc in sample_results:
        source = doc.metadata.get('source', 'Unknown')
        sources.add(source)
    
    print(f"Found {len(sources)} unique sources in index:")
    for source in sorted(sources)[:20]:  # Show first 20
        print(f"  - {source}")
    
    if len(sources) > 20:
        print(f"  ... and {len(sources) - 20} more")
    
    # Now search through indexed documents for the quote
    print(f"\n{'='*70}")
    print("Searching indexed documents for quote...")
    print("=" * 70)
    
    # Get many documents and search through them
    all_docs = vectorstore.similarity_search("", k=100)  # Get diverse sample
    
    found_in_index = []
    for doc in all_docs:
        content_lower = doc.page_content.lower()
        source = doc.metadata.get('source', 'Unknown')
        
        for term in [QUOTE] + QUOTE_VARIATIONS:
            if term.lower() in content_lower:
                idx = content_lower.find(term.lower())
                start = max(0, idx - 100)
                end = min(len(doc.page_content), idx + len(term) + 100)
                context = doc.page_content[start:end]
                
                print(f"\n✓ FOUND '{term}' in indexed document!")
                print(f"  Source: {source}")
                print(f"  Lines: {doc.metadata.get('loc.lines.from', '?')}-{doc.metadata.get('loc.lines.to', '?')}")
                print(f"  Context: ...{context}...")
                found_in_index.append((source, term))
                break  # Only report once per document
    
    if found_in_index:
        print(f"\n{'='*70}")
        print(f"✓ Quote EXISTS in {len(found_in_index)} indexed document(s)")
        print("=" * 70)
        print("The problem is RETRIEVAL, not indexing!")
        print("The hybrid search should help fix this.")
    else:
        print(f"\n{'='*70}")
        print("⚠ Quote NOT FOUND in sampled indexed documents")
        print("=" * 70)
        print("This could mean:")
        print("  1. The quote is in documents not in this sample")
        print("  2. The quote was split across chunks")
        print("  3. The quote doesn't exist in the indexed documents")
        print("\nTry searching with: python3 cli_query.py")
        print("Then use: !search swimming naked")
        
except ImportError:
    print("⚠ Cannot check Pinecone - missing dependencies")
except Exception as e:
    print(f"⚠ Error checking Pinecone: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
print("Verification complete")
print("=" * 70)
