#!/usr/bin/env python3
"""Search for the quote in raw text to see if it exists at all"""

import os
import glob
from pathlib import Path

# Search terms
search_terms = [
    "swimming naked",
    "swimming",
    "naked",
    "tide goes out",
    "tide going out",
    "only when the tide",
    "discover who's been",
]

print("=" * 70)
print("Searching for quote in local files")
print("=" * 70)

# Look for PDF files
pdf_files = glob.glob("*.pdf")
txt_files = glob.glob("*.txt")
all_files = pdf_files + txt_files

if not all_files:
    print("No PDF or TXT files found in current directory")
    print(f"Current dir: {os.getcwd()}")
    print("\nLooking in common locations...")
    # Check if there's a documents folder
    for folder in ["documents", "pdfs", "data", "berkshire"]:
        if os.path.exists(folder):
            pdf_files = glob.glob(f"{folder}/*.pdf")
            all_files.extend(pdf_files)
            print(f"Found {len(pdf_files)} PDFs in {folder}/")

if not all_files:
    print("\nNo files found. The quote might be in Google Drive.")
    print("You may need to re-index the documents or check if they contain the quote.")
else:
    print(f"\nFound {len(all_files)} files to search")
    print("Note: This will only search file names and basic text extraction")
    print("For full PDF search, you'd need PyPDF2 or similar")
    
    # Try to read text files
    for file in txt_files[:5]:  # Limit to first 5
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                for term in search_terms:
                    if term.lower() in content:
                        print(f"\n✓ Found '{term}' in {file}")
                        # Find context
                        idx = content.find(term.lower())
                        start = max(0, idx - 100)
                        end = min(len(content), idx + len(term) + 200)
                        context = content[start:end]
                        print(f"  Context: ...{context}...")
        except Exception as e:
            pass

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("If the quote exists in the source documents but isn't being found,")
print("the problem is likely:")
print("1. The text was split across multiple chunks during indexing")
print("2. The embedding model isn't capturing the semantic meaning")
print("3. The chunks are too small (1000 chars with 200 overlap)")
print("\nSOLUTION: We need to either:")
print("- Increase chunk size and overlap")
print("- Use a hybrid search (keyword + semantic)")
print("- Re-index with better chunking strategy")
