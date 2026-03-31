"""
Main Script for Inverted Index Demonstration

This script loads documents from the 'documents/' directory, builds an inverted index
using the InvIndex class, displays the index, and provides an interactive query interface
for searching documents.

Author: [Your Name]
Date: [Current Date]
"""

import os
import nltk

# nltk.download() fetches required NLTK data packages (e.g., tokenizers, corpora) from the NLTK repository.
# These are needed for text processing functions like tokenization and lemmatization.
# The 'quiet=True' parameter suppresses download messages for cleaner output.
# Packages are cached locally after first download.
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from inv_index import InvIndex


def load_documents(directory):
    """
    Load documents from a specified directory.

    Reads all .txt files in the directory, assigns document IDs based on filenames
    (e.g., '1.txt' -> ID 1), and stores their content in a dictionary.

    Args:
        directory (str): Path to the directory containing .txt files.

    Returns:
        dict: Dictionary of {doc_id: text} pairs.
    """
    docs = {}  # Dictionary to store document ID and content
    for filename in sorted(os.listdir(directory)):  # Iterate over sorted filenames
        if filename.endswith('.txt'):  # Only process .txt files
            doc_id = int(filename.split('.')[0])  # Extract ID from filename (e.g., '1.txt' -> 1)
            filepath = os.path.join(directory, filename)  # Build full file path
            with open(filepath, 'r', encoding='utf-8') as f:  # Open file with UTF-8 encoding
                docs[doc_id] = f.read().strip()  # Read content and strip whitespace
    return docs


def main():
    """
    Main function to demonstrate the inverted index.

    Loads documents, builds the index, displays it, and enters an interactive search loop.
    """
    docs = load_documents('documents')  # Load documents from 'documents/' directory

    print("=== Documents ===")  # Display section header
    for doc_id, text in docs.items():  # Print each document
        print(f"  Doc {doc_id}: {text}")

    index = InvIndex()  # Create an InvIndex instance
    index.build_index(docs)  # Build the inverted index from documents

    print("\n=== Inverted Index ===")  # Display section header
    for term in sorted(index.inv.keys()):  # Print each term and its postings list
        print(f"  {term}: {index.inv[term]}")

    print("\n=== Query Search ===")  # Display section header for search
    while True:  # Interactive loop for queries
        query = input("\nEnter query (or 'quit' to exit): ").strip()  # Get user input
        if not query or query.lower() == 'quit':  # Check for exit condition
            break
        results = index.search(query)  # Perform search
        if results:  # If results found
            result_str = ', '.join(f'd{doc_id}' for doc_id in results)  # Format results
            print(f"  Results: {result_str}")
        else:  # No results
            print("  No matching documents found.")


if __name__ == '__main__':
    main()  # Run the main function when script is executed directly
    main()
