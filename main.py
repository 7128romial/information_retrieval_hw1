import os
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from inv_index import InvIndex


def load_documents(directory):
    docs = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):
            doc_id = int(filename.split('.')[0])
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                docs[doc_id] = f.read().strip()
    return docs


def main():
    docs = load_documents('documents')

    print("=== Documents ===")
    for doc_id, text in docs.items():
        print(f"  Doc {doc_id}: {text}")

    index = InvIndex()
    index.build_index(docs)

    print("\n=== Inverted Index ===")
    for term in sorted(index.inv.keys()):
        print(f"  {term}: {index.inv[term]}")

    print("\n=== Query Search ===")
    while True:
        query = input("\nEnter query (or 'quit' to exit): ").strip()
        if not query or query.lower() == 'quit':
            break
        results = index.search(query)
        if results:
            result_str = ', '.join(f'd{doc_id}' for doc_id in results)
            print(f"  Results: {result_str}")
        else:
            print("  No matching documents found.")


if __name__ == '__main__':
    main()
