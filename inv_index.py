"""
Inverted Index Implementation for Information Retrieval

This module implements an Inverted Index class that processes documents,
builds an inverted index, and supports query searches. It utilizes NLTK for
text preprocessing including tokenization, stop word removal, stemming, and lemmatization.

Author: [Your Name]
Date: [Current Date]
"""

import re
import collections
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


class InvIndex:
    """
    A class to build and manage an inverted index for information retrieval.

    The inverted index maps terms to lists of [document_id, term_frequency] pairs.
    Text preprocessing includes lowercasing, punctuation removal, tokenization,
    stop word filtering, stemming, and lemmatization.
    """

    def __init__(self):
        """
        Initialize the Inverted Index.

        Sets up the index dictionary, stemmer, lemmatizer, and stop words set.
        """
        self.inv = {}  # Dictionary to store the inverted index: term -> list of [doc_id, freq]
        self.stemmer = SnowballStemmer('english')  # Stemmer for reducing words to root forms
        self.lemmatizer = WordNetLemmatizer()  # Lemmatizer for canonical word forms
        self.stop_words = set(stopwords.words('english'))  # Set of English stop words for filtering

    def preprocess(self, text):
        """
        Preprocess the input text for indexing or querying.

        Steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Tokenize into words
        4. Remove stop words
        5. Apply stemming
        6. Apply lemmatization

        Args:
            text (str): The input text to preprocess.

        Returns:
            list: List of preprocessed tokens.
        """
        text = text.lower()  # Convert text to lowercase for case-insensitive processing
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation using regex
        tokens = word_tokenize(text)  # Tokenize text into individual words
        tokens = [t for t in tokens if t not in self.stop_words]  # Filter out stop words
        tokens = [self.stemmer.stem(t) for t in tokens]  # Apply stemming to reduce to root forms
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]  # Apply lemmatization for base forms
        return tokens

    def add_document(self, doc_id, text):
        """
        Add a document to the inverted index.

        Preprocesses the text, counts term frequencies, and updates the index.

        Args:
            doc_id (int): Unique identifier for the document.
            text (str): The content of the document.
        """
        tokens = self.preprocess(text)  # Preprocess the document text
        freq = collections.Counter(tokens)  # Count frequency of each token
        for term, count in freq.items():  # Iterate over term-frequency pairs
            if term not in self.inv:  # If term not in index, initialize list
                self.inv[term] = []
            self.inv[term].append([doc_id, count])  # Append [doc_id, frequency] to term's list

    def build_index(self, docs):
        """
        Build the inverted index from a collection of documents.

        Args:
            docs (dict): Dictionary of {doc_id: text} pairs.
        """
        for doc_id, text in docs.items():  # Iterate over each document
            self.add_document(doc_id, text)  # Add each document to the index

    def search(self, query):
        """
        Search the inverted index for documents matching the query.

        Preprocesses the query and finds documents containing all query terms.

        Args:
            query (str): The search query.

        Returns:
            list: Sorted list of document IDs that match the query.
        """
        query_terms = self.preprocess(query)  # Preprocess the query
        if not query_terms:  # If no terms after preprocessing, return empty
            return []
        result_sets = []  # List to hold sets of doc IDs for each term
        for term in query_terms:  # For each term in the query
            if term in self.inv:  # If term exists in index
                doc_ids = set(pair[0] for pair in self.inv[term])  # Get set of doc IDs for the term
            else:  # If any term not found, no results
                return []
            result_sets.append(doc_ids)  # Add the set to result_sets
        # Return intersection of all doc ID sets, sorted
        return sorted(set.intersection(*result_sets))
