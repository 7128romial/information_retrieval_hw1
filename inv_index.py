import re
import collections
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


class InvIndex:
    def __init__(self):
        self.inv = {}
        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.stemmer.stem(t) for t in tokens]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def add_document(self, doc_id, text):
        tokens = self.preprocess(text)
        freq = collections.Counter(tokens)
        for term, count in freq.items():
            if term not in self.inv:
                self.inv[term] = []
            self.inv[term].append([doc_id, count])

    def build_index(self, docs):
        for doc_id, text in docs.items():
            self.add_document(doc_id, text)

    def search(self, query):
        query_terms = self.preprocess(query)
        if not query_terms:
            return []
        result_sets = []
        for term in query_terms:
            if term in self.inv:
                doc_ids = set(pair[0] for pair in self.inv[term])
            else:
                return []
            result_sets.append(doc_ids)
        return sorted(set.intersection(*result_sets))
