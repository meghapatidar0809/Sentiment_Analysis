import pickle
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  # Not using it
from tqdm import tqdm
from typing import Tuple
from scipy import sparse as sp
from collections import Counter  # For TF-IDF Vectorizer



class Vectorizer:
    """
    A vectorizer class that converts text data into a sparse matrix
    """
    def __init__(self, max_vocab_len=10_000) -> None:
        """
        Initialize the vectorizer
        """
        self.vocab = None
        
        # TODO: Add more class variables if needed
        self.max_vocab_len = max_vocab_len
        self.word_to_index = None
    
    
    
    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit the vectorizer on the training data
        :param X_train: np.ndarray
            The training sentences
        :return: None
        """
        
        # TODO: count the occurrences of each word
        word_counts = {}             # Dictionary to store word frequencies
        for sentence in X_train:
            words = sentence.split() # Tokenize sentence into words
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1 # Count word occurrences
        
        # TODO: sort the words based on frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # TODO: retain the top 10k words
        self.vocab = []  

        for word, _ in sorted_words[:self.max_vocab_len]: # Iterate over the sorted words up to max_vocab_len(10k) and store them in vocab
            self.vocab.append(word)  

        self.word_to_index = {}                           # Initialize an empty dictionary to store word-to-index mapping

        for index, word in enumerate(self.vocab):         # Assign an index to each word in vocab
            self.word_to_index[word] = index
    
    
    
    
    def transform(self, X: np.ndarray) -> sp.csr_matrix:
        """
        Transform the input sentences into a sparse matrix based on the
        vocabulary obtained after fitting the vectorizer
        ! Do NOT return a dense matrix, as it will be too large to fit in memory
        :param X: np.ndarray
            Input sentences (can be either train, val or test)
        :return: sp.csr_matrix
            The sparse matrix representation of the input sentences
        """
        assert self.vocab is not None, "Vectorizer not fitted yet"
        
        # TODO: convert the input sentences into vectors
        rows, columns, values = [], [], [] # Lists to store non-zero values and their positions.

        for i, sentence in enumerate(X):
            word_counts = {}            # Dictionary to count word occurrences in the sentence.
            words = sentence.split()    # Tokenize the sentence into words.
            for word in words:
                if word in self.word_to_index:                          # Check if the word is in the vocabulary.
                    index = self.word_to_index[word]                    # Get the word's index in the vocabulary.
                    word_counts[index] = word_counts.get(index, 0) + 1  # Update the word count.
            
            for index, count in word_counts.items():
                rows.append(i)          # Store row index corresponding to the sentence.
                columns.append(index)   # Store column index corresponding to the word.
                values.append(count)    # Store word frequency in the matrix.

        # Create a sparse matrix from the collected values.
        return sp.csr_matrix((values, (rows, columns)), shape=(len(X), len(self.vocab)))





class TFIDFVectorizer:
    """
    A vectorizer class that converts text data into a sparse matrix using TF-IDF
    with double normalization IDF.
    """
    def __init__(self, max_vocab_len=10_000) -> None:
        """
        Initialize the vectorizer
        """
        self.vocab = None
        self.max_vocab_len = max_vocab_len
        self.word_to_index = None
    
    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit the vectorizer on the training data
        :param X_train: np.ndarray
            The training sentences
        :return: None
        """
        word_counts = Counter()
        doc_counts = Counter()
        total_docs = len(X_train)
        
        # Count term frequencies and document frequencies
        for doc in X_train:
            words = doc.split()
            word_counts.update(words)
            doc_counts.update(set(words))
        
        # Retain the most frequent words up to max_vocab_len
        most_common = word_counts.most_common(self.max_vocab_len)
        self.vocab = {word: index for index, (word, _) in enumerate(most_common)}
        
        # Compute IDF using double normalization IDF
        self.idf = {
            word: np.log((total_docs / (1 + doc_counts[word])))
            for word in self.vocab
        }
    
    def transform(self, X: np.ndarray) -> sp.csr_matrix:
        """
        Transform the input sentences into a sparse TF-IDF matrix based on double normalization IDF
        :param X: np.ndarray
            Input sentences (can be train, val, or test)
        :return: sp.csr_matrix
            The sparse matrix representation of the input sentences
        """
        assert self.vocab is not None, "Vectorizer not fitted yet"
        
        rows, columns, values = [], [], []
        
        for row_index, doc in enumerate(X):
            words = doc.split()
            word_freqs = Counter(words)
            max_freq = max(word_freqs.values()) if word_freqs else 1
            
            for word, freq in word_freqs.items():
                if word in self.vocab:
                    col_index = self.vocab[word]
                    tf = 0.5 + 0.5 * (freq / max_freq)  # Double normalization TF
                    idf = self.idf.get(word, 0)         # IDF
                    rows.append(row_index)
                    columns.append(col_index)
                    values.append(tf * idf)
        
        return sp.csr_matrix((values, (rows, columns)), shape=(len(X), len(self.vocab)))



# No changes here onwards

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_data(
        path: str,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load twitter sentiment data from csv file and split into train, val and
    test set. Relabel the targets to -1 (for negative) and +1 (for positive).

    :param path: str
        The path to the csv file
    :param seed: int
        The random state for reproducibility
    :return:
        Tuple of numpy arrays - (data, labels) x (train, val) respectively
    """
    # load data
    df = pd.read_csv(path, encoding='utf-8')
    
    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # split into train, val and test set
    train_size = int(0.8 * len(df))  # ~1M for training, remaining ~250k for val
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    x_train, y_train =\
        train_df['stemmed_content'].values, train_df['target'].values
    x_val, y_val = val_df['stemmed_content'].values, val_df['target'].values
    return x_train, y_train, x_val, y_val





    
    
