"""
Simple fine-grained topic modeling using TF-IDF and K-Means.
This is a lightweight alternative to embedding-based methods that only requires scikit-learn.

Architecture:
1. TF-IDF Vectorization (captures word importance)
2. Dimensionality Reduction with TruncatedSVD (like PCA but faster)
3. K-Means Clustering (simple, fast, effective)
4. Top words per cluster using TF-IDF scores
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


class SimpleFineGrainedTopicModel:
    """
    Simple fine-grained topic model using TF-IDF + K-Means.
    
    This is much simpler than embedding-based methods but still effective:
    - Uses TF-IDF to capture word importance
    - Reduces dimensions with SVD
    - Clusters with K-Means
    - No heavy dependencies (PyTorch, transformers, etc.)
    """
    
    def __init__(
        self,
        n_topics: int = 30,
        max_features: int = 5000,
        n_components: int = 50,
        top_n_words: int = 30,
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """
        Initialize the simple topic model.
        
        Args:
            n_topics: Number of topics to discover (30 is good for fine-grained)
            max_features: Maximum vocabulary size
            n_components: Dimensions for SVD reduction
            top_n_words: Number of top words per topic (30 for detailed descriptions)
            min_df: Minimum document frequency for a word
            max_df: Maximum document frequency for a word (filters common words)
        """
        self.n_topics = n_topics
        self.max_features = max_features
        self.n_components = n_components
        self.top_n_words = top_n_words
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = None
        self.svd = None
        self.kmeans = None
        self.topic_words = {}
        self.topic_labels = None
        self.feature_names = None
    
    def fit(self, documents: List[str], progress_callback=None) -> 'SimpleFineGrainedTopicModel':
        """
        Fit the topic model on documents.
        
        Args:
            documents: List of text documents
            progress_callback: Optional callback for progress updates
            
        Returns:
            Self for chaining
        """
        if not documents:
            raise ValueError("No documents provided")
        
        # Step 1: TF-IDF Vectorization
        if progress_callback:
            progress_callback("Computing TF-IDF vectors...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            strip_accents='unicode',
            lowercase=True
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Step 2: Dimensionality Reduction with SVD
        if progress_callback:
            progress_callback("Reducing dimensionality with SVD...")
        
        n_components = min(self.n_components, tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_vectors = self.svd.fit_transform(tfidf_matrix)
        
        # Normalize for better clustering
        reduced_vectors = normalize(reduced_vectors)
        
        # Step 3: K-Means Clustering
        if progress_callback:
            progress_callback(f"Clustering into {self.n_topics} topics with K-Means...")
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_topics,
            random_state=42,
            batch_size=1000,
            n_init=10
        )
        self.topic_labels = self.kmeans.fit_predict(reduced_vectors)
        
        # Step 4: Extract top words per topic
        if progress_callback:
            progress_callback("Extracting topic words...")
        
        self._extract_topic_words(tfidf_matrix)
        
        return self
    
    def _extract_topic_words(self, tfidf_matrix):
        """
        Extract top words for each topic based on TF-IDF scores.
        """
        # For each topic, get the centroid in TF-IDF space
        for topic_id in range(self.n_topics):
            # Get all documents in this topic
            topic_mask = self.topic_labels == topic_id
            
            if not np.any(topic_mask):
                self.topic_words[topic_id] = []
                continue
            
            # Get average TF-IDF vector for this topic
            topic_tfidf = tfidf_matrix[topic_mask].mean(axis=0).A1
            
            # Get top words by TF-IDF score
            top_indices = topic_tfidf.argsort()[-self.top_n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_indices if topic_tfidf[i] > 0]
            
            self.topic_words[topic_id] = top_words
    
    def get_topic_assignments(self) -> np.ndarray:
        """Get topic assignments for each document."""
        return self.topic_labels
    
    def get_topic_words(self, topic_id: int, n: int = 30) -> List[str]:
        """Get top words for a specific topic."""
        if topic_id in self.topic_words:
            return self.topic_words[topic_id][:n]
        return []
    
    def get_all_topics(self) -> Dict[int, List[str]]:
        """Get all topics and their top words."""
        return {
            topic_id: words  # Return all extracted words (up to top_n_words)
            for topic_id, words in self.topic_words.items()
        }
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about all topics."""
        topic_info = []
        
        for topic_id in range(self.n_topics):
            count = np.sum(self.topic_labels == topic_id)
            words = self.get_topic_words(topic_id, n=30)  # Get all 30 words
            
            topic_info.append({
                'topic_id': topic_id,
                'top_words': ', '.join(words),
                'count': count
            })
        
        return pd.DataFrame(topic_info)


def train_simple_topic_model(
    documents: List[str],
    n_topics: int = 30,
    progress_callback=None
) -> Tuple[SimpleFineGrainedTopicModel, np.ndarray]:
    """
    Train a simple fine-grained topic model using TF-IDF + K-Means.
    
    This is a lightweight alternative that only requires scikit-learn.
    
    Args:
        documents: List of text documents
        n_topics: Number of topics to discover (30 is good for fine-grained analysis)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (fitted model, topic assignments per document)
    """
    # Create and fit model
    model = SimpleFineGrainedTopicModel(
        n_topics=n_topics,
        max_features=5000,
        n_components=50,
        top_n_words=30,  # Extract 30 words per topic for detailed descriptions
        min_df=2,
        max_df=0.95
    )
    
    model.fit(documents, progress_callback=progress_callback)
    
    topic_assignments = model.get_topic_assignments()
    
    return model, topic_assignments

