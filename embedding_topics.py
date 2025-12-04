"""Embedding-based topic modeling for fine-grained topic discovery.

This module implements BERTopic-style topic modeling using:
- Sentence embeddings (sentence-transformers)
- UMAP for dimensionality reduction
- HDBSCAN for density-based clustering
- Class-based TF-IDF for topic word extraction
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from collections import Counter
import warnings

# Check for required libraries
try:
    # Check PyTorch version first
    import torch
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (2, 0):
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        warnings.warn(f"PyTorch {torch.__version__} is too old. sentence-transformers requires PyTorch 2.0+. "
                     "Upgrade with: pip install --upgrade torch torchvision torchaudio")
    else:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    if 'torch' in str(e):
        warnings.warn("PyTorch not available. Install with: pip install torch>=2.0.0 sentence-transformers")
    else:
        warnings.warn("sentence-transformers not available. Install with: pip install sentence-transformers")
except Exception as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn(f"Error loading sentence-transformers: {e}")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("umap-learn not available. Install with: pip install umap-learn")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("hdbscan not available. Install with: pip install hdbscan")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def check_dependencies():
    """Check if all required dependencies are available."""
    missing = []
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing.append("sentence-transformers")
    if not UMAP_AVAILABLE:
        missing.append("umap-learn")
    if not HDBSCAN_AVAILABLE:
        missing.append("hdbscan")
    
    return len(missing) == 0, missing


class EmbeddingTopicModel:
    """
    Embedding-based topic modeling using BERTopic-style approach.
    
    Architecture:
    1. Sentence Embeddings: all-MiniLM-L6-v2 (384-dim, fast, good quality)
    2. Dimensionality Reduction: UMAP (to 5 dimensions)
    3. Clustering: HDBSCAN (density-based, auto-determines cluster count)
    4. Topic Words: Class-based TF-IDF (extracts representative words per cluster)
    
    This provides fine-grained topics compared to traditional LDA.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.0,
        hdbscan_min_cluster_size: int = 10,
        hdbscan_min_samples: int = 5,
        top_n_words: int = 30
    ):
        """
        Initialize the embedding-based topic model.
        
        Args:
            embedding_model_name: Sentence transformer model name
                - "all-MiniLM-L6-v2": Fast, 384-dim, good quality (default)
                - "all-mpnet-base-v2": Slower, 768-dim, best quality
            umap_n_components: Target dimensionality for UMAP (5 works well)
            umap_n_neighbors: UMAP neighborhood size (15 balances local/global)
            umap_min_dist: UMAP minimum distance (0.0 for tight clusters)
            hdbscan_min_cluster_size: Minimum cluster size (10 for fine-grained)
            hdbscan_min_samples: HDBSCAN noise threshold (5 is conservative)
            top_n_words: Number of top words to extract per topic (30 for detailed descriptions)
        """
        self.embedding_model_name = embedding_model_name
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.top_n_words = top_n_words
        
        self.embedding_model = None
        self.umap_model = None
        self.hdbscan_model = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.topic_labels = None
        self.topic_words = {}
        self.num_topics = 0
    
    def fit(self, documents: List[str], progress_callback=None) -> 'EmbeddingTopicModel':
        """
        Fit the embedding-based topic model.
        
        Args:
            documents: List of text documents
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Self for chaining
        """
        if not documents:
            raise ValueError("No documents provided")
        
        # Step 1: Compute sentence embeddings
        if progress_callback:
            progress_callback("Computing sentence embeddings...")
        
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embeddings = self.embedding_model.encode(
            documents, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Step 2: Dimensionality reduction with UMAP
        if progress_callback:
            progress_callback("Reducing dimensionality with UMAP...")
        
        self.umap_model = umap.UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric='cosine',
            random_state=42
        )
        self.reduced_embeddings = self.umap_model.fit_transform(self.embeddings)
        
        # Step 3: Clustering with HDBSCAN
        if progress_callback:
            progress_callback("Clustering with HDBSCAN...")
        
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        self.topic_labels = self.hdbscan_model.fit_predict(self.reduced_embeddings)
        
        # Topic -1 is outliers/noise, count actual topics
        unique_topics = set(self.topic_labels)
        unique_topics.discard(-1)  # Remove noise label
        self.num_topics = len(unique_topics)
        
        # Step 4: Extract representative words using class-based TF-IDF
        if progress_callback:
            progress_callback("Extracting topic words...")
        
        self._extract_topic_words(documents)
        
        return self
    
    def _extract_topic_words(self, documents: List[str]):
        """
        Extract representative words for each topic using class-based TF-IDF.
        
        This treats all documents in a cluster as a single "class" and computes
        TF-IDF scores to find words that are distinctive to that cluster.
        """
        # Group documents by topic
        topic_docs = {}
        for doc, topic in zip(documents, self.topic_labels):
            if topic == -1:  # Skip outliers
                continue
            if topic not in topic_docs:
                topic_docs[topic] = []
            topic_docs[topic].append(doc)
        
        # For each topic, compute class-based TF-IDF
        for topic_id, docs in topic_docs.items():
            if not docs:
                continue
            
            # Combine all documents in this topic
            topic_text = ' '.join(docs)
            
            # Use CountVectorizer to get word frequencies
            vectorizer = CountVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams for more context
                min_df=2
            )
            
            try:
                # Fit on all topic documents
                all_topic_texts = [' '.join(topic_docs[tid]) for tid in topic_docs.keys()]
                vectorizer.fit(all_topic_texts)
                
                # Transform this topic's text
                topic_vector = vectorizer.transform([topic_text])
                
                # Get feature names and frequencies
                feature_names = vectorizer.get_feature_names_out()
                scores = topic_vector.toarray()[0]
                
                # Sort by frequency and get top words
                top_indices = scores.argsort()[-self.top_n_words:][::-1]
                top_words = [feature_names[i] for i in top_indices if scores[i] > 0]
                
                self.topic_words[topic_id] = top_words
            except Exception as e:
                # Fallback: just use most common words
                from collections import Counter
                words = topic_text.lower().split()
                word_counts = Counter(words)
                self.topic_words[topic_id] = [w for w, _ in word_counts.most_common(self.top_n_words)]
    
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
        
        for topic_id in sorted(self.topic_words.keys()):
            count = np.sum(self.topic_labels == topic_id)
            words = self.get_topic_words(topic_id, n=5)
            
            topic_info.append({
                'topic_id': topic_id,
                'top_words': ', '.join(words),
                'count': count
            })
        
        return pd.DataFrame(topic_info)


def train_embedding_topic_model(
    documents: List[str],
    min_cluster_size: int = 10,
    progress_callback=None
) -> Tuple[EmbeddingTopicModel, np.ndarray]:
    """
    Train an embedding-based topic model on documents.
    
    Args:
        documents: List of text documents
        min_cluster_size: Minimum size for a cluster to be considered a topic
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (fitted model, topic assignments per document)
    """
    # Check dependencies
    available, missing = check_dependencies()
    if not available:
        raise ImportError(
            f"Missing required packages: {', '.join(missing)}\n"
            "Install with: pip install sentence-transformers umap-learn hdbscan"
        )
    
    # Create and fit model
    model = EmbeddingTopicModel(
        embedding_model_name="all-MiniLM-L6-v2",  # Fast, good quality
        umap_n_components=5,
        umap_n_neighbors=15,
        umap_min_dist=0.0,
        hdbscan_min_cluster_size=min_cluster_size,
        hdbscan_min_samples=5,
        top_n_words=30  # Extract 30 words per topic for detailed descriptions
    )
    
    model.fit(documents, progress_callback=progress_callback)
    
    topic_assignments = model.get_topic_assignments()
    
    return model, topic_assignments

