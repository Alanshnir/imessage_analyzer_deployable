"""Topic modeling functions using Gensim LDA and optional MALLET."""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models import CoherenceModel
from utils import get_mallet_home


def train_lda(
    docs: List[List[str]],
    num_topics: int = 10,
    passes: int = 10,
    alpha: str = 'auto',
    beta: float = 0.01,
    use_mallet: bool = False,
    mallet_home: Optional[str] = None,
    random_state: int = 42
) -> Tuple[models.LdaModel, corpora.Dictionary, List, List]:
    """
    Train an LDA topic model on documents.
    
    Args:
        docs: List of tokenized documents (each doc is a list of tokens)
        num_topics: Number of topics (K)
        passes: Number of passes through the corpus
        alpha: Prior for document-topic distribution ('auto', 'symmetric', or float)
        beta: Prior for topic-word distribution
        use_mallet: Whether to use MALLET LDA
        mallet_home: Path to MALLET home directory
        random_state: Random seed
        
    Returns:
        Tuple of (model, dictionary, corpus, doc_topics)
        where doc_topics is a list of topic distributions per document
    """
    if not docs or len(docs) == 0:
        raise ValueError("No documents provided for topic modeling")
    
    # Create dictionary
    dictionary = corpora.Dictionary(docs)
    
    # Filter extremes
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    
    # Create corpus
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    
    if use_mallet and mallet_home:
        # Use MALLET LDA
        try:
            mallet_path = f"{mallet_home}/bin/mallet"
            if not os.path.exists(mallet_path):
                mallet_path = f"{mallet_path}.bat"
            
            model = models.wrappers.LdaMallet(
                mallet_path,
                corpus=corpus,
                num_topics=num_topics,
                id2word=dictionary,
                alpha=alpha,
                beta=beta,
                iterations=passes * 50,  # MALLET uses iterations
                random_seed=random_state
            )
        except Exception as e:
            print(f"Warning: MALLET failed, falling back to Gensim LDA: {e}")
            use_mallet = False
    
    if not use_mallet:
        # Use Gensim LDA
        model = models.LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=passes,
            alpha=alpha,
            eta=beta,
            random_state=random_state,
            per_word_topics=True
        )
    
    # Get topic distributions for each document
    doc_topics = []
    for doc_bow in corpus:
        topic_dist = model.get_document_topics(doc_bow, minimum_probability=0.0)
        # Convert to array format
        topic_array = np.zeros(num_topics)
        for topic_id, prob in topic_dist:
            topic_array[topic_id] = prob
        doc_topics.append(topic_array)
    
    return model, dictionary, corpus, doc_topics


def get_topic_words(model: models.LdaModel, dictionary: corpora.Dictionary, 
                    topic_id: int, num_words: int = 10) -> List[Tuple[str, float]]:
    """
    Get top words for a topic.
    
    Args:
        model: Trained LDA model
        dictionary: Gensim dictionary
        topic_id: Topic ID
        num_words: Number of top words to return
        
    Returns:
        List of (word, probability) tuples
    """
    if hasattr(model, 'show_topic'):
        return model.show_topic(topic_id, topn=num_words)
    else:
        # MALLET wrapper
        return model.show_topic(topic_id, topn=num_words)


def compute_topic_coherence(model: models.LdaModel, dictionary: corpora.Dictionary,
                           corpus: List, docs: List[List[str]], 
                           coherence: str = 'c_v') -> float:
    """
    Compute topic coherence score.
    
    Args:
        model: Trained LDA model
        dictionary: Gensim dictionary
        corpus: Gensim corpus
        docs: Original tokenized documents
        coherence: Coherence measure ('c_v', 'u_mass', 'c_uci', 'c_npmi')
        
    Returns:
        Coherence score
    """
    try:
        coherence_model = CoherenceModel(
            model=model,
            texts=docs,
            dictionary=dictionary,
            corpus=corpus,
            coherence=coherence
        )
        return coherence_model.get_coherence()
    except Exception as e:
        print(f"Warning: Could not compute coherence: {e}")
        return 0.0


def infer_topic_distribution(model: models.LdaModel, doc_bow: List) -> np.ndarray:
    """
    Infer topic distribution for a single document.
    
    Args:
        model: Trained LDA model
        doc_bow: Document as bag-of-words
        
    Returns:
        Array of topic probabilities
    """
    topic_dist = model.get_document_topics(doc_bow, minimum_probability=0.0)
    num_topics = model.num_topics
    topic_array = np.zeros(num_topics)
    for topic_id, prob in topic_dist:
        topic_array[topic_id] = prob
    return topic_array


import os

