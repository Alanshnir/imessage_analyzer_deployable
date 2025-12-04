"""Analytics functions for RQ1, RQ2, and RQ3."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from topics import train_lda, get_topic_words, infer_topic_distribution
from responses import compute_reluctance_score
from preprocess import preprocess_documents
from gensim import corpora


def rank_reluctant_topics(
    df_received_with_rt: pd.DataFrame,
    doc_topics: List[np.ndarray],
    model,
    dictionary,
    topic_threshold: float = 0.3,
    cap_minutes: float = 1440.0
) -> pd.DataFrame:
    """
    Rank topics by reluctance to respond (RQ1).
    
    Args:
        df_received_with_rt: DataFrame of received messages with response times
        doc_topics: List of topic distributions per document
        model: Trained LDA model
        dictionary: Gensim dictionary
        topic_threshold: Minimum topic probability to count a message
        cap_minutes: Cap for reluctance score calculation
        
    Returns:
        DataFrame with columns: topic_id, top_words, reluctance_score, frequency, examples
    """
    # Compute reluctance scores
    reluctance_scores = []
    for idx, row in df_received_with_rt.iterrows():
        score = compute_reluctance_score(
            row.get('response_time_min', np.nan),
            row.get('got_reply', False),
            cap_minutes
        )
        reluctance_scores.append(score)
    
    df_received_with_rt = df_received_with_rt.copy()
    df_received_with_rt['reluctance_score'] = reluctance_scores
    df_received_with_rt = df_received_with_rt.reset_index(drop=True)  # Ensure sequential index
    
    # Aggregate by topic
    num_topics = len(doc_topics[0]) if doc_topics else 0
    topic_stats = []
    
    for topic_id in range(num_topics):
        # Get topic words
        try:
            top_words_list = get_topic_words(model, dictionary, topic_id, num_words=30)
            top_words = ', '.join([word for word, _ in top_words_list])
        except:
            top_words = "N/A"
        
        # Weighted average reluctance
        weighted_reluctance = 0.0
        total_weight = 0.0
        frequency = 0
        example_texts = []
        
        # Iterate by position to match doc_topics order
        for doc_idx, (_, row) in enumerate(df_received_with_rt.iterrows()):
            if doc_idx < len(doc_topics):
                topic_prob = doc_topics[doc_idx][topic_id]
                if topic_prob > 0:
                    weight = topic_prob
                    weighted_reluctance += weight * row['reluctance_score']
                    total_weight += weight
                    
                    if topic_prob >= topic_threshold:
                        frequency += 1
                        if len(example_texts) < 3:
                            example_texts.append(str(row.get('text', ''))[:100])
        
        avg_reluctance = weighted_reluctance / total_weight if total_weight > 0 else 0.0
        
        # Final rank = reluctance * log(1 + frequency)
        final_rank = avg_reluctance * np.log1p(frequency)
        
        topic_stats.append({
            'topic_id': topic_id,
            'top_words': top_words,
            'reluctance_score': avg_reluctance,
            'frequency': frequency,
            'final_rank': final_rank,
            'examples': ' | '.join(example_texts[:3])
        })
    
    result_df = pd.DataFrame(topic_stats)
    result_df = result_df.sort_values('final_rank', ascending=False).reset_index(drop=True)
    
    return result_df


def get_reluctant_topic_examples(
    df_received_with_rt: pd.DataFrame,
    doc_topics: List[np.ndarray],
    top_topic_ids: List[int],
    num_examples: int = 5,
    topic_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Get example messages with highest prevalence of reluctant topics.
    
    Args:
        df_received_with_rt: DataFrame of received messages with response times
        doc_topics: List of topic distributions per document
        top_topic_ids: List of topic IDs to find examples for
        num_examples: Number of examples per topic
        topic_threshold: Minimum topic probability to include
        
    Returns:
        DataFrame with columns: topic_id, text, participant, chat_name, topic_prob, reluctance_score
    """
    df_received_with_rt = df_received_with_rt.reset_index(drop=True)
    examples = []
    
    for topic_id in top_topic_ids:
        topic_messages = []
        
        for doc_idx, (_, row) in enumerate(df_received_with_rt.iterrows()):
            if doc_idx < len(doc_topics):
                topic_prob = doc_topics[doc_idx][topic_id]
                if topic_prob >= topic_threshold:
                    # Prefer chat_name over participant for display
                    contact_name = str(row.get('chat_name', ''))
                    if not contact_name or contact_name == 'Unknown Chat':
                        contact_name = str(row.get('participant', 'Unknown'))
                    
                    topic_messages.append({
                        'topic_id': topic_id,
                        'text': str(row.get('text', ''))[:200],  # Limit text length
                        'participant': contact_name,
                        'chat_name': str(row.get('chat_name', 'Unknown Chat')),
                        'topic_prob': topic_prob,
                        'reluctance_score': row.get('reluctance_score', 0.0)
                    })
        
        # Sort by reluctance score (highest first) to show messages you were most reluctant to respond to
        # Secondary sort by topic probability to ensure relevance to the topic
        topic_messages.sort(key=lambda x: (x['reluctance_score'], x['topic_prob']), reverse=True)
        examples.extend(topic_messages[:num_examples])
    
    return pd.DataFrame(examples)


def get_contact_reluctant_topic_stats(
    df_received_with_rt: pd.DataFrame,
    doc_topics: List[np.ndarray],
    top_topic_ids: List[int],
    topic_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Get statistics about reluctant topics per contact.
    
    Args:
        df_received_with_rt: DataFrame of received messages with response times
        doc_topics: List of topic distributions per document
        top_topic_ids: List of top reluctant topic IDs
        topic_threshold: Minimum topic probability to count as high prevalence
        
    Returns:
        DataFrame with columns: contact, high_reluctant_topic_count, avg_reluctance_score
    """
    df_received_with_rt = df_received_with_rt.reset_index(drop=True)
    
    # Prefer chat_name over participant for display
    def get_contact_name(row):
        contact_name = str(row.get('chat_name', ''))
        if not contact_name or contact_name == 'Unknown Chat':
            contact_name = str(row.get('participant', 'Unknown'))
        return contact_name
    
    df_received_with_rt['contact'] = df_received_with_rt.apply(get_contact_name, axis=1)
    
    contact_stats = {}
    
    for doc_idx, (_, row) in enumerate(df_received_with_rt.iterrows()):
        if doc_idx < len(doc_topics):
            contact = row['contact']
            reluctance_score = row.get('reluctance_score', 0.0)
            
            if contact not in contact_stats:
                contact_stats[contact] = {
                    'high_reluctant_topic_count': 0,
                    'total_reluctance': 0.0,
                    'message_count': 0
                }
            
            # Check if message has high prevalence of any reluctant topic
            has_reluctant_topic = False
            for topic_id in top_topic_ids:
                if doc_idx < len(doc_topics):
                    topic_prob = doc_topics[doc_idx][topic_id]
                    if topic_prob >= topic_threshold:
                        has_reluctant_topic = True
                        break
            
            if has_reluctant_topic:
                contact_stats[contact]['high_reluctant_topic_count'] += 1
            
            contact_stats[contact]['total_reluctance'] += reluctance_score
            contact_stats[contact]['message_count'] += 1
    
    # Convert to DataFrame
    results = []
    for contact, stats in contact_stats.items():
        avg_reluctance = stats['total_reluctance'] / stats['message_count'] if stats['message_count'] > 0 else 0.0
        results.append({
            'contact': contact,
            'high_reluctant_topic_count': stats['high_reluctant_topic_count'],
            'avg_reluctance_score': avg_reluctance,
            'total_messages': stats['message_count']
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('high_reluctant_topic_count', ascending=False)
    
    return result_df


def topic_over_time(
    df: pd.DataFrame,
    doc_topics: List[np.ndarray],
    freq: str = 'M'
) -> pd.DataFrame:
    """
    Compute topic prevalence over time (RQ2).
    
    Args:
        df: DataFrame with messages and timestamps
        doc_topics: List of topic distributions per document
        freq: Time frequency ('M' for monthly, 'Y' for yearly)
        
    Returns:
        DataFrame with columns: period, topic_id, prevalence
    """
    if 'timestamp_local' not in df.columns or df['timestamp_local'].isna().all():
        return pd.DataFrame()
    
    df = df.copy()
    df = df.reset_index(drop=True)  # Ensure sequential index
    df['period'] = pd.to_datetime(df['timestamp_local']).dt.to_period(freq)
    
    num_topics = len(doc_topics[0]) if doc_topics else 0
    results = []
    
    for period in df['period'].dropna().unique():
        period_mask = df['period'] == period
        period_positions = df[period_mask].index.tolist()
        
        # Average topic distribution for this period
        period_topics = np.zeros(num_topics)
        count = 0
        
        for doc_idx in period_positions:
            if doc_idx < len(doc_topics):
                period_topics += doc_topics[doc_idx]
                count += 1
        
        if count > 0:
            period_topics /= count
            
            for topic_id in range(num_topics):
                results.append({
                    'period': str(period),
                    'topic_id': topic_id,
                    'prevalence': period_topics[topic_id]
                })
    
    return pd.DataFrame(results)


def get_per_contact_topic_mix(
    df: pd.DataFrame,
    doc_topics: List[np.ndarray],
    top_n_contacts: int = 10
) -> pd.DataFrame:
    """
    Get topic distribution per contact (RQ2).
    
    Args:
        df: DataFrame with messages
        doc_topics: List of topic distributions per document
        top_n_contacts: Number of top contacts to include
        
    Returns:
        DataFrame with columns: participant, topic_id, proportion
    """
    # Get top contacts by message count
    contact_counts = df[df['direction'] == 'in']['participant'].value_counts()
    top_contacts = contact_counts.head(top_n_contacts).index.tolist()
    
    num_topics = len(doc_topics[0]) if doc_topics else 0
    results = []
    
    # Ensure df and doc_topics are aligned
    df_aligned = df.reset_index(drop=True)
    
    for contact in top_contacts:
        contact_mask = (df_aligned['participant'] == contact) & (df_aligned['direction'] == 'in')
        contact_positions = df_aligned[contact_mask].index.tolist()
        
        # Average topic distribution for this contact
        contact_topics = np.zeros(num_topics)
        count = 0
        
        for doc_idx in contact_positions:
            if doc_idx < len(doc_topics):
                contact_topics += doc_topics[doc_idx]
                count += 1
        
        if count > 0:
            contact_topics /= count
            
            for topic_id in range(num_topics):
                results.append({
                    'participant': contact,
                    'topic_id': topic_id,
                    'proportion': contact_topics[topic_id]
                })
    
    return pd.DataFrame(results)


def group_vs_dm_stats(df_received_with_rt: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute group vs one-to-one responsiveness statistics (RQ3).
    
    Args:
        df_received_with_rt: DataFrame of received messages with response times and group_size
        
    Returns:
        Dictionary with statistics
    """
    df = df_received_with_rt.copy()
    
    # Categorize by group size
    def categorize_group_size(size):
        if pd.isna(size) or size < 2:
            return "Unknown"
        elif size == 2:
            return "One-to-One"
        elif size <= 4:
            return "Small (3-4)"
        elif size <= 8:
            return "Medium (5-8)"
        else:
            return "Large (9+)"
    
    df['group_category'] = df['group_size'].apply(categorize_group_size)
    
    stats = {}
    
    # Overall stats
    stats['overall'] = {
        'total_received': len(df),
        'reply_rate': df['got_reply'].mean(),
        'median_response_time': df['response_time_min'].median(),
        'mean_response_time': df['response_time_min'].mean()
    }
    
    # Per category stats
    category_stats = []
    for category in df['group_category'].unique():
        cat_df = df[df['group_category'] == category]
        category_stats.append({
            'category': category,
            'count': len(cat_df),
            'reply_rate': cat_df['got_reply'].mean(),
            'median_response_time': cat_df['response_time_min'].median(),
            'mean_response_time': cat_df['response_time_min'].mean(),
            'p25_response_time': cat_df['response_time_min'].quantile(0.25),
            'p75_response_time': cat_df['response_time_min'].quantile(0.75)
        })
    
    stats['by_category'] = pd.DataFrame(category_stats)
    
    # Identify contacts only responded to in groups
    contact_stats = []
    for contact in df['participant'].unique():
        contact_df = df[df['participant'] == contact]
        one_on_one = contact_df[contact_df['group_size'] == 2]
        group = contact_df[contact_df['group_size'] > 2]
        
        one_on_one_rate = one_on_one['got_reply'].mean() if len(one_on_one) > 0 else 0
        group_rate = group['got_reply'].mean() if len(group) > 0 else 0
        
        contact_stats.append({
            'participant': contact,
            'one_on_one_reply_rate': one_on_one_rate,
            'group_reply_rate': group_rate,
            'one_on_one_count': len(one_on_one),
            'group_count': len(group),
            'prefers_group': group_rate > one_on_one_rate + 0.2 and len(one_on_one) > 5
        })
    
    stats['contact_comparison'] = pd.DataFrame(contact_stats)
    
    return stats


def get_topic_message_counts(
    doc_topics: List[np.ndarray],
    topic_threshold: float = 0.3
) -> Dict[int, int]:
    """
    Count messages per topic (RQ2 enhancement).
    
    Args:
        doc_topics: List of topic distributions per document
        topic_threshold: Minimum topic probability to count a message
        
    Returns:
        Dictionary mapping topic_id to message count
    """
    num_topics = len(doc_topics[0]) if doc_topics else 0
    topic_counts = {i: 0 for i in range(num_topics)}
    
    for doc_topic in doc_topics:
        for topic_id in range(num_topics):
            if doc_topic[topic_id] >= topic_threshold:
                topic_counts[topic_id] += 1
    
    return topic_counts


def get_contact_reluctance_proportions(
    df_received_with_rt: pd.DataFrame,
    doc_topics: List[np.ndarray],
    top_topic_ids: List[int],
    topic_threshold: float = 0.3,
    min_messages: int = 500
) -> pd.DataFrame:
    """
    Get contacts by proportion of high reluctance messages (RQ1 enhancement).
    
    Args:
        df_received_with_rt: DataFrame of received messages with response times
        doc_topics: List of topic distributions per document
        top_topic_ids: List of top reluctant topic IDs
        topic_threshold: Minimum topic probability to count as high prevalence
        min_messages: Minimum total messages to include contact
        
    Returns:
        DataFrame with columns: contact, high_reluctance_proportion, total_messages, high_reluctance_count
    """
    df_received_with_rt = df_received_with_rt.reset_index(drop=True)
    
    # Prefer chat_name over participant for display
    def get_contact_name(row):
        contact_name = str(row.get('chat_name', ''))
        if not contact_name or contact_name == 'Unknown Chat':
            contact_name = str(row.get('participant', 'Unknown'))
        return contact_name
    
    df_received_with_rt['contact'] = df_received_with_rt.apply(get_contact_name, axis=1)
    
    contact_stats = {}
    
    for doc_idx, (_, row) in enumerate(df_received_with_rt.iterrows()):
        if doc_idx < len(doc_topics):
            contact = row['contact']
            
            if contact not in contact_stats:
                contact_stats[contact] = {
                    'high_reluctance_count': 0,
                    'total_messages': 0
                }
            
            # Check if message has high prevalence of any reluctant topic
            has_reluctant_topic = False
            for topic_id in top_topic_ids:
                if doc_idx < len(doc_topics):
                    topic_prob = doc_topics[doc_idx][topic_id]
                    if topic_prob >= topic_threshold:
                        has_reluctant_topic = True
                        break
            
            if has_reluctant_topic:
                contact_stats[contact]['high_reluctance_count'] += 1
            
            contact_stats[contact]['total_messages'] += 1
    
    # Convert to DataFrame and filter
    results = []
    for contact, stats in contact_stats.items():
        if stats['total_messages'] >= min_messages:
            proportion = stats['high_reluctance_count'] / stats['total_messages']
            results.append({
                'contact': contact,
                'high_reluctance_proportion': proportion,
                'total_messages': stats['total_messages'],
                'high_reluctance_count': stats['high_reluctance_count']
            })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('high_reluctance_proportion', ascending=False)
    
    return result_df


def analyze_conversation_starters(
    df_received_with_rt: pd.DataFrame,
    doc_topics: List[np.ndarray],
    topic_threshold: float = 0.3,
    session_gap_minutes: float = 180.0
) -> pd.DataFrame:
    """
    Identify conversation starter topics (RQ4).
    
    Args:
        df_received_with_rt: DataFrame of received messages with response times
        doc_topics: List of topic distributions per document
        topic_threshold: Minimum topic probability to count
        session_gap_minutes: Gap to consider as new conversation session
        
    Returns:
        DataFrame with columns: topic_id, reply_rate, avg_response_time, starter_probability
    """
    df = df_received_with_rt.copy().reset_index(drop=True)
    df = df.sort_values(['chat_id', 'timestamp_local'])
    
    num_topics = len(doc_topics[0]) if doc_topics else 0
    topic_stats = {i: {'replies': 0, 'total': 0, 'response_times': [], 'is_starter': 0} for i in range(num_topics)}
    
    # Identify session starters
    df['is_starter'] = False
    for chat_id in df['chat_id'].unique():
        chat_mask = df['chat_id'] == chat_id
        chat_df = df[chat_mask].copy()
        
        if len(chat_df) == 0:
            continue
        
        # First message is always a starter
        df.loc[chat_df.index[0], 'is_starter'] = True
        
        # Check time gaps
        for i in range(1, len(chat_df)):
            time_gap = (chat_df.iloc[i]['timestamp_local'] - chat_df.iloc[i-1]['timestamp_local']).total_seconds() / 60.0
            if time_gap >= session_gap_minutes:
                df.loc[chat_df.index[i], 'is_starter'] = True
    
    # Aggregate by topic
    for doc_idx, (_, row) in enumerate(df.iterrows()):
        if doc_idx < len(doc_topics):
            for topic_id in range(num_topics):
                topic_prob = doc_topics[doc_idx][topic_id]
                if topic_prob >= topic_threshold:
                    topic_stats[topic_id]['total'] += 1
                    
                    if row.get('got_reply', False):
                        topic_stats[topic_id]['replies'] += 1
                    
                    if pd.notna(row.get('response_time_min')):
                        topic_stats[topic_id]['response_times'].append(row['response_time_min'])
                    
                    if row['is_starter']:
                        topic_stats[topic_id]['is_starter'] += 1
    
    # Convert to DataFrame
    results = []
    for topic_id, stats in topic_stats.items():
        reply_rate = stats['replies'] / stats['total'] if stats['total'] > 0 else 0
        avg_response_time = np.mean(stats['response_times']) if stats['response_times'] else np.nan
        starter_prob = stats['is_starter'] / stats['total'] if stats['total'] > 0 else 0
        
        results.append({
            'topic_id': topic_id,
            'reply_rate': reply_rate,
            'avg_response_time': avg_response_time,
            'starter_probability': starter_prob,
            'message_count': stats['total']
        })
    
    result_df = pd.DataFrame(results)
    # Rank by combination of high reply rate, short response time, and high starter probability
    result_df['starter_score'] = (
        result_df['reply_rate'] * 0.4 +
        (1 - result_df['avg_response_time'].fillna(1440) / 1440) * 0.3 +
        result_df['starter_probability'] * 0.3
    )
    result_df = result_df.sort_values('starter_score', ascending=False)
    
    return result_df


def analyze_conversation_enders(
    df_received_with_rt: pd.DataFrame,
    doc_topics: List[np.ndarray],
    topic_threshold: float = 0.3,
    session_gap_minutes: float = 180.0
) -> pd.DataFrame:
    """
    Identify conversation ender topics (RQ5).
    
    Args:
        df_received_with_rt: DataFrame of received messages with response times
        doc_topics: List of topic distributions per document
        topic_threshold: Minimum topic probability to count
        session_gap_minutes: Gap to consider as end of conversation
        
    Returns:
        DataFrame with columns: topic_id, no_reply_rate, avg_response_time, ender_probability
    """
    df = df_received_with_rt.copy().reset_index(drop=True)
    df = df.sort_values(['chat_id', 'timestamp_local'])
    
    num_topics = len(doc_topics[0]) if doc_topics else 0
    topic_stats = {i: {'no_replies': 0, 'total': 0, 'response_times': [], 'is_ender': 0} for i in range(num_topics)}
    
    # Identify session enders
    df['is_ender'] = False
    for chat_id in df['chat_id'].unique():
        chat_mask = df['chat_id'] == chat_id
        chat_df = df[chat_mask].copy()
        
        if len(chat_df) == 0:
            continue
        
        # Last message is always an ender
        df.loc[chat_df.index[-1], 'is_ender'] = True
        
        # Check time gaps
        for i in range(len(chat_df) - 1):
            time_gap = (chat_df.iloc[i+1]['timestamp_local'] - chat_df.iloc[i]['timestamp_local']).total_seconds() / 60.0
            if time_gap >= session_gap_minutes:
                df.loc[chat_df.index[i], 'is_ender'] = True
    
    # Aggregate by topic
    for doc_idx, (_, row) in enumerate(df.iterrows()):
        if doc_idx < len(doc_topics):
            for topic_id in range(num_topics):
                topic_prob = doc_topics[doc_idx][topic_id]
                if topic_prob >= topic_threshold:
                    topic_stats[topic_id]['total'] += 1
                    
                    if not row.get('got_reply', False):
                        topic_stats[topic_id]['no_replies'] += 1
                    
                    if pd.notna(row.get('response_time_min')):
                        topic_stats[topic_id]['response_times'].append(row['response_time_min'])
                    
                    if row['is_ender']:
                        topic_stats[topic_id]['is_ender'] += 1
    
    # Convert to DataFrame
    results = []
    for topic_id, stats in topic_stats.items():
        no_reply_rate = stats['no_replies'] / stats['total'] if stats['total'] > 0 else 0
        avg_response_time = np.mean(stats['response_times']) if stats['response_times'] else np.nan
        ender_prob = stats['is_ender'] / stats['total'] if stats['total'] > 0 else 0
        
        results.append({
            'topic_id': topic_id,
            'no_reply_rate': no_reply_rate,
            'avg_response_time': avg_response_time,
            'ender_probability': ender_prob,
            'message_count': stats['total']
        })
    
    result_df = pd.DataFrame(results)
    # Rank by combination of high no-reply rate, long response time, and high ender probability
    result_df['ender_score'] = (
        result_df['no_reply_rate'] * 0.4 +
        (result_df['avg_response_time'].fillna(0) / 1440) * 0.3 +
        result_df['ender_probability'] * 0.3
    )
    result_df = result_df.sort_values('ender_score', ascending=False)
    
    return result_df


def analyze_topics_by_closeness(
    df_received_with_rt: pd.DataFrame,
    doc_topics: List[np.ndarray],
    topic_threshold: float = 0.3,
    close_contact_min_messages: int = 100,
    close_contact_min_reply_rate: float = 0.5
) -> pd.DataFrame:
    """
    Compare topics between close contacts and acquaintances (RQ6).
    
    Args:
        df_received_with_rt: DataFrame of received messages with response times
        doc_topics: List of topic distributions per document
        topic_threshold: Minimum topic probability to count
        close_contact_min_messages: Minimum messages to be considered close
        close_contact_min_reply_rate: Minimum reply rate to be considered close
        
    Returns:
        DataFrame with columns: topic_id, close_freq, acquaintance_freq, odds_ratio, category
    """
    df = df_received_with_rt.copy().reset_index(drop=True)
    
    # Classify contacts as close or acquaintances
    def get_contact_name(row):
        contact_name = str(row.get('chat_name', ''))
        if not contact_name or contact_name == 'Unknown Chat':
            contact_name = str(row.get('participant', 'Unknown'))
        return contact_name
    
    df['contact'] = df.apply(get_contact_name, axis=1)
    
    contact_stats = df.groupby('contact').agg({
        'message_id': 'count',
        'got_reply': 'mean'
    }).rename(columns={'message_id': 'message_count', 'got_reply': 'reply_rate'})
    
    close_contacts = contact_stats[
        (contact_stats['message_count'] >= close_contact_min_messages) &
        (contact_stats['reply_rate'] >= close_contact_min_reply_rate)
    ].index.tolist()
    
    df['is_close'] = df['contact'].isin(close_contacts)
    
    # Aggregate topics by closeness
    num_topics = len(doc_topics[0]) if doc_topics else 0
    close_counts = {i: 0 for i in range(num_topics)}
    acquaintance_counts = {i: 0 for i in range(num_topics)}
    close_total = 0
    acquaintance_total = 0
    
    for doc_idx, (_, row) in enumerate(df.iterrows()):
        if doc_idx < len(doc_topics):
            is_close = row['is_close']
            
            for topic_id in range(num_topics):
                topic_prob = doc_topics[doc_idx][topic_id]
                if topic_prob >= topic_threshold:
                    if is_close:
                        close_counts[topic_id] += 1
                    else:
                        acquaintance_counts[topic_id] += 1
            
            if is_close:
                close_total += 1
            else:
                acquaintance_total += 1
    
    # Calculate odds ratios
    results = []
    for topic_id in range(num_topics):
        close_freq = close_counts[topic_id] / close_total if close_total > 0 else 0
        acquaintance_freq = acquaintance_counts[topic_id] / acquaintance_total if acquaintance_total > 0 else 0
        
        # Odds ratio with small constant to avoid division by zero
        if acquaintance_freq > 0:
            odds_ratio = close_freq / acquaintance_freq
        else:
            odds_ratio = float('inf') if close_freq > 0 else 1.0
        
        # Categorize
        if odds_ratio > 1.5:
            category = 'Close contacts'
        elif odds_ratio < 0.67:
            category = 'Acquaintances'
        else:
            category = 'Neutral'
        
        results.append({
            'topic_id': topic_id,
            'close_frequency': close_freq,
            'acquaintance_frequency': acquaintance_freq,
            'odds_ratio': min(odds_ratio, 10.0),  # Cap for visualization
            'category': category,
            'close_count': close_counts[topic_id],
            'acquaintance_count': acquaintance_counts[topic_id]
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('odds_ratio', ascending=False)
    
    return result_df


def analyze_topics_by_time_of_day(
    df_received_with_rt: pd.DataFrame,
    doc_topics: List[np.ndarray],
    topic_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Analyze topics by time of day (RQ7).
    
    Args:
        df_received_with_rt: DataFrame of received messages with timestamps
        doc_topics: List of topic distributions per document
        topic_threshold: Minimum topic probability to count
        
    Returns:
        DataFrame with columns: topic_id, morning, afternoon, evening, night
    """
    df = df_received_with_rt.copy().reset_index(drop=True)
    
    # Extract hour of day
    df['hour'] = pd.to_datetime(df['timestamp_local']).dt.hour
    
    # Define time periods
    def get_time_period(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
    
    df['time_period'] = df['hour'].apply(get_time_period)
    
    # Aggregate topics by time period
    num_topics = len(doc_topics[0]) if doc_topics else 0
    time_periods = ['Morning', 'Afternoon', 'Evening', 'Night']
    topic_by_time = {period: {i: 0 for i in range(num_topics)} for period in time_periods}
    time_totals = {period: 0 for period in time_periods}
    
    for doc_idx, (_, row) in enumerate(df.iterrows()):
        if doc_idx < len(doc_topics) and pd.notna(row.get('time_period')):
            time_period = row['time_period']
            
            for topic_id in range(num_topics):
                topic_prob = doc_topics[doc_idx][topic_id]
                if topic_prob >= topic_threshold:
                    topic_by_time[time_period][topic_id] += 1
            
            time_totals[time_period] += 1
    
    # Calculate frequencies
    results = []
    for topic_id in range(num_topics):
        row_data = {'topic_id': topic_id}
        for period in time_periods:
            freq = topic_by_time[period][topic_id] / time_totals[period] if time_totals[period] > 0 else 0
            row_data[period.lower()] = freq
        results.append(row_data)
    
    result_df = pd.DataFrame(results)
    
    return result_df


def analyze_high_response_topics(
    df_received_with_rt: pd.DataFrame,
    doc_topics: List[np.ndarray],
    topic_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Identify topics with highest response likelihood (RQ8).
    
    Args:
        df_received_with_rt: DataFrame of received messages with response times
        doc_topics: List of topic distributions per document
        topic_threshold: Minimum topic probability to count
        
    Returns:
        DataFrame with columns: topic_id, reply_rate, avg_response_time, message_count
    """
    df = df_received_with_rt.copy().reset_index(drop=True)
    
    num_topics = len(doc_topics[0]) if doc_topics else 0
    topic_stats = {i: {'replies': 0, 'total': 0, 'response_times': []} for i in range(num_topics)}
    
    # Aggregate by topic
    for doc_idx, (_, row) in enumerate(df.iterrows()):
        if doc_idx < len(doc_topics):
            for topic_id in range(num_topics):
                topic_prob = doc_topics[doc_idx][topic_id]
                if topic_prob >= topic_threshold:
                    topic_stats[topic_id]['total'] += 1
                    
                    if row.get('got_reply', False):
                        topic_stats[topic_id]['replies'] += 1
                    
                    if pd.notna(row.get('response_time_min')):
                        topic_stats[topic_id]['response_times'].append(row['response_time_min'])
    
    # Convert to DataFrame
    results = []
    for topic_id, stats in topic_stats.items():
        reply_rate = stats['replies'] / stats['total'] if stats['total'] > 0 else 0
        avg_response_time = np.mean(stats['response_times']) if stats['response_times'] else np.nan
        
        results.append({
            'topic_id': topic_id,
            'reply_rate': reply_rate,
            'avg_response_time': avg_response_time,
            'message_count': stats['total']
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('reply_rate', ascending=False)
    
    return result_df



def rank_reluctant_topics_embedding(
    df_received_with_rt: pd.DataFrame,
    topic_labels: np.ndarray,
    embedding_model,
    cap_minutes: float = 1440.0
) -> pd.DataFrame:
    """
    Rank topics by reluctance to respond using embedding-based topics (RQ8).
    
    This is the fine-grained version of RQ1 using embedding-based topic modeling.
    Uses the same reluctance logic as RQ1, but with BERT-style embedding topics.
    
    Args:
        df_received_with_rt: DataFrame of received messages with response times
        topic_labels: Array of topic assignments from embedding model (one per message)
        embedding_model: Fitted EmbeddingTopicModel instance
        cap_minutes: Cap for reluctance score calculation
        
    Returns:
        DataFrame with columns: topic_id, top_words, reluctance_score, frequency, final_rank
    """
    from responses import compute_reluctance_score
    
    # Compute reluctance scores
    reluctance_scores = []
    for idx, row in df_received_with_rt.iterrows():
        score = compute_reluctance_score(
            row.get('response_time_min', np.nan),
            row.get('got_reply', False),
            cap_minutes
        )
        reluctance_scores.append(score)
    
    df_received_with_rt = df_received_with_rt.copy()
    df_received_with_rt['reluctance_score'] = reluctance_scores
    df_received_with_rt = df_received_with_rt.reset_index(drop=True)
    
    # Get unique topics (excluding noise label -1)
    unique_topics = sorted([t for t in set(topic_labels) if t != -1])
    
    topic_stats = []
    
    for topic_id in unique_topics:
        # Get top words for this topic
        top_words_list = embedding_model.get_topic_words(topic_id, n=30)
        top_words = ', '.join(top_words_list)
        
        # Calculate weighted average reluctance for this topic
        topic_mask = topic_labels == topic_id
        topic_messages = df_received_with_rt[topic_mask]
        
        if len(topic_messages) == 0:
            continue
        
        # Simple average (all messages in cluster have equal weight)
        avg_reluctance = topic_messages['reluctance_score'].mean()
        frequency = len(topic_messages)
        
        # Final rank = reluctance * log(1 + frequency)
        # Same formula as RQ1 for consistency
        final_rank = avg_reluctance * np.log1p(frequency)
        
        topic_stats.append({
            'topic_id': topic_id,
            'top_words': top_words,
            'reluctance_score': avg_reluctance,
            'frequency': frequency,
            'final_rank': final_rank
        })
    
    result_df = pd.DataFrame(topic_stats)
    result_df = result_df.sort_values('final_rank', ascending=False).reset_index(drop=True)
    
    return result_df
