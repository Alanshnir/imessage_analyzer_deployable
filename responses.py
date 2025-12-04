"""Response time calculations and analysis."""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_response_times(
    df: pd.DataFrame,
    max_gap_minutes: float = 1440.0,  # 24 hours
    remove_outliers: bool = True,
    outlier_percentile: float = 99.0,
    count_tapbacks_as_replies: bool = False
) -> pd.DataFrame:
    """
    Compute response times for received messages with Tapback and threaded reply support.
    
    Args:
        df: DataFrame with messages (must have direction, timestamp_local, chat_id, guid, 
            associated_message_guid, associated_message_type, thread_originator_guid, reply_to_guid)
        max_gap_minutes: Maximum gap to consider as a response (minutes)
        remove_outliers: Whether to remove outliers
        outlier_percentile: Percentile threshold for outlier removal
        count_tapbacks_as_replies: If True, Tapbacks count as valid replies. If False, ignore them.
        
    Returns:
        DataFrame with added columns: response_time_min, got_reply, got_tapback, tapback_type, 
                                       reply_message_id
    """
    df = df.copy()
    df = df.sort_values(['chat_id', 'timestamp_local']).reset_index(drop=True)
    
    # Initialize response time columns
    df['response_time_min'] = np.nan
    df['got_reply'] = False
    df['got_tapback'] = False
    df['tapback_type'] = None
    df['reply_message_id'] = None
    
    # Tapback type mapping (associated_message_type values)
    TAPBACK_TYPES = {
        2000: 'Loved',
        2001: 'Liked',
        2002: 'Disliked',
        2003: 'Laughed',
        2004: 'Emphasized',
        2005: 'Questioned',
        3000: 'Removed Loved',
        3001: 'Removed Liked',
        3002: 'Removed Disliked',
        3003: 'Removed Laughed',
        3004: 'Removed Emphasized',
        3005: 'Removed Questioned'
    }
    
    # Identify Tapbacks: messages where associated_message_guid is not null
    if 'associated_message_guid' in df.columns and 'associated_message_type' in df.columns:
        tapback_mask = df['associated_message_guid'].notna()
        df.loc[tapback_mask, 'tapback_type'] = df.loc[tapback_mask, 'associated_message_type'].map(TAPBACK_TYPES)
    
    # Build a GUID to index mapping for fast lookups
    if 'guid' in df.columns:
        guid_to_idx = {guid: idx for idx, guid in enumerate(df['guid']) if pd.notna(guid)}
    else:
        guid_to_idx = {}
    
    # Process each chat separately - ULTRA OPTIMIZED VERSION
    import time
    unique_chats = df['chat_id'].dropna().unique()
    total_chats = len(unique_chats)
    print(f"\n⚙️  Computing response times for {total_chats:,} chats...")
    
    start_time = time.time()
    processed_messages = 0
    
    # Collect results in bulk instead of updating df repeatedly
    response_times = {}  # original_index -> response_time_min
    got_replies = {}     # original_index -> got_reply
    got_tapbacks = {}    # original_index -> got_tapback
    tapback_types = {}   # original_index -> tapback_type
    reply_msg_ids = {}   # original_index -> reply_message_id
    
    for chat_idx, chat_id in enumerate(unique_chats):
        chat_df = df[df['chat_id'] == chat_id].copy()
        chat_df = chat_df.sort_values('timestamp_local').reset_index()
        
        # Convert timestamps to numpy arrays for faster operations
        timestamps = chat_df['timestamp_local'].values
        directions = chat_df['direction'].values
        original_indices = chat_df['index'].values  # Original df indices
        
        # OPTIMIZATION: Pre-filter to only incoming messages
        incoming_mask = directions == 'in'
        incoming_positions = np.where(incoming_mask)[0]
        
        # OPTIMIZATION: Pre-filter outgoing messages (potential replies)
        if count_tapbacks_as_replies:
            outgoing_mask = directions == 'out'
        else:
            # Exclude Tapbacks from potential replies
            has_assoc_guid = chat_df['associated_message_guid'].notna().values
            outgoing_mask = (directions == 'out') & (~has_assoc_guid)
        
        outgoing_positions = np.where(outgoing_mask)[0]
        outgoing_timestamps = timestamps[outgoing_mask]
        
        # OPTIMIZATION: Pre-build lookup dictionaries for Tapbacks and threaded replies ONCE per chat
        # This avoids expensive pandas filtering for every incoming message
        tapback_lookup = {}  # guid -> [(timestamp, message_id, tapback_type), ...]
        threaded_lookup = {}  # guid -> [(timestamp, message_id), ...]
        
        if count_tapbacks_as_replies or 'associated_message_guid' in chat_df.columns:
            # Build Tapback lookup
            tapback_rows = chat_df[
                (chat_df['associated_message_guid'].notna()) & 
                (chat_df['direction'] == 'out')
            ]
            for _, row in tapback_rows.iterrows():
                target_guid = row['associated_message_guid']
                if target_guid not in tapback_lookup:
                    tapback_lookup[target_guid] = []
                tapback_lookup[target_guid].append((
                    row['timestamp_local'],
                    row['message_id'],
                    row.get('tapback_type')
                ))
        
        if 'thread_originator_guid' in chat_df.columns or 'reply_to_guid' in chat_df.columns:
            # Build threaded reply lookup
            threaded_rows = chat_df[
                (
                    (chat_df.get('thread_originator_guid', pd.Series([None]*len(chat_df))).notna()) |
                    (chat_df.get('reply_to_guid', pd.Series([None]*len(chat_df))).notna())
                ) &
                (chat_df['direction'] == 'out') &
                (chat_df['associated_message_guid'].isna())
            ]
            for _, row in threaded_rows.iterrows():
                # Check both possible fields
                target_guids = []
                if pd.notna(row.get('thread_originator_guid')):
                    target_guids.append(row['thread_originator_guid'])
                if pd.notna(row.get('reply_to_guid')):
                    target_guids.append(row['reply_to_guid'])
                
                for target_guid in target_guids:
                    if target_guid not in threaded_lookup:
                        threaded_lookup[target_guid] = []
                    threaded_lookup[target_guid].append((
                        row['timestamp_local'],
                        row['message_id']
                    ))
        
        # OPTIMIZATION: Use numpy searchsorted for binary search instead of linear scan
        for pos in incoming_positions:
            current_timestamp = timestamps[pos]
            original_idx = original_indices[pos]
            current_guid = chat_df.iloc[pos].get('guid')
            
            potential_replies = []
            
            # Method 1: Check for Tapbacks (if enabled) - Now using pre-built lookup
            if count_tapbacks_as_replies and pd.notna(current_guid) and current_guid in tapback_lookup:
                for tapback_ts, msg_id, tb_type in tapback_lookup[current_guid]:
                    if tapback_ts > current_timestamp:
                        time_diff = (tapback_ts - current_timestamp).total_seconds() / 60.0
                        if 0 < time_diff <= max_gap_minutes:
                            potential_replies.append({
                                'message_id': msg_id,
                                'time_diff': time_diff,
                                'is_tapback': True,
                                'tapback_type': tb_type
                            })
                            got_tapbacks[original_idx] = True
            
            # Method 2: Check for threaded replies - Now using pre-built lookup
            if pd.notna(current_guid) and current_guid in threaded_lookup:
                for reply_ts, msg_id in threaded_lookup[current_guid]:
                    if reply_ts > current_timestamp:
                        time_diff = (reply_ts - current_timestamp).total_seconds() / 60.0
                        if 0 < time_diff <= max_gap_minutes:
                            potential_replies.append({
                                'message_id': msg_id,
                                'time_diff': time_diff,
                                'is_tapback': False,
                                'tapback_type': None
                            })
            
            # Method 3: SUPER OPTIMIZED chronological reply using binary search
            # Find index of first outgoing message after current timestamp
            insert_idx = np.searchsorted(outgoing_timestamps, current_timestamp, side='right')
            
            if insert_idx < len(outgoing_timestamps):
                next_out_timestamp = outgoing_timestamps[insert_idx]
                time_diff = (next_out_timestamp - current_timestamp) / np.timedelta64(1, 'm')  # Convert to minutes
                
                if 0 < time_diff <= max_gap_minutes:
                    # Get the actual message
                    next_out_pos = outgoing_positions[insert_idx]
                    next_reply = chat_df.iloc[next_out_pos]
                    is_tapback = pd.notna(next_reply.get('associated_message_guid'))
                    potential_replies.append({
                        'message_id': next_reply['message_id'],
                        'time_diff': time_diff,
                        'is_tapback': is_tapback,
                        'tapback_type': next_reply.get('tapback_type') if is_tapback else None
                    })
            
            # Select the earliest reply
            if potential_replies:
                earliest_reply = min(potential_replies, key=lambda x: x['time_diff'])
                response_times[original_idx] = earliest_reply['time_diff']
                got_replies[original_idx] = True
                reply_msg_ids[original_idx] = earliest_reply['message_id']
        
        processed_messages += len(chat_df)
        
        # Show progress every 10 chats or at the end
        if (chat_idx + 1) % 10 == 0 or (chat_idx + 1) == total_chats:
            elapsed = time.time() - start_time
            rate = processed_messages / elapsed if elapsed > 0 else 0
            percent = 100 * (chat_idx + 1) / total_chats
            print(f"    Progress: {chat_idx+1}/{total_chats} chats ({percent:.1f}%) | "
                  f"Processed {processed_messages:,} messages | Rate: {rate:.0f} msgs/sec")
    
    # OPTIMIZATION: Bulk update the dataframe once instead of per-message updates
    for idx, val in response_times.items():
        df.loc[idx, 'response_time_min'] = val
    for idx, val in got_replies.items():
        df.loc[idx, 'got_reply'] = val
    for idx, val in got_tapbacks.items():
        df.loc[idx, 'got_tapback'] = val
    for idx, val in reply_msg_ids.items():
        df.loc[idx, 'reply_message_id'] = val
    
    elapsed_total = time.time() - start_time
    print(f"  ✅ Response times computed for {processed_messages:,} messages in {elapsed_total:.1f}s")
    
    # Remove outliers if requested
    if remove_outliers and df['response_time_min'].notna().any():
        threshold = df['response_time_min'].quantile(outlier_percentile / 100.0)
        df.loc[df['response_time_min'] > threshold, 'response_time_min'] = np.nan
        df.loc[df['response_time_min'] > threshold, 'got_reply'] = False
    
    return df


def compute_reluctance_score(
    response_time_min: float,
    got_reply: bool,
    cap_minutes: float = 1440.0
) -> float:
    """
    Compute reluctance score for a message.
    
    Args:
        response_time_min: Response time in minutes (NaN if no reply)
        got_reply: Whether a reply was sent
        cap_minutes: Cap for response time scaling
        
    Returns:
        Reluctance score between 0 and 1 (higher = more reluctant)
    """
    if pd.isna(response_time_min) or not got_reply:
        # No reply - high reluctance
        return 1.0
    
    # Scale response time to [0, 1]
    score = min(response_time_min / cap_minutes, 1.0)
    return score


def get_response_statistics(df: pd.DataFrame) -> dict:
    """
    Get summary statistics about response times.
    
    Args:
        df: DataFrame with response_time_min and got_reply columns
        
    Returns:
        Dictionary with statistics
    """
    received = df[df['direction'] == 'in'].copy()
    
    stats = {
        'total_received': len(received),
        'total_replied': received['got_reply'].sum(),
        'reply_rate': received['got_reply'].mean() if len(received) > 0 else 0.0,
        'median_response_time': received['response_time_min'].median(),
        'mean_response_time': received['response_time_min'].mean(),
        'p25_response_time': received['response_time_min'].quantile(0.25),
        'p75_response_time': received['response_time_min'].quantile(0.75),
    }
    
    return stats


def compute_daily_response_times(df: pd.DataFrame, include_details: bool = False) -> pd.DataFrame:
    """
    Compute average response time per day for received messages that got replies.
    
    Args:
        df: DataFrame with response_time_min, got_reply, direction, and timestamp_local columns
        include_details: If True, include message details (text, participant) for hover tooltips
        
    Returns:
        DataFrame with columns: date, avg_response_time_min, count, and optionally message_details
    """
    received = df[df['direction'] == 'in'].copy()
    received_with_reply = received[received['got_reply'] & received['response_time_min'].notna()].copy()
    
    if len(received_with_reply) == 0 or 'timestamp_local' not in received_with_reply.columns:
        return pd.DataFrame(columns=['date', 'avg_response_time_min', 'count'])
    
    # Extract date from timestamp
    received_with_reply['date'] = pd.to_datetime(received_with_reply['timestamp_local']).dt.date
    
    if include_details:
        # Group by date and compute average, also collect message details
        def collect_details(group):
            # Limit to top 5 messages by response time for each day to avoid huge tooltips
            top_messages = group.nlargest(5, 'response_time_min')
            details = []
            for _, row in top_messages.iterrows():
                text = str(row.get('text', ''))[:100]  # Limit text length
                # Prefer chat_name (actual contact/group name) over participant (de-identified ID)
                contact_name = str(row.get('chat_name', ''))
                if not contact_name or contact_name == 'Unknown Chat':
                    # Fall back to participant if chat_name is not available
                    contact_name = str(row.get('participant', 'Unknown'))
                rt = row['response_time_min']
                details.append({
                    'text': text,
                    'participant': contact_name,  # Use contact_name instead of de-identified participant
                    'response_time': rt
                })
            return details
        
        # Group by date and compute average
        daily_stats = received_with_reply.groupby('date').agg({
            'response_time_min': ['mean', 'count']
        }).reset_index()
        daily_stats.columns = ['date', 'avg_response_time_min', 'count']
        
        # Add message details - use merge to ensure proper alignment
        message_details_df = received_with_reply.groupby('date').apply(collect_details).reset_index()
        message_details_df.columns = ['date', 'message_details']
        daily_stats = daily_stats.merge(message_details_df, on='date', how='left')
    else:
        # Group by date and compute average
        daily_stats = received_with_reply.groupby('date').agg({
            'response_time_min': ['mean', 'count']
        }).reset_index()
        daily_stats.columns = ['date', 'avg_response_time_min', 'count']
    
    daily_stats = daily_stats.sort_values('date')
    
    return daily_stats


def compute_daily_text_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average text length in words per day.
    
    Args:
        df: DataFrame with text and timestamp_local columns
        
    Returns:
        DataFrame with columns: date, avg_word_count, count
    """
    if len(df) == 0 or 'timestamp_local' not in df.columns:
        return pd.DataFrame(columns=['date', 'avg_word_count', 'count'])
    
    df_copy = df.copy()
    
    # Extract date from timestamp
    df_copy['date'] = pd.to_datetime(df_copy['timestamp_local']).dt.date
    
    # Compute word count for each message
    def count_words(text):
        if pd.isna(text) or not isinstance(text, str):
            return 0
        return len(text.split())
    
    df_copy['word_count'] = df_copy['text'].apply(count_words)
    
    # Group by date and compute average
    daily_stats = df_copy.groupby('date').agg({
        'word_count': ['mean', 'count']
    }).reset_index()
    
    daily_stats.columns = ['date', 'avg_word_count', 'count']
    daily_stats = daily_stats.sort_values('date')
    
    return daily_stats


def compute_daily_sent_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute number of texts sent out per day.
    
    Args:
        df: DataFrame with direction and timestamp_local columns
        
    Returns:
        DataFrame with columns: date, sent_count
    """
    sent = df[df['direction'] == 'out'].copy()
    
    if len(sent) == 0 or 'timestamp_local' not in sent.columns:
        return pd.DataFrame(columns=['date', 'sent_count'])
    
    # Extract date from timestamp
    sent['date'] = pd.to_datetime(sent['timestamp_local']).dt.date
    
    # Count messages per day
    daily_counts = sent.groupby('date').size().reset_index(name='sent_count')
    daily_counts = daily_counts.sort_values('date')
    
    return daily_counts

