"""
Test script to inspect raw and preprocessed iMessage data.
Usage: python testrun.py /path/to/chat.db
"""

import sys
import os
import pandas as pd
import sqlite3
from data_loader import load_messages
from preprocess import preprocess_text, preprocess_documents, METADATA_STOPLIST
from responses import compute_response_times

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)


def inspect_raw_database(db_path: str):
    """Inspect the raw database structure and content."""
    print("\n" + "="*80)
    print("RAW DATABASE INSPECTION")
    print("="*80)
    
    con = sqlite3.connect(db_path)
    
    # List all tables
    print("\n📋 Tables in database:")
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", 
        con
    )
    for table in tables['name']:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  - {table}: {count:,} rows")
    
    # Check message table columns
    print("\n📊 Columns in 'message' table:")
    cursor = con.execute("PRAGMA table_info(message)")
    for row in cursor.fetchall():
        col_id, col_name, col_type, not_null, default_val, pk = row
        print(f"  - {col_name} ({col_type})")
    
    # Sample raw messages
    print("\n💬 Sample raw messages (first 10):")
    query = """
    SELECT 
        ROWID as message_id,
        text,
        is_from_me,
        date,
        handle_id,
        cache_has_attachments,
        associated_message_type
    FROM message
    ORDER BY date DESC
    LIMIT 10
    """
    try:
        raw_sample = pd.read_sql_query(query, con)
        print(raw_sample.to_string(index=False))
    except Exception as e:
        print(f"  Error: {e}")
    
    # Check for reactions
    print("\n🎭 Message type distribution:")
    try:
        type_dist = pd.read_sql_query(
            """
            SELECT 
                associated_message_type,
                COUNT(*) as count
            FROM message
            WHERE associated_message_type IS NOT NULL
            GROUP BY associated_message_type
            ORDER BY count DESC
            LIMIT 10
            """,
            con
        )
        if len(type_dist) > 0:
            print(type_dist.to_string(index=False))
            print("\n  Note: associated_message_type values:")
            print("    0 = Regular message")
            print("    2000-3000 = Reactions (likes, laughs, etc.)")
        else:
            print("  No associated_message_type data found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Export raw database to CSV
    print("\n📤 Exporting raw database to CSV...")
    try:
        output_dir = "test_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        raw_db_csv = os.path.join(output_dir, "0_raw_database.csv")
        query = """
        SELECT 
            m.ROWID as message_id,
            m.guid,
            m.text,
            m.is_from_me,
            m.date,
            m.handle_id,
            m.associated_message_type,
            m.associated_message_guid,
            h.id as handle_identifier,
            c.display_name as chat_name
        FROM message m
        LEFT JOIN handle h ON h.ROWID = m.handle_id
        LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        LEFT JOIN chat c ON c.ROWID = cmj.chat_id
        ORDER BY m.date DESC
        LIMIT 100000
        """
        raw_db_df = pd.read_sql_query(query, con)
        raw_db_df.to_csv(raw_db_csv, index=False)
        print(f"  ✅ Saved to: {raw_db_csv}")
        print(f"     ({len(raw_db_df):,} rows)")
    except Exception as e:
        print(f"  ⚠️  Error exporting raw database: {e}")
    
    con.close()


def inspect_loaded_data(df: pd.DataFrame):
    """Inspect data after loading but before preprocessing."""
    print("\n" + "="*80)
    print("LOADED DATA (After data_loader, before preprocessing)")
    print("="*80)
    
    print(f"\n📊 Total messages loaded: {len(df):,}")
    
    print("\n📋 Columns:")
    for col in df.columns:
        print(f"  - {col}")
    
    print("\n📈 Data summary:")
    print(f"  - Date range: {df['timestamp_local'].min()} to {df['timestamp_local'].max()}")
    if 'direction' in df.columns:
        print(f"  - Inbound messages: {len(df[df['direction'] == 'in']):,}")
        print(f"  - Outbound messages: {len(df[df['direction'] == 'out']):,}")
    if 'group_size' in df.columns:
        print(f"  - One-to-one chats (size=2): {len(df[df['group_size'] == 2]):,}")
        print(f"  - Group chats (size>2): {len(df[df['group_size'] > 2]):,}")
    
    # Text length statistics
    df['text_length'] = df['text'].str.len()
    print(f"\n📝 Text length statistics:")
    print(f"  - Mean: {df['text_length'].mean():.1f} characters")
    print(f"  - Median: {df['text_length'].median():.1f} characters")
    print(f"  - Max: {df['text_length'].max():.0f} characters")
    print(f"  - Empty texts: {len(df[df['text_length'] == 0]):,}")
    
    # Sample messages
    print("\n💬 Sample messages (first 10):")
    sample_cols = ['message_id', 'chat_name', 'participant', 'direction', 'text', 'timestamp_local']
    available_cols = [c for c in sample_cols if c in df.columns]
    print(df[available_cols].head(10).to_string(index=False))


def inspect_preprocessed_data(df: pd.DataFrame, processed_docs: list):
    """Inspect data after preprocessing."""
    print("\n" + "="*80)
    print("PREPROCESSED DATA (After text cleaning and tokenization)")
    print("="*80)
    
    # Count tokens
    total_tokens = sum(len(doc) for doc in processed_docs)
    non_empty_docs = [doc for doc in processed_docs if len(doc) > 0]
    
    print(f"\n📊 Preprocessing statistics:")
    print(f"  - Total documents: {len(processed_docs):,}")
    print(f"  - Non-empty documents: {len(non_empty_docs):,}")
    print(f"  - Empty documents: {len(processed_docs) - len(non_empty_docs):,}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Average tokens per document: {total_tokens / max(len(processed_docs), 1):.1f}")
    print(f"  - Average tokens per non-empty doc: {total_tokens / max(len(non_empty_docs), 1):.1f}")
    
    # Token frequency
    from collections import Counter
    all_tokens = [token for doc in processed_docs for token in doc]
    token_freq = Counter(all_tokens)
    
    print(f"\n🔤 Most common tokens (top 20):")
    for token, count in token_freq.most_common(20):
        print(f"  - '{token}': {count:,}")
    
    # Vocabulary size
    print(f"\n📚 Vocabulary statistics:")
    print(f"  - Unique tokens: {len(token_freq):,}")
    print(f"  - Tokens appearing once: {sum(1 for c in token_freq.values() if c == 1):,}")
    print(f"  - Tokens appearing 10+ times: {sum(1 for c in token_freq.values() if c >= 10):,}")
    
    # Sample preprocessed documents
    print("\n📝 Sample preprocessed documents (first 10 non-empty):")
    count = 0
    for i, (doc, row) in enumerate(zip(processed_docs, df.itertuples())):
        if len(doc) > 0 and count < 10:
            original_text = row.text[:80] if hasattr(row, 'text') else "N/A"
            print(f"\n  [{count+1}] Original: {original_text}")
            print(f"      Tokens ({len(doc)}): {' '.join(doc[:20])}")
            if len(doc) > 20:
                print(f"      ... ({len(doc) - 20} more tokens)")
            count += 1
        if count >= 10:
            break
    
    # Check for metadata tokens that might have leaked
    print("\n🔍 Checking for metadata leakage:")
    metadata_found = {}
    for token in all_tokens:
        if token.lower() in METADATA_STOPLIST:
            metadata_found[token] = metadata_found.get(token, 0) + 1
    
    if metadata_found:
        print("  ⚠️  WARNING: Found metadata tokens in preprocessed data:")
        for token, count in sorted(metadata_found.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    - '{token}': {count:,} occurrences")
    else:
        print("  ✅ No metadata tokens found in preprocessed data")


def inspect_with_response_times(df: pd.DataFrame):
    """Inspect data with response times calculated."""
    print("\n" + "="*80)
    print("RESPONSE TIME ANALYSIS")
    print("="*80)
    
    df_with_rt = compute_response_times(
        df,
        max_gap_minutes=1440.0,
        remove_outliers=True,
        outlier_percentile=99.0
    )
    
    received = df_with_rt[df_with_rt['direction'] == 'in'].copy()
    
    print(f"\n📊 Response statistics:")
    print(f"  - Total received messages: {len(received):,}")
    print(f"  - Messages with replies: {received['got_reply'].sum():,}")
    print(f"  - Reply rate: {received['got_reply'].mean():.2%}")
    print(f"  - Mean response time: {received['response_time_min'].mean():.1f} minutes")
    print(f"  - Median response time: {received['response_time_min'].median():.1f} minutes")
    
    # Sample messages with response times
    print("\n💬 Sample received messages with response times (first 10):")
    sample_cols = ['chat_name', 'participant', 'text', 'got_reply', 'response_time_min']
    available_cols = [c for c in sample_cols if c in received.columns]
    print(received[available_cols].head(10).to_string(index=False))


def main():
    if len(sys.argv) < 2:
        print("Usage: python testrun.py /path/to/chat.db")
        print("\nExample:")
        print("  python testrun.py ~/Library/Messages/chat.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    if not os.path.exists(db_path):
        print(f"❌ Error: File not found: {db_path}")
        sys.exit(1)
    
    print(f"🔍 Inspecting iMessage database: {db_path}")
    
    # Step 1: Inspect raw database
    inspect_raw_database(db_path)
    
    # Step 2: Load data using data_loader
    print("\n⏳ Loading data using data_loader...")
    
    # Create a temporary file object to mimic uploaded file
    import io
    with open(db_path, 'rb') as f:
        file_content = f.read()
    
    class MockUploadedFile:
        def __init__(self, content, name):
            self.content = content
            self.name = name
            self._buffer = io.BytesIO(content)
        
        def read(self, size=-1):
            return self._buffer.read(size)
        
        def seek(self, pos):
            return self._buffer.seek(pos)
        
        def tell(self):
            return self._buffer.tell()
        
        def getbuffer(self):
            return self._buffer.getbuffer()
    
    mock_file = MockUploadedFile(file_content, "chat.db")
    
    try:
        df = load_messages([mock_file], deidentify=False)
        inspect_loaded_data(df)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Preprocess data
    print("\n⏳ Preprocessing text (MINIMAL FILTERING - keeping raw information)...")
    
    # Use received messages only (like RQ1)
    received = df[df['direction'] == 'in'].copy()
    
    processed_docs = preprocess_documents(
        received['text'],
        lowercase=True,
        remove_stopwords=False,  # DISABLED
        remove_emojis_flag=True,  # Keep emoji removal
        stem=False,
        lemmatize=False,
        min_length=1,  # Changed from 2 to 1 (keep everything except empty)
        custom_stopwords=set(),
        disabled_metadata_stopwords=set(),
        remove_metadata_tokens=False  # DISABLED - keep all tokens, no metadata filtering
    )
    
    print("  ⚠️  NOTE: Stopword removal, minimum length filtering, and metadata filtering are ALL DISABLED")
    print("  This shows ALL tokens after basic tokenization (keeping maximum raw information)")
    
    inspect_preprocessed_data(received, processed_docs)
    
    # Step 4: Response time analysis
    print("\n⏳ Computing response times...")
    inspect_with_response_times(df)
    
    # Step 5: Save full datasets to CSV for inspection
    print("\n" + "="*80)
    print("EXPORTING FULL DATASETS")
    print("="*80)
    
    output_dir = "test_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Export raw loaded data
    raw_csv = os.path.join(output_dir, "1_raw_loaded_data.csv")
    df.to_csv(raw_csv, index=False)
    print(f"\n✅ Exported raw loaded data: {raw_csv}")
    print(f"   ({len(df):,} rows × {len(df.columns)} columns)")
    
    # Export received messages only
    received_csv = os.path.join(output_dir, "2_received_messages_only.csv")
    received.to_csv(received_csv, index=False)
    print(f"\n✅ Exported received messages: {received_csv}")
    print(f"   ({len(received):,} rows × {len(received.columns)} columns)")
    
    # Export preprocessed data with tokens
    preprocessed_df = received.copy()
    preprocessed_df['preprocessed_tokens'] = processed_docs
    preprocessed_df['token_count'] = preprocessed_df['preprocessed_tokens'].apply(len)
    preprocessed_df['tokens_as_text'] = preprocessed_df['preprocessed_tokens'].apply(lambda x: ' '.join(x))
    
    preprocessed_csv = os.path.join(output_dir, "3_preprocessed_with_tokens.csv")
    preprocessed_df[['message_id', 'chat_name', 'participant', 'text', 'tokens_as_text', 'token_count', 'timestamp_local']].to_csv(
        preprocessed_csv, index=False
    )
    print(f"\n✅ Exported preprocessed data: {preprocessed_csv}")
    print(f"   ({len(preprocessed_df):,} rows × 7 columns)")
    
    # Export with response times
    df_with_rt_csv = os.path.join(output_dir, "4_with_response_times.csv")
    df_with_rt = compute_response_times(df, max_gap_minutes=1440.0, remove_outliers=True, outlier_percentile=99.0)
    df_with_rt.to_csv(df_with_rt_csv, index=False)
    print(f"\n✅ Exported data with response times: {df_with_rt_csv}")
    print(f"   ({len(df_with_rt):,} rows × {len(df_with_rt.columns)} columns)")
    
    # Export empty documents for inspection
    empty_docs_df = preprocessed_df[preprocessed_df['token_count'] == 0].copy()
    if len(empty_docs_df) > 0:
        empty_csv = os.path.join(output_dir, "5_empty_documents.csv")
        empty_docs_df[['message_id', 'chat_name', 'participant', 'text', 'timestamp_local']].to_csv(
            empty_csv, index=False
        )
        print(f"\n✅ Exported empty documents: {empty_csv}")
        print(f"   ({len(empty_docs_df):,} rows × 5 columns)")
    
    print("\n" + "="*80)
    print("✅ INSPECTION COMPLETE")
    print("="*80)
    print(f"\n📁 All files saved to: {os.path.abspath(output_dir)}/")
    print("\nYou can now open these CSV files to inspect:")
    print("  0. 0_raw_database.csv - Direct export from SQLite database (before data_loader)")
    print("  1. 1_raw_loaded_data.csv - All messages after data_loader")
    print("  2. 2_received_messages_only.csv - Only inbound messages")
    print("  3. 3_preprocessed_with_tokens.csv - Preprocessed tokens alongside original text")
    print("  4. 4_with_response_times.csv - All messages with response time calculations")
    print("  5. 5_empty_documents.csv - Messages that became empty after preprocessing")


if __name__ == "__main__":
    main()

