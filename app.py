"""Streamlit app for iMessage Analyzer."""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime
import base64
from io import BytesIO

# Import our modules
from data_loader import load_messages
from preprocess import preprocess_documents
from topics import train_lda, get_topic_words, compute_topic_coherence
from responses import (
    compute_response_times, get_response_statistics,
    compute_daily_response_times, compute_daily_text_length, compute_daily_sent_count
)
from analytics import (
    rank_reluctant_topics, topic_over_time, get_per_contact_topic_mix, group_vs_dm_stats,
    get_reluctant_topic_examples, get_contact_reluctant_topic_stats, get_topic_message_counts,
    get_contact_reluctance_proportions, analyze_conversation_starters, analyze_conversation_enders,
    analyze_topics_by_closeness, analyze_topics_by_time_of_day, analyze_high_response_topics,
    rank_reluctant_topics_embedding
)
from viz import (
    plot_reluctant_topics, plot_response_times_by_topic, plot_topic_prevalence_over_time,
    plot_per_contact_topic_mix, plot_group_vs_dm_response, plot_response_time_boxplot,
    plot_daily_response_times, plot_daily_text_length, plot_daily_sent_count,
    plot_contact_reluctant_topic_count, plot_contact_avg_reluctance, plot_contact_reluctance_proportions,
    plot_conversation_starters, plot_conversation_enders, plot_topics_by_closeness,
    plot_topics_by_time_heatmap, plot_high_response_topics, plot_reluctant_embedding_topics
)
from utils import get_mallet_home, ensure_exports_dir, format_duration
from chatbot import build_behavior_summary, run_rq9_chatbot, get_example_questions

# Page config
st.set_page_config(
    page_title="iMessage Analyzer",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if 'messages_df' not in st.session_state:
    st.session_state.messages_df = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None


def export_html_report(rq1_df, rq2_contact_df, rq2_time_df, rq3_stats, plots_dict):
    """Export a self-contained HTML report."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>iMessage Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>iMessage Analysis Report</h1>
        <p>Generated: {timestamp}</p>
        
        <h2>Research Question 1: Topics You Tend Not to Engage With</h2>
        {rq1_table}
        
        <h2>Research Question 2: Most Commonly Discussed Topics</h2>
        <h3>Topic Distribution by Contact</h3>
        {rq2_contact_table}
        <h3>Topic Prevalence Over Time</h3>
        {rq2_time_table}
        
        <h2>Research Question 3: Group vs One-to-One Responsiveness</h2>
        {rq3_table}
        
        <h2>Plots</h2>
        {plots_section}
    </body>
    </html>
    """
    
    # Convert DataFrames to HTML
    rq1_table_html = rq1_df.to_html(classes='table', escape=False) if rq1_df is not None else "<p>No data</p>"
    rq2_contact_table_html = rq2_contact_df.to_html(classes='table', escape=False) if rq2_contact_df is not None else "<p>No data</p>"
    rq2_time_table_html = rq2_time_df.to_html(classes='table', escape=False) if rq2_time_df is not None else "<p>No data</p>"
    
    rq3_table_html = ""
    if rq3_stats and 'by_category' in rq3_stats:
        rq3_table_html = rq3_stats['by_category'].to_html(classes='table', escape=False)
    
    # Embed plots as base64 images
    plots_html = ""
    for plot_name, fig in plots_dict.items():
        if fig is not None:
            try:
                # Save plot to bytes
                img_bytes = BytesIO()
                if hasattr(fig, 'write_image'):
                    # Plotly figure
                    fig.write_image(img_bytes, format='png')
                else:
                    # Matplotlib figure
                    fig.savefig(img_bytes, format='png', bbox_inches='tight')
                img_bytes.seek(0)
                img_base64 = base64.b64encode(img_bytes.read()).decode()
                plots_html += f'<h3>{plot_name}</h3><img src="data:image/png;base64,{img_base64}" />'
            except:
                plots_html += f'<p>Could not render {plot_name}</p>'
    
    html_content = html_content.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        rq1_table=rq1_table_html,
        rq2_contact_table=rq2_contact_table_html,
        rq2_time_table=rq2_time_table_html,
        rq3_table=rq3_table_html,
        plots_section=plots_html
    )
    
    return html_content


def main():
    st.title("💬 iMessage Analyzer")
    st.markdown("**Privacy-first local analysis of your iMessage conversations**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Text Cleaning")
        # Basic cleaning for LDA
        lowercase = True  # Always lowercase
        remove_stopwords = False  # DISABLED: Only use custom stopwords from UI, not generic stopwords
        remove_emojis_flag = True  # Always remove emojis
        
        st.info("ℹ️ Basic tokenization enabled: lowercases text, removes emojis. Generic stopwords are DISABLED - only custom stopwords from below will be filtered.")
        
        # Custom stopwords management
        st.subheader("Custom Stopwords (Junk Token Filtering)")
        st.markdown("**These are the ONLY stopwords that will be filtered** (generic NLTK stopwords are disabled). Add words you want to remove from your corpus:")
        
        # Get current NLTK stopwords
        try:
            from nltk.corpus import stopwords
            nltk_stopwords = set(stopwords.words('english'))
        except:
            nltk_stopwords = set()
        
        # Initialize custom stopwords in session state
        if 'custom_stopwords' not in st.session_state:
            st.session_state.custom_stopwords = set()
        
        # Display current custom stopwords
        if st.session_state.custom_stopwords:
            st.write("**Current custom stopwords:**")
            custom_list = sorted(list(st.session_state.custom_stopwords))
            
            # Use a more compact display with selectbox for removal
            if len(custom_list) <= 20:
                # Show as buttons for small lists
                cols = st.columns(min(5, len(custom_list)))
                for idx, word in enumerate(custom_list):
                    col_idx = idx % 5
                    with cols[col_idx]:
                        if st.button(f"❌ {word}", key=f"remove_{word}"):
                            st.session_state.custom_stopwords.remove(word)
                            st.rerun()
            else:
                # For larger lists, show in a selectbox
                selected_to_remove = st.selectbox(
                    "Select a word to remove:",
                    [""] + custom_list,
                    key="select_remove_stopword"
                )
                if selected_to_remove:
                    if st.button("Remove Selected", key="remove_selected"):
                        st.session_state.custom_stopwords.remove(selected_to_remove)
                        st.rerun()
        
        # Add new stopwords
        new_stopwords_input = st.text_input(
            "Add stopwords (comma-separated or one per line)",
            placeholder="e.g., word1, word2, word3",
            key="new_stopwords_input"
        )
        if st.button("➕ Add Stopwords", key="add_stopwords"):
            if new_stopwords_input:
                # Parse input - handle both comma-separated and newline-separated
                words = []
                for line in new_stopwords_input.split('\n'):
                    words.extend([w.strip().lower() for w in line.split(',') if w.strip()])
                
                # Add to custom stopwords
                added_count = 0
                for word in words:
                    if word:  # Only add non-empty words
                        st.session_state.custom_stopwords.add(word.lower())
                        added_count += 1
                
                if added_count > 0:
                    st.success(f"Added {added_count} stopword(s)")
                    st.rerun()
        
        # Show total stopwords count (only custom since NLTK is disabled)
        st.caption(f"Custom stopwords active: {len(st.session_state.custom_stopwords)} | NLTK stopwords: DISABLED for LDA")
        
        # Hardcoded metadata stopwords management
        st.subheader("Hardcoded Metadata Stopwords")
        st.markdown("These are hardcoded junk tokens from attributedBody decoding that are always filtered. You can toggle them on/off individually.")
        
        # Import METADATA_STOPLIST from preprocess
        from preprocess import METADATA_STOPLIST
        
        # Initialize disabled metadata stopwords in session state
        if 'disabled_metadata_stopwords' not in st.session_state:
            st.session_state.disabled_metadata_stopwords = set()
        
        # Group metadata stopwords by category for better organization
        metadata_categories = {
            'Binary/Plist Artifacts': ['bplist', 'bplist00', 'bplist01', 'nskeyedarchiver', 'xversionyarchiver', 'nsdictionary'],
            'String Classes': ['nsattributedstring', 'nsmutableattributedstring', 'nsstring', 'nsmutablestring'],
            'Decode Junk': ['rmsv', 'topx', 'emqrstux', 'tdate', 'tu', 'ydayofweek', 'ydaynumber', 'ijk', 'qv'],
            'RTF/Plist Artifacts': ['uhours', 'uvalue', 'utime', 'ttime', 'abc', 'de', 'ij', 'uv', 'ghi', 'adhimu', 'adhlu'],
            'Malformed Decoding': ['pthat', 'amttime', 'pmttime', 'ajust', 'rjust', 'luhours'],
            'Other Artifacts': ['ddscannerresult', 'wversionyddresult', 'classrarqtqprsrrvn', 'classnamex', 'znsobjects',
                               'kimfiletransferguidattributename', 'kimmessagepartattributename', 'kimbasewritingdirectionattributename',
                               'nsnumber', 'nsvalue', 'nsrange', 'nslocation', 'nslength', 'nsdata',
                               'nsarray', 'nsmutablearray', 'nsset', 'nsmutableset',
                               'nsfont', 'nsparagraphstyle', 'nscolor', 'nsunderline', 'nsstrikethrough',
                               'nssuperscript', 'nslink', 'nsattachment', '$objects', '$archiver', '$version', '$top', '$class',
                               'rtf1', 'cocoasubrtf', 'null', 'objects', 'ef', 'ar', 'qt', 'pr', 'rr', 'vn', 'dd']
        }
        
        # Display metadata stopwords by category with toggles
        for category, words in metadata_categories.items():
            with st.expander(f"{category} ({len(words)} words)", expanded=False):
                # Create columns for checkboxes (3 columns)
                words_list = sorted([w for w in words if w in METADATA_STOPLIST])
                if words_list:
                    num_cols = 3
                    cols = st.columns(num_cols)
                    for idx, word in enumerate(words_list):
                        col_idx = idx % num_cols
                        with cols[col_idx]:
                            # Check if word is currently disabled
                            is_disabled = word in st.session_state.disabled_metadata_stopwords
                            # Checkbox: checked = enabled (not disabled), unchecked = disabled
                            checkbox_value = st.checkbox(word, value=not is_disabled, key=f"metadata_{word}")
                            
                            # Update session state based on checkbox state
                            if checkbox_value and is_disabled:
                                # Was disabled, now enabled - remove from disabled set
                                st.session_state.disabled_metadata_stopwords.discard(word)
                            elif not checkbox_value and not is_disabled:
                                # Was enabled, now disabled - add to disabled set
                                st.session_state.disabled_metadata_stopwords.add(word)
        
        # Show summary
        enabled_count = len(METADATA_STOPLIST) - len(st.session_state.disabled_metadata_stopwords)
        st.caption(f"Metadata stopwords: {enabled_count} enabled, {len(st.session_state.disabled_metadata_stopwords)} disabled (out of {len(METADATA_STOPLIST)} total)")
        
        stem = st.checkbox("Stemming", value=False)
        lemmatize = st.checkbox("Lemmatization", value=False)
        min_length = st.slider("Min Token Length", 1, 5, 2)
        
        st.subheader("Topic Modeling")
        num_topics = st.slider("Number of Topics (K)", 5, 30, 10,
                              help="How many topics to discover. More topics = more specific, fewer = more general.")
        passes = st.slider("Passes", 1, 50, 10,
                          help="Number of times to iterate through the data. More passes = better quality but slower.")
        # Use Gensim defaults: alpha='auto', beta=0.01
        alpha = 'auto'
        beta = 0.01
        
        mallet_home = get_mallet_home()
        use_mallet = False
        if mallet_home:
            use_mallet = st.checkbox("Use MALLET (optional)", value=False, 
                                    help=f"MALLET found at {mallet_home}. MALLET can provide slightly better topic modeling results, but Gensim LDA works great too.")
        else:
            # Only show a subtle note - MALLET is optional
            with st.expander("ℹ️ About MALLET (optional)", expanded=False):
                st.caption("MALLET is an optional tool for topic modeling. The app works perfectly fine with Gensim LDA (default). To use MALLET, download it from http://mallet.cs.umass.edu/ and set the MALLET_HOME environment variable.")
        
        st.subheader("Response Time")
        remove_outliers = st.checkbox("Remove Outliers", value=True,
                                      help="Remove extreme response times that might skew the analysis (e.g., very long delays).")
        
        count_tapbacks_as_replies = st.checkbox("Count Tapbacks (❤️, 👍, 😂) as Replies", value=False,
                                                help="If enabled, iMessage reactions (Loved, Liked, Laughed, etc.) count as valid replies. If disabled, only actual text messages count as replies.")
        
        # Only show time gap and percentile options when remove_outliers is enabled
        if remove_outliers:
            max_gap_minutes = st.slider("Max Gap (minutes)", 60, 2880, 1440, 60,
                                       help="Maximum time window to count a message as a 'response'. Messages replied to after this window are marked as 'no reply'. Default 1440 = 24 hours.")
            outlier_percentile = st.slider("Outlier Percentile", 90.0, 99.9, 99.0, 0.1,
                                           help="Remove response times above this percentile. 99.0 = remove top 1% longest responses. Higher = keep more data.")
        else:
            # Use default values when not shown
            max_gap_minutes = 1440.0
            outlier_percentile = 99.0
        
        st.subheader("Privacy")
        deidentify = st.checkbox("De-identify Participants", value=True)
    
    # Consent and file upload section
    st.header("📁 Upload Database Files")
    
    # Privacy notice and consent
    st.markdown("**Privacy-first local analysis of your iMessage conversations**")
    
    consent = st.checkbox(
        "✅ I am uploading my text data at my own risk and acknowledge to use this tool at my own risk",
        value=False,
        key="user_consent"
    )
    
    if not consent:
        st.warning("⚠️ Please review and accept the consent above to proceed with file upload.")
        st.info("""
        **What you're agreeing to:**
        - You understand this tool processes your personal message data
        - Analysis happens locally on your computer (no data uploaded to external servers by default)
        - You use this tool at your own discretion
        - Optional RQ9 chatbot feature sends only aggregated statistics to OpenAI (if you choose to use it)
        """)
        uploaded_files = None
    else:
        # Show instructions for finding chat.db
        with st.expander("📖 How to find your chat.db file"):
            st.markdown("""
            **Location:** Your iMessage database is typically located at:
            
            `~/Library/Messages/chat.db`
            
            **To locate it:**
            1. Open **Finder**
            2. Press **Cmd+Shift+G** (Go to Folder)
            3. Type: `~/Library/Messages`
            4. Press Enter
            5. Look for `chat.db` (and optionally `chat.db-wal` and `chat.db-shm`)
            
            **Note:** You may need to copy these files to your Desktop first, as the Library folder is protected.
            """)
        
        uploaded_files = st.file_uploader(
            "Select your chat.db file (and optionally chat.db-wal, chat.db-shm):",
            type=['db', 'db-wal', 'db-shm'],
            accept_multiple_files=True,
            help="Located at ~/Library/Messages/chat.db - Press Cmd+Shift+G in Finder to navigate there"
        )
    
    if uploaded_files:
        # Detect file types
        db_file = None
        wal_file = None
        shm_file = None
        
        for f in uploaded_files:
            if f.name == "chat.db":
                db_file = f
            elif f.name == "chat.db-wal":
                wal_file = f
            elif f.name == "chat.db-shm":
                shm_file = f
        
        if db_file is None:
            st.error("❌ Please upload chat.db file")
        else:
            if st.button("Load Messages", type="primary"):
                with st.spinner("Loading messages from database..."):
                    try:
                        df = load_messages(uploaded_files, deidentify=deidentify)
                        st.session_state.messages_df = df
                        if 'temp_dir' in df.attrs:
                            st.session_state.temp_dir = df.attrs.get('temp_dir')
                        
                        # Clear response time cache when new data is loaded
                        if 'df_with_rt_cache_key' in st.session_state:
                            del st.session_state.df_with_rt_cache_key
                        if 'df_with_rt' in st.session_state:
                            del st.session_state.df_with_rt
                        if 'received_with_rt' in st.session_state:
                            del st.session_state.received_with_rt
                        
                        st.success(f"✅ Loaded {len(df):,} messages")
                    except Exception as e:
                        st.error(f"❌ Error loading messages: {e}")
                        st.exception(e)
    
    # Data snapshot
    if st.session_state.messages_df is not None:
        df = st.session_state.messages_df
        
        st.header("📊 Data Snapshot")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Messages", f"{len(df):,}")
        with col2:
            inbound_count = len(df[df['direction'] == 'in']) if 'direction' in df.columns else 0
            st.metric("Messages Received", f"{inbound_count:,}")
        with col3:
            outbound_count = len(df[df['direction'] == 'out']) if 'direction' in df.columns else 0
            st.metric("Messages Sent", f"{outbound_count:,}")
        with col4:
            unique_chats = df['chat_id'].nunique()
            st.metric("Unique Chats", f"{unique_chats:,}")
        with col5:
            unique_participants = df['participant'].nunique()
            st.metric("Unique Participants", f"{unique_participants:,}")
        with col6:
            group_chats = df[df['group_size'] > 2]['chat_id'].nunique() if 'chat_id' in df.columns else 0
            st.metric("Group Chats", f"{group_chats:,}")
        
        # Date range on a separate row since it's wider
        col_date = st.columns(1)[0]
        with col_date:
            if df['timestamp_local'].notna().any():
                date_range = f"{df['timestamp_local'].min().date()} to {df['timestamp_local'].max().date()}"
                st.metric("Date Range", date_range)
            else:
                st.metric("Date Range", "N/A")
        
        # Analysis sections
        st.header("🔬 Analysis")
        
        # Compute response times first (cached in session to avoid recomputation)
        # Create a cache key based on settings that affect response time calculation
        rt_cache_key = f"rt_{max_gap_minutes}_{remove_outliers}_{outlier_percentile}_{count_tapbacks_as_replies}"
        
        if 'df_with_rt_cache_key' not in st.session_state or st.session_state.df_with_rt_cache_key != rt_cache_key:
            # Need to compute or recompute
            with st.spinner("Computing response times..."):
                df_with_rt = compute_response_times(
                    df,
                    max_gap_minutes=max_gap_minutes,
                    remove_outliers=remove_outliers,
                    outlier_percentile=outlier_percentile,
                    count_tapbacks_as_replies=count_tapbacks_as_replies
                )
                received_with_rt = df_with_rt[df_with_rt['direction'] == 'in'].copy()
                received_with_rt = received_with_rt.reset_index(drop=True)  # Ensure sequential index
                
                # Cache the results
                st.session_state.df_with_rt = df_with_rt
                st.session_state.received_with_rt = received_with_rt
                st.session_state.df_with_rt_cache_key = rt_cache_key
        else:
            # Use cached results
            df_with_rt = st.session_state.df_with_rt
            received_with_rt = st.session_state.received_with_rt
        
        # Daily metrics visualizations
        st.header("📈 Daily Metrics Over Time")
        st.markdown("""
        Explore your messaging patterns over time with these daily metrics.
        """)
        
        if len(df_with_rt) > 0 and df_with_rt['timestamp_local'].notna().any():
            # Compute daily metrics
            daily_response_times = compute_daily_response_times(df_with_rt, include_details=True)
            daily_text_length = compute_daily_text_length(df_with_rt)
            daily_sent_count = compute_daily_sent_count(df_with_rt)
            
            # Plot 1: Average response time per day
            if len(daily_response_times) > 0:
                st.subheader("Average Response Time Per Day")
                fig1 = plot_daily_response_times(daily_response_times, use_plotly=True)
                if fig1:
                    try:
                        st.plotly_chart(fig1, use_container_width=True)
                    except:
                        st.pyplot(fig1)
            
            # Plot 2: Average text length per day
            if len(daily_text_length) > 0:
                st.subheader("Average Text Length (Words) Per Day")
                fig2 = plot_daily_text_length(daily_text_length, use_plotly=True)
                if fig2:
                    try:
                        st.plotly_chart(fig2, use_container_width=True)
                    except:
                        st.pyplot(fig2)
            
            # Plot 3: Number of texts sent per day
            if len(daily_sent_count) > 0:
                st.subheader("Number of Texts Sent Per Day")
                fig3 = plot_daily_sent_count(daily_sent_count, use_plotly=True)
                if fig3:
                    try:
                        st.plotly_chart(fig3, use_container_width=True)
                    except:
                        st.pyplot(fig3)
        
        st.markdown("---")  # Separator before research questions
        
        # Preprocessing - must match order of received_with_rt
        with st.spinner("Preprocessing text..."):
            if len(received_with_rt) > 0:
                # Get custom stopwords and disabled metadata stopwords from session state
                custom_stopwords = st.session_state.get('custom_stopwords', set())
                disabled_metadata_stopwords = st.session_state.get('disabled_metadata_stopwords', set())
                processed_docs = preprocess_documents(
                    received_with_rt['text'],
                    lowercase=lowercase,
                    remove_stopwords=remove_stopwords,
                    remove_emojis_flag=remove_emojis_flag,
                    stem=stem,
                    lemmatize=lemmatize,
                    min_length=min_length,
                    custom_stopwords=custom_stopwords,
                    disabled_metadata_stopwords=disabled_metadata_stopwords,
                    remove_metadata_tokens=False  # Disable aggressive metadata filtering to keep real words
                )
                # Store for display
                st.session_state.processed_docs = processed_docs
                st.session_state.received_with_rt = received_with_rt
            else:
                processed_docs = []
                st.session_state.processed_docs = []
        
        # Display preprocessed data preview
        if 'processed_docs' in st.session_state and len(st.session_state.processed_docs) > 0:
            with st.expander("🔍 View Preprocessed Data", expanded=False):
                st.subheader("Preprocessing Preview")
                
                # Create a preview dataframe
                preview_data = []
                received_df = st.session_state.received_with_rt
                
                # Show first 20 examples
                num_preview = min(20, len(received_df))
                for i in range(num_preview):
                    original_text = str(received_df.iloc[i]['text'])[:200]  # First 200 chars
                    processed_tokens = st.session_state.processed_docs[i]
                    processed_text = ' '.join(processed_tokens) if processed_tokens else '(empty)'
                    
                    # Get contact name
                    contact_name = str(received_df.iloc[i].get('chat_name', ''))
                    if not contact_name or contact_name == 'Unknown Chat':
                        contact_name = str(received_df.iloc[i].get('participant', 'Unknown'))
                    
                    preview_data.append({
                        'Index': i,
                        'Contact': contact_name,
                        'Original Text': original_text,
                        'Processed Tokens': processed_text,
                        'Token Count': len(processed_tokens)
                    })
                
                preview_df = pd.DataFrame(preview_data)
                st.dataframe(preview_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("Preprocessing Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                total_tokens = sum(len(doc) for doc in st.session_state.processed_docs)
                avg_tokens = total_tokens / len(st.session_state.processed_docs) if len(st.session_state.processed_docs) > 0 else 0
                empty_docs = sum(1 for doc in st.session_state.processed_docs if len(doc) == 0)
                
                with col1:
                    st.metric("Total Documents", f"{len(st.session_state.processed_docs):,}")
                with col2:
                    st.metric("Total Tokens", f"{total_tokens:,}")
                with col3:
                    st.metric("Avg Tokens/Doc", f"{avg_tokens:.1f}")
                with col4:
                    st.metric("Empty Documents", f"{empty_docs:,}")
        
        # RQ1: Topics user tends not to engage with
        st.header("📌 RQ1: Topics You Tend Not to Engage With")
        st.markdown("""
        **Methodology**: For each received message, we compute response time until your next reply.
        Messages with longer response times (or no reply) are assigned higher "reluctance scores."
        We train LDA on received messages and weight topics by reluctance scores to identify
        topics you're less likely to engage with.
        """)
        
        if len(processed_docs) > 0 and len(received_with_rt) > 0:
            if st.button("Run RQ1 Analysis", key="rq1"):
                with st.spinner("Training topic model and computing reluctance scores..."):
                    try:
                        # Train LDA on received messages
                        model, dictionary, corpus, doc_topics = train_lda(
                            processed_docs,
                            num_topics=num_topics,
                            passes=passes,
                            alpha=alpha,
                            beta=beta,
                            use_mallet=use_mallet,
                            mallet_home=mallet_home if use_mallet else None
                        )
                        
                        # Rank reluctant topics (this also computes reluctance_score and adds it to a copy)
                        rq1_df = rank_reluctant_topics(
                            received_with_rt,
                            doc_topics,
                            model,
                            dictionary
                        )
                        
                        # Ensure received_with_rt has reluctance_score column
                        # rank_reluctant_topics computes it internally, so we need to add it
                        if 'reluctance_score' not in received_with_rt.columns:
                            from responses import compute_reluctance_score
                            cap_minutes = max_gap_minutes
                            reluctance_scores = []
                            for idx, row in received_with_rt.iterrows():
                                score = compute_reluctance_score(
                                    row.get('response_time_min', np.nan),
                                    row.get('got_reply', False),
                                    cap_minutes
                                )
                                reluctance_scores.append(score)
                            received_with_rt = received_with_rt.copy()
                            received_with_rt['reluctance_score'] = reluctance_scores
                        
                        # Get top reluctant topic IDs
                        top_topic_ids = rq1_df.head(5)['topic_id'].tolist()
                        
                        # Get message examples for top reluctant topics
                        topic_examples = get_reluctant_topic_examples(
                            received_with_rt,
                            doc_topics,
                            top_topic_ids,
                            num_examples=5
                        )
                        
                        # Get contact-level statistics
                        contact_stats = get_contact_reluctant_topic_stats(
                            received_with_rt,
                            doc_topics,
                            top_topic_ids
                        )
                        
                        st.session_state.rq1_df = rq1_df
                        st.session_state.rq1_model = model
                        st.session_state.rq1_dictionary = dictionary
                        st.session_state.rq1_doc_topics = doc_topics
                        st.session_state.rq1_topic_examples = topic_examples
                        st.session_state.rq1_contact_stats = contact_stats
                        
                        # Train fine-grained model for RQ4-7 (50 topics for more granularity)
                        with st.spinner("Training fine-grained topic model for RQ4-7 (50 topics)..."):
                            fine_model, fine_dictionary, fine_corpus, fine_doc_topics = train_lda(
                                processed_docs,
                                num_topics=50,
                                passes=passes,
                                alpha=alpha,
                                beta=beta,
                                use_mallet=use_mallet,
                                mallet_home=mallet_home if use_mallet else None
                            )
                            
                            st.session_state.fine_model = fine_model
                            st.session_state.fine_dictionary = fine_dictionary
                            st.session_state.fine_doc_topics = fine_doc_topics
                        
                        st.success("✅ RQ1 analysis complete (including fine-grained model for RQ4-7)")
                    except Exception as e:
                        st.error(f"❌ Error in RQ1 analysis: {e}")
                        st.exception(e)
        
        if 'rq1_df' in st.session_state:
            rq1_df = st.session_state.rq1_df
            st.subheader("Results")
            st.dataframe(rq1_df[['topic_id', 'top_words', 'reluctance_score', 'frequency', 'final_rank']])
            
            # Plots
            fig1 = plot_reluctant_topics(rq1_df, top_n=10, use_plotly=True)
            if fig1:
                try:
                    st.plotly_chart(fig1, use_container_width=True)
                except:
                    # Fallback to matplotlib if plotly fails
                    st.pyplot(fig1)
            
            top_topic_ids = rq1_df.head(5)['topic_id'].tolist()
            fig2 = plot_response_times_by_topic(
                received_with_rt,
                st.session_state.rq1_doc_topics,
                top_topic_ids,
                use_plotly=True
            )
            if fig2:
                try:
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.pyplot(fig2)
            
            # Message examples for top reluctant topics
            if 'rq1_topic_examples' in st.session_state:
                st.subheader("📝 Example Messages from Top Reluctant Topics")
                topic_examples = st.session_state.rq1_topic_examples
                
                if len(topic_examples) > 0:
                    # Group by topic_id for display
                    for topic_id in top_topic_ids[:3]:  # Show top 3 topics
                        topic_examples_subset = topic_examples[topic_examples['topic_id'] == topic_id]
                        if len(topic_examples_subset) > 0:
                            # Get topic words
                            topic_row = rq1_df[rq1_df['topic_id'] == topic_id]
                            if len(topic_row) > 0:
                                topic_words = topic_row.iloc[0]['top_words']
                                st.markdown(f"**Topic {topic_id}** ({topic_words}):")
                                
                                for idx, example in topic_examples_subset.head(5).iterrows():
                                    with st.expander(f"From {example['participant']} (Topic Prob: {example['topic_prob']:.3f}, Reluctance: {example['reluctance_score']:.3f})"):
                                        st.text(example['text'])
            
            # Contact-level statistics
            if 'rq1_contact_stats' in st.session_state:
                contact_stats = st.session_state.rq1_contact_stats
                
                if len(contact_stats) > 0:
                    st.subheader("👥 Contacts by High Reluctant Topic Messages")
                    fig3 = plot_contact_reluctant_topic_count(contact_stats, top_n=7, use_plotly=True)
                    if fig3:
                        try:
                            st.plotly_chart(fig3, use_container_width=True)
                        except:
                            st.pyplot(fig3)
                    
                    st.subheader("📊 Contacts by Average Reluctance Score")
                    fig4 = plot_contact_avg_reluctance(contact_stats, top_n=7, use_plotly=True)
                    if fig4:
                        try:
                            st.plotly_chart(fig4, use_container_width=True)
                        except:
                            st.pyplot(fig4)
                    
                    # RQ1 Enhancement: Proportion-based ranking
                    st.subheader("📈 Contacts by Proportion of High Reluctance Messages")
                    
                    # User-controlled minimum message filter
                    min_messages_rq1 = st.slider(
                        "Minimum total messages to include contact:",
                        min_value=50,
                        max_value=1000,
                        value=500,
                        step=50,
                        key="rq1_min_messages_proportion"
                    )
                    
                    st.markdown(f"*Showing contacts with {min_messages_rq1}+ messages, ranked by percentage of high reluctance messages*")
                    
                    # Get proportion-based statistics
                    proportion_stats = get_contact_reluctance_proportions(
                        received_with_rt,
                        st.session_state.rq1_doc_topics,
                        top_topic_ids,
                        min_messages=min_messages_rq1
                    )
                    
                    if len(proportion_stats) > 0:
                        fig5 = plot_contact_reluctance_proportions(proportion_stats, top_n=10, use_plotly=True)
                        if fig5:
                            try:
                                st.plotly_chart(fig5, use_container_width=True)
                            except:
                                st.pyplot(fig5)
                    else:
                        st.info(f"No contacts with {min_messages_rq1}+ messages found.")
        
        # RQ2: Most commonly discussed topics
        st.header("💭 RQ2: Most Commonly Discussed Topics")
        st.markdown("""
        **Methodology**: We train LDA on all messages (grouped by chat or time period) to identify
        the main topics of conversation. We then analyze topic distribution per contact and how
        topics evolve over time.
        """)
        
        if len(df) > 0:
            if st.button("Run RQ2 Analysis", key="rq2"):
                with st.spinner("Training topic model on all messages..."):
                    try:
                        # Preprocess all messages
                        custom_stopwords = st.session_state.get('custom_stopwords', set())
                        disabled_metadata_stopwords = st.session_state.get('disabled_metadata_stopwords', set())
                        all_processed = preprocess_documents(
                            df['text'],
                            lowercase=lowercase,
                            remove_stopwords=remove_stopwords,
                            remove_emojis_flag=remove_emojis_flag,
                            stem=stem,
                            lemmatize=lemmatize,
                            min_length=min_length,
                            custom_stopwords=custom_stopwords,
                            disabled_metadata_stopwords=disabled_metadata_stopwords,
                            remove_metadata_tokens=False  # Disable aggressive metadata filtering to keep real words
                        )
                        
                        # Train LDA
                        model, dictionary, corpus, doc_topics = train_lda(
                            all_processed,
                            num_topics=num_topics,
                            passes=passes,
                            alpha=alpha,
                            beta=beta,
                            use_mallet=use_mallet,
                            mallet_home=mallet_home if use_mallet else None
                        )
                        
                        # Get topic words and message counts (RQ2 Enhancement)
                        topic_message_counts = get_topic_message_counts(doc_topics, topic_threshold=0.3)
                        
                        topic_words_list = []
                        for topic_id in range(num_topics):
                            words = get_topic_words(model, dictionary, topic_id, num_words=30)
                            topic_words_list.append({
                                'topic_id': topic_id,
                                'top_words': ', '.join([w for w, _ in words]),
                                'message_count': topic_message_counts.get(topic_id, 0)
                            })
                        topic_words_df = pd.DataFrame(topic_words_list)
                        
                        # Per-contact topic mix
                        contact_topic_df = get_per_contact_topic_mix(df, doc_topics, top_n_contacts=10)
                        
                        # Topic over time
                        time_topic_df = topic_over_time(df, doc_topics, freq='M')
                        
                        st.session_state.rq2_model = model
                        st.session_state.rq2_dictionary = dictionary
                        st.session_state.rq2_doc_topics = doc_topics
                        st.session_state.rq2_topic_words = topic_words_df
                        st.session_state.rq2_contact_topic = contact_topic_df
                        st.session_state.rq2_time_topic = time_topic_df
                        
                        st.success("✅ RQ2 analysis complete")
                    except Exception as e:
                        st.error(f"❌ Error in RQ2 analysis: {e}")
                        st.exception(e)
        
        if 'rq2_topic_words' in st.session_state:
            st.subheader("Topic Words")
            st.dataframe(st.session_state.rq2_topic_words)
            
            st.subheader("Topic Distribution by Contact")
            contact_topic_df = st.session_state.rq2_contact_topic
            if len(contact_topic_df) > 0:
                fig = plot_per_contact_topic_mix(contact_topic_df, use_plotly=True)
                if fig:
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.pyplot(fig)
            
            st.subheader("Topic Prevalence Over Time")
            time_topic_df = st.session_state.rq2_time_topic
            if len(time_topic_df) > 0:
                # RQ2 Enhancement: Plot all topics, not just top 5
                fig = plot_topic_prevalence_over_time(time_topic_df, top_n_topics=None, use_plotly=True)
                if fig:
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.pyplot(fig)
        
        # RQ3: Group vs one-to-one responsiveness
        st.header("👥 RQ3: Group vs One-to-One Responsiveness")
        st.markdown("""
        **Methodology**: We compare response rates and response times between one-to-one conversations
        and group chats of different sizes. We identify contacts you're more likely to respond to
        in groups vs individually.
        """)
        
        if len(received_with_rt) > 0:
            if st.button("Run RQ3 Analysis", key="rq3"):
                with st.spinner("Computing group vs DM statistics..."):
                    try:
                        rq3_stats = group_vs_dm_stats(received_with_rt)
                        st.session_state.rq3_stats = rq3_stats
                        st.success("✅ RQ3 analysis complete")
                    except Exception as e:
                        st.error(f"❌ Error in RQ3 analysis: {e}")
                        st.exception(e)
        
        if 'rq3_stats' in st.session_state:
            rq3_stats = st.session_state.rq3_stats
            
            st.subheader("Summary Statistics")
            st.dataframe(rq3_stats.get('by_category', pd.DataFrame()))
            
            # Plots
            fig1 = plot_group_vs_dm_response(rq3_stats, use_plotly=True)
            if fig1:
                try:
                    st.plotly_chart(fig1, use_container_width=True)
                except:
                    st.pyplot(fig1)
            
            fig2 = plot_response_time_boxplot(received_with_rt, use_plotly=True)
            if fig2:
                try:
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.pyplot(fig2)
            
            # Contacts that prefer group responses (RQ3 Enhancements)
            contact_comp = rq3_stats.get('contact_comparison', pd.DataFrame())
            if len(contact_comp) > 0:
                # Add reply_difference and total_messages columns
                contact_comp = contact_comp.copy()
                contact_comp['reply_difference'] = contact_comp['group_reply_rate'] - contact_comp['one_on_one_reply_rate']
                contact_comp['total_messages_with_contact'] = contact_comp['one_on_one_count'] + contact_comp['group_count']
                
                # User-controlled filtering
                st.subheader("Contacts You Respond to More in Groups")
                min_messages_filter = st.slider(
                    "Minimum total messages to include:",
                    min_value=0,
                    max_value=int(contact_comp['total_messages_with_contact'].max()),
                    value=10,
                    step=10,
                    key="rq3_min_messages"
                )
                
                # Apply filters
                prefers_group = contact_comp[
                    (contact_comp['reply_difference'] >= 0.05) &
                    (contact_comp['total_messages_with_contact'] >= min_messages_filter)
                ].sort_values('reply_difference', ascending=False)
                
                if len(prefers_group) > 0:
                    st.dataframe(prefers_group[['participant', 'one_on_one_reply_rate', 
                                                'group_reply_rate', 'reply_difference',
                                                'total_messages_with_contact', 
                                                'one_on_one_count', 'group_count']])
                else:
                    st.info("No contacts match the current filter criteria.")
        
        # RQ4: Conversation Starter Topics (Fine-Grained Model)
        st.header("🚀 RQ4: Conversation Starter Topics")
        st.markdown("""
        **Methodology**: Using **fine-grained topic modeling (50 topics)** for higher resolution.
        Topics that are conversation starters are more likely to receive replies,
        have shorter response times, and often appear at the beginning of message threads.
        
        **Fine-Grained Model**: Trained with 50 LDA topics (vs 10-30 for RQ1-3) to capture more nuanced subtopics.
        """)
        
        if len(received_with_rt) > 0 and 'fine_doc_topics' in st.session_state:
            if st.button("Run RQ4 Analysis", key="rq4"):
                with st.spinner("Analyzing conversation starter topics..."):
                    try:
                        rq4_starters = analyze_conversation_starters(
                            received_with_rt,
                            st.session_state.fine_doc_topics,
                            topic_threshold=0.3,
                            session_gap_minutes=180.0
                        )
                        
                        # Add top_words column
                        rq4_starters['top_words'] = rq4_starters['topic_id'].apply(
                            lambda tid: ', '.join([w for w, _ in get_topic_words(
                                st.session_state.fine_model, 
                                st.session_state.fine_dictionary, 
                                tid, 
                                num_words=30
                            )])
                        )
                        
                        st.session_state.rq4_starters = rq4_starters
                        st.success("✅ RQ4 analysis complete")
                    except Exception as e:
                        st.error(f"❌ Error in RQ4 analysis: {e}")
                        st.exception(e)
        
        if 'rq4_starters' in st.session_state and 'fine_model' in st.session_state:
            rq4_starters = st.session_state.rq4_starters
            
            st.subheader("Top Conversation Starter Topics")
            st.dataframe(rq4_starters[['topic_id', 'top_words', 'starter_score', 'reply_rate', 
                                      'avg_response_time', 'starter_probability', 'message_count']].head(15))
            
            fig = plot_conversation_starters(rq4_starters, st.session_state.fine_model, 
                                            st.session_state.fine_dictionary, top_n=10, use_plotly=True)
            if fig:
                try:
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.pyplot(fig)
        
        # RQ5: Conversation Ender Topics (Fine-Grained Model)
        st.header("🔚 RQ5: Conversation Ender Topics")
        st.markdown("""
        **Methodology**: Using **fine-grained topic modeling (50 topics)** for higher resolution.
        Topics that end conversations are associated with no replies,
        longer response times, or appear as the last messages in conversation sessions.
        
        **Fine-Grained Model**: Same 50-topic LDA model as RQ4 to capture nuanced conversation dynamics.
        """)
        
        if len(received_with_rt) > 0 and 'fine_doc_topics' in st.session_state:
            if st.button("Run RQ5 Analysis", key="rq5"):
                with st.spinner("Analyzing conversation ender topics..."):
                    try:
                        rq5_enders = analyze_conversation_enders(
                            received_with_rt,
                            st.session_state.fine_doc_topics,
                            topic_threshold=0.3,
                            session_gap_minutes=180.0
                        )
                        
                        # Add top_words column
                        rq5_enders['top_words'] = rq5_enders['topic_id'].apply(
                            lambda tid: ', '.join([w for w, _ in get_topic_words(
                                st.session_state.fine_model, 
                                st.session_state.fine_dictionary, 
                                tid, 
                                num_words=30
                            )])
                        )
                        
                        st.session_state.rq5_enders = rq5_enders
                        st.success("✅ RQ5 analysis complete")
                    except Exception as e:
                        st.error(f"❌ Error in RQ5 analysis: {e}")
                        st.exception(e)
        
        if 'rq5_enders' in st.session_state and 'fine_model' in st.session_state:
            rq5_enders = st.session_state.rq5_enders
            
            st.subheader("Top Conversation Ender Topics")
            st.dataframe(rq5_enders[['topic_id', 'top_words', 'ender_score', 'no_reply_rate', 
                                     'avg_response_time', 'ender_probability', 'message_count']].head(15))
            
            fig = plot_conversation_enders(rq5_enders, st.session_state.fine_model, 
                                          st.session_state.fine_dictionary, top_n=10, use_plotly=True)
            if fig:
                try:
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.pyplot(fig)
        
        # RQ6: Topics by Closeness (Close Contacts vs Acquaintances) - Fine-Grained Model
        st.header("👨‍👩‍👧‍👦 RQ6: Topics by Closeness")
        st.markdown("""
        **Methodology**: Using **fine-grained topic modeling (50 topics)** trained on ALL messages.
        Compare topic distributions between close contacts (high message count
        and mutual reply rate) and acquaintances.
        
        **Fine-Grained Model**: Single 50-topic model trained on all texts, then split by contact type.
        **Odds Ratio**: (topic freq among close) / (topic freq among acquaintances).
        Higher odds ratio = more prevalent among close contacts.
        """)
        
        # User controls for closeness definition
        col1, col2 = st.columns(2)
        with col1:
            close_min_messages = st.number_input(
                "Minimum messages for close contact:",
                min_value=10,
                value=100,
                step=10,
                key="rq6_min_messages"
            )
        with col2:
            close_min_reply_rate = st.slider(
                "Minimum reply rate for close contact:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="rq6_min_reply_rate"
            )
        
        if len(received_with_rt) > 0 and 'fine_doc_topics' in st.session_state:
            if st.button("Run RQ6 Analysis", key="rq6"):
                with st.spinner("Analyzing topics by closeness..."):
                    try:
                        rq6_closeness = analyze_topics_by_closeness(
                            received_with_rt,
                            st.session_state.fine_doc_topics,
                            topic_threshold=0.3,
                            close_contact_min_messages=close_min_messages,
                            close_contact_min_reply_rate=close_min_reply_rate
                        )
                        
                        # Add top_words column
                        rq6_closeness['top_words'] = rq6_closeness['topic_id'].apply(
                            lambda tid: ', '.join([w for w, _ in get_topic_words(
                                st.session_state.fine_model, 
                                st.session_state.fine_dictionary, 
                                tid, 
                                num_words=30
                            )])
                        )
                        
                        st.session_state.rq6_closeness = rq6_closeness
                        st.success("✅ RQ6 analysis complete")
                    except Exception as e:
                        st.error(f"❌ Error in RQ6 analysis: {e}")
                        st.exception(e)
        
        if 'rq6_closeness' in st.session_state and 'fine_model' in st.session_state:
            rq6_closeness = st.session_state.rq6_closeness
            
            st.subheader("Topics by Closeness (Odds Ratio)")
            st.dataframe(rq6_closeness[['topic_id', 'top_words', 'odds_ratio', 'category', 
                                       'close_frequency', 'acquaintance_frequency', 
                                       'close_count', 'acquaintance_count']].head(15))
            
            fig = plot_topics_by_closeness(rq6_closeness, st.session_state.fine_model, 
                                          st.session_state.fine_dictionary, top_n=10, use_plotly=True)
            if fig:
                try:
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.pyplot(fig)
        
        # RQ7: Topics by Time of Day (Fine-Grained Model)
        st.header("🕐 RQ7: Topics by Time of Day")
        st.markdown("""
        **Methodology**: Using **fine-grained topic modeling (50 topics)** for higher resolution.
        Analyze which topics are prevalent during different times of the day.
        
        **Fine-Grained Model**: Same 50-topic LDA model to identify time-specific conversation patterns.
        
        **Time Periods**:
        - **Morning**: 6:00 AM - 12:00 PM
        - **Afternoon**: 12:00 PM - 6:00 PM
        - **Evening**: 6:00 PM - 10:00 PM
        - **Night**: 10:00 PM - 6:00 AM
        
        **Values Shown**: Normalized frequencies (proportion of messages in each time period assigned to each topic).
        """)
        
        if len(received_with_rt) > 0 and 'fine_doc_topics' in st.session_state:
            if st.button("Run RQ7 Analysis", key="rq7"):
                with st.spinner("Analyzing topics by time of day..."):
                    try:
                        rq7_time = analyze_topics_by_time_of_day(
                            received_with_rt,
                            st.session_state.fine_doc_topics,
                            topic_threshold=0.3
                        )
                        
                        # Add top_words column
                        rq7_time['top_words'] = rq7_time['topic_id'].apply(
                            lambda tid: ', '.join([w for w, _ in get_topic_words(
                                st.session_state.fine_model, 
                                st.session_state.fine_dictionary, 
                                tid, 
                                num_words=30
                            )])
                        )
                        
                        st.session_state.rq7_time = rq7_time
                        st.success("✅ RQ7 analysis complete")
                    except Exception as e:
                        st.error(f"❌ Error in RQ7 analysis: {e}")
                        st.exception(e)
        
        if 'rq7_time' in st.session_state and 'fine_model' in st.session_state:
            rq7_time = st.session_state.rq7_time
            
            st.subheader("Topics by Time of Day")
            # Reorder columns to show top_words first
            display_cols = ['topic_id', 'top_words', 'morning', 'afternoon', 'evening', 'night']
            st.dataframe(rq7_time[display_cols].head(30))
            
            fig = plot_topics_by_time_heatmap(rq7_time, st.session_state.fine_model, 
                                             st.session_state.fine_dictionary, use_plotly=True)
            if fig:
                try:
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.pyplot(fig)
        
        # RQ8: Fine-Grained Topics You Tend Not to Engage With
        st.header("🔬 RQ8: Fine-Grained Topics You Tend Not to Engage With")
        st.markdown("""
        **Methodology**: This is a **fine-grained version of RQ1** using TF-IDF + K-Means.
        
        Instead of traditional LDA, we use a simpler but effective approach:
        1. **TF-IDF Vectorization**: Captures word importance in messages
        2. **Dimensionality Reduction**: SVD (like PCA, but faster)
        3. **Clustering**: K-Means (fast, reliable clustering)
        4. **Topic Words**: Top words per cluster from TF-IDF scores
        
        **Reluctance Logic**: Same as RQ1 - measures response time and no-reply rate.
        
        **Advantages**:
        - No heavy dependencies (only scikit-learn, already installed)
        - Fast (completes in seconds)
        - Reliable results
        """)
        
        # User controls (always show)
        n_topics_rq8 = st.slider(
            "Number of fine-grained topics to discover:",
            min_value=15,
            max_value=50,
            value=30,
            step=5,
            key="rq8_n_topics",
            help="More topics = finer granularity. 30 is a good balance."
        )
        
        if len(received_with_rt) > 0 and len(processed_docs) > 0:
            if st.button("Run RQ8 Analysis", key="rq8"):
                try:
                    from simple_topics import train_simple_topic_model
                    
                    with st.spinner(f"Training fine-grained topic model ({n_topics_rq8} topics)..."):
                        # Get raw text (not preprocessed tokens)
                        raw_texts = received_with_rt['text'].fillna('').tolist()
                        
                        # Filter out empty texts
                        valid_indices = [i for i, text in enumerate(raw_texts) if text.strip()]
                        valid_texts = [raw_texts[i] for i in valid_indices]
                        valid_df = received_with_rt.iloc[valid_indices].copy().reset_index(drop=True)
                        
                        if len(valid_texts) == 0:
                            st.error("No valid texts found for topic modeling.")
                        else:
                            # Train simple topic model
                            def progress_update(msg):
                                st.info(msg)
                            
                            simple_model, topic_labels = train_simple_topic_model(
                                valid_texts,
                                n_topics=n_topics_rq8,
                                progress_callback=progress_update
                            )
                            
                            st.info(f"Discovered {n_topics_rq8} fine-grained topics")
                            
                            # Rank topics by reluctance
                            rq8_df = rank_reluctant_topics_embedding(
                                valid_df,
                                topic_labels,
                                simple_model,
                                cap_minutes=max_gap_minutes
                            )
                            
                            st.session_state.rq8_df = rq8_df
                            st.session_state.rq8_model = simple_model
                            st.session_state.rq8_topic_labels = topic_labels
                            
                            st.success(f"✅ RQ8 analysis complete - {n_topics_rq8} fine-grained topics discovered")
                
                except Exception as e:
                    st.error(f"❌ Error in RQ8 analysis: {e}")
                    st.exception(e)
        
        # Display RQ8 results if available
        if 'rq8_df' in st.session_state:
                rq8_df = st.session_state.rq8_df
                
                st.subheader("Fine-Grained Avoided Topics")
                st.markdown("""
                These are fine-grained topics you tend to avoid or delay responding to.
                Each topic is discovered automatically based on semantic similarity of messages.
                """)
                
                # Show table
                display_cols = ['topic_id', 'top_words', 'reluctance_score', 'frequency', 'final_rank']
                st.dataframe(rq8_df[display_cols].head(20))
                
                # Plot
                fig = plot_reluctant_embedding_topics(rq8_df, top_n=15, use_plotly=True)
                if fig:
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.pyplot(fig)
        
        # RQ9: GPT-Powered Texting Behavior Chatbot
        st.header("🤖 RQ9: Ask a Chatbot About Your Texting Behavior")
        st.markdown("""
        **Ask questions about your texting patterns in plain English!**
        
        This chatbot uses GPT-4 to interpret your texting behavior based on **aggregated statistics only**.
        
        **Privacy Guarantee**: 
        - ✅ Only sends aggregated patterns from RQ1-8
        - ✅ NO raw message texts are sent to the API
        - ✅ NO phone numbers, emails, or contact names are sent
        - ✅ Contacts are anonymized as "Contact A", "Contact B", etc.
        
        **Requires**: OpenAI API key (your data stays private, only aggregates are sent)
        """)
        
        # API Key Input
        api_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            help="Your API key is only stored in session state and never saved to disk.",
            key="openai_api_key"
        )
        
        if not api_key:
            st.info("👆 Enter your OpenAI API key above to enable the chatbot.")
            st.markdown("""
            **Don't have an API key?** Get one at: https://platform.openai.com/api-keys
            
            **Cost**: ~$0.01-0.03 per question (using GPT-4o)
            """)
        else:
            st.success("✅ API key entered (only stored in session)")
            
            # Initialize conversation history
            if 'rq9_conversation' not in st.session_state:
                st.session_state.rq9_conversation = []
            
            # Build behavior summary from all available RQ results
            rq_results = {
                'rq1_df': st.session_state.get('rq1_df'),
                'rq2_topic_words': st.session_state.get('rq2_topic_words'),
                'rq2_contact_topic': st.session_state.get('rq2_contact_topic'),
                'rq3_stats': st.session_state.get('rq3_stats'),
                'rq4_starters': st.session_state.get('rq4_starters'),
                'rq5_enders': st.session_state.get('rq5_enders'),
                'rq6_closeness': st.session_state.get('rq6_closeness'),
                'rq7_time': st.session_state.get('rq7_time'),
                'rq8_df': st.session_state.get('rq8_df'),
                'stats': {
                    'total_messages': len(df) if len(df) > 0 else 0,
                    'messages_received': len(df[df['direction'] == 'in']) if len(df) > 0 else 0,
                    'messages_sent': len(df[df['direction'] == 'out']) if len(df) > 0 else 0,
                    'reply_rate': received_with_rt['got_reply'].mean() if len(received_with_rt) > 0 else 0,
                    'median_response_time': received_with_rt['response_time_min'].median() if len(received_with_rt) > 0 else 0
                }
            }
            
            # Check if any RQ has been run
            has_results = any([
                rq_results['rq1_df'] is not None,
                rq_results['rq2_topic_words'] is not None,
                rq_results['rq3_stats'] is not None,
                rq_results['rq4_starters'] is not None,
                rq_results['rq5_enders'] is not None,
                rq_results['rq6_closeness'] is not None,
                rq_results['rq7_time'] is not None,
                rq_results['rq8_df'] is not None
            ])
            
            if not has_results:
                st.warning("⚠️ Run at least one RQ analysis (RQ1-8) before using the chatbot.")
            else:
                # Build behavior summary (rebuild on each run to capture latest results)
                st.session_state.rq9_behavior_summary = build_behavior_summary(rq_results)
                
                # Show example questions
                with st.expander("💡 Example Questions"):
                    st.markdown("**Try asking:**")
                    for example in get_example_questions():
                        st.markdown(f"- {example}")
                
                # Chat interface
                st.subheader("💬 Chat with Your Texting Behavior AI")
                
                # Question input
                user_question = st.text_area(
                    "Ask a question about your texting patterns:",
                    placeholder="e.g., What topics do I reply to the fastest?",
                    key="rq9_question_input",
                    height=100
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    ask_button = st.button("Ask", key="rq9_ask", type="primary")
                with col2:
                    clear_button = st.button("Clear Conversation", key="rq9_clear")
                
                if clear_button:
                    st.session_state.rq9_conversation = []
                    st.rerun()
                
                if ask_button and user_question.strip():
                    with st.spinner("Asking GPT-4..."):
                        try:
                            response, updated_history = run_rq9_chatbot(
                                user_question,
                                st.session_state.rq9_behavior_summary,
                                api_key,
                                st.session_state.rq9_conversation
                            )
                            
                            st.session_state.rq9_conversation = updated_history
                            
                        except Exception as e:
                            st.error(f"❌ Error calling OpenAI API: {e}")
                            st.info("Check that your API key is valid and you have credits available.")
                
                # Display conversation history
                if st.session_state.rq9_conversation:
                    st.subheader("Conversation")
                    
                    for msg in st.session_state.rq9_conversation:
                        if msg['role'] == 'user':
                            with st.chat_message("user"):
                                st.write(msg['content'])
                        elif msg['role'] == 'assistant':
                            with st.chat_message("assistant"):
                                st.write(msg['content'])
                
                # Show privacy reminder
                with st.expander("🔒 Privacy & Data Sent to OpenAI"):
                    st.markdown("""
                    **What is sent to OpenAI:**
                    - Aggregated statistics (reply rates, response times, topic frequencies)
                    - Topic keywords (extracted by LDA/K-Means)
                    - Anonymized contact labels ("Contact A", "Contact B")
                    
                    **What is NOT sent:**
                    - ❌ Raw message texts
                    - ❌ Phone numbers or email addresses
                    - ❌ Real contact/group names
                    - ❌ Timestamps or locations
                    - ❌ Any personally identifiable information
                    
                    You can view the exact data summary that's sent by expanding the section below.
                    """)
                    
                    if st.checkbox("Show behavior summary sent to GPT", key="show_summary"):
                        st.code(st.session_state.rq9_behavior_summary, language="markdown")
        
        # Export section
        st.header("💾 Export Results")
        if st.button("Export Report"):
            try:
                exports_dir = ensure_exports_dir()
                
                # Collect data for export
                rq1_df_export = st.session_state.get('rq1_df', pd.DataFrame())
                rq2_contact_df_export = st.session_state.get('rq2_contact_topic', pd.DataFrame())
                rq2_time_df_export = st.session_state.get('rq2_time_topic', pd.DataFrame())
                rq3_stats_export = st.session_state.get('rq3_stats', {})
                
                # Collect plots
                plots_dict = {}
                if 'rq1_df' in st.session_state and len(rq1_df_export) > 0:
                    plots_dict['RQ1 Reluctant Topics'] = plot_reluctant_topics(rq1_df_export, top_n=10, use_plotly=False)
                
                # Generate HTML report
                html_content = export_html_report(
                    rq1_df_export, rq2_contact_df_export, rq2_time_df_export,
                    rq3_stats_export, plots_dict
                )
                
                # Save HTML
                html_path = os.path.join(exports_dir, "imessage_report.html")
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Save CSV tables
                csv_path = os.path.join(exports_dir, "imessage_tables.csv")
                with open(csv_path, 'w', encoding='utf-8') as f:
                    if len(rq1_df_export) > 0:
                        f.write("=== RQ1: Reluctant Topics ===\n")
                        rq1_df_export.to_csv(f, index=False)
                        f.write("\n\n")
                    if len(rq2_contact_df_export) > 0:
                        f.write("=== RQ2: Topic Distribution by Contact ===\n")
                        rq2_contact_df_export.to_csv(f, index=False)
                        f.write("\n\n")
                    if len(rq2_time_df_export) > 0:
                        f.write("=== RQ2: Topic Prevalence Over Time ===\n")
                        rq2_time_df_export.to_csv(f, index=False)
                        f.write("\n\n")
                    if rq3_stats_export and 'by_category' in rq3_stats_export:
                        f.write("=== RQ3: Group vs DM Statistics ===\n")
                        rq3_stats_export['by_category'].to_csv(f, index=False)
                
                st.success(f"✅ Report exported to {html_path}")
                st.success(f"✅ Tables exported to {csv_path}")
            except Exception as e:
                st.error(f"❌ Error exporting: {e}")
                st.exception(e)
        
        # Cleanup temp files
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            if st.button("🗑️ Delete Temporary Files"):
                try:
                    shutil.rmtree(st.session_state.temp_dir)
                    st.session_state.temp_dir = None
                    st.success("✅ Temporary files deleted")
                except Exception as e:
                    st.warning(f"⚠️ Could not delete temp files: {e}")


if __name__ == "__main__":
    main()

