"""Simple message viewer for chat.db files."""

import streamlit as st
import pandas as pd
import sqlite3
import tempfile
import os
from datetime import datetime, timezone
from data_loader import load_messages, merge_wal_if_needed
from utils import create_participant_map


def convert_apple_timestamp(apple_time: float):
    """Convert Apple timestamp to datetime."""
    if pd.isna(apple_time) or apple_time == 0:
        return None
    
    # Detect if nanoseconds
    if abs(apple_time) > 1e12:
        seconds = apple_time / 1e9
    else:
        seconds = apple_time
    
    apple_epoch = datetime(2001, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    from datetime import timedelta
    return apple_epoch + timedelta(seconds=seconds)


def main():
    st.set_page_config(
        page_title="iMessage Viewer",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 iMessage Database Viewer")
    st.markdown("**Simple viewer for browsing messages in your chat.db file**")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        deidentify = st.checkbox("De-identify Participants", value=True)
        show_raw_ids = st.checkbox("Show Raw IDs", value=False)
        
        st.header("🔍 Filters")
        search_text = st.text_input("Search in messages", "")
        direction_filter = st.selectbox("Message Direction", ["All", "Received", "Sent"])
        
        st.header("📅 Date Range")
        use_date_filter = st.checkbox("Filter by Date", value=False)
        date_start = st.date_input("Start Date", value=None)
        date_end = st.date_input("End Date", value=None)
    
    # File upload
    st.header("📁 Upload Database File")
    uploaded_files = st.file_uploader(
        "Upload chat.db (and optionally chat.db-wal, chat.db-shm)",
        type=['db', 'db-wal', 'db-shm'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Save uploaded files
        temp_dir = tempfile.mkdtemp()
        db_path = None
        wal_path = None
        shm_path = None
        
        files_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        
        for f in files_list:
            file_path = os.path.join(temp_dir, f.name)
            with open(file_path, "wb") as out_file:
                out_file.write(f.getbuffer())
            
            if f.name == "chat.db":
                db_path = file_path
            elif f.name == "chat.db-wal":
                wal_path = file_path
            elif f.name == "chat.db-shm":
                shm_path = file_path
        
        if db_path is None:
            st.error("❌ Please upload chat.db file")
        else:
            if st.button("Load Messages", type="primary"):
                with st.spinner("Loading messages..."):
                    try:
                        # Load messages using our existing function
                        df = load_messages(files_list, deidentify=deidentify)
                        
                        # Add raw identifiers if requested
                        if show_raw_ids and not deidentify:
                            # Reload without de-identification to get raw IDs
                            df_raw = load_messages(files_list, deidentify=False)
                            if 'handle_identifier' in df_raw.columns:
                                df['raw_identifier'] = df_raw['handle_identifier']
                        
                        st.session_state.messages_df = df
                        st.session_state.temp_dir = temp_dir
                        st.success(f"✅ Loaded {len(df):,} messages")
                    except Exception as e:
                        st.error(f"❌ Error loading messages: {e}")
                        st.exception(e)
    
    # Display messages
    if 'messages_df' in st.session_state and st.session_state.messages_df is not None:
        df = st.session_state.messages_df.copy()
        
        # Apply filters
        if search_text:
            df = df[df['text'].str.contains(search_text, case=False, na=False)]
        
        if direction_filter == "Received":
            df = df[df['direction'] == 'in']
        elif direction_filter == "Sent":
            df = df[df['direction'] == 'out']
        
        if use_date_filter and 'timestamp_local' in df.columns:
            if date_start:
                df = df[df['timestamp_local'] >= pd.Timestamp(date_start)]
            if date_end:
                df = df[df['timestamp_local'] <= pd.Timestamp(date_end) + pd.Timedelta(days=1)]
        
        # Sort by timestamp
        if 'timestamp_local' in df.columns:
            df = df.sort_values('timestamp_local', ascending=False)
        
        st.header(f"📨 Messages ({len(df):,} shown)")
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", f"{len(df):,}")
        with col2:
            st.metric("Received", f"{(df['direction'] == 'in').sum():,}")
        with col3:
            st.metric("Sent", f"{(df['direction'] == 'out').sum():,}")
        with col4:
            unique_chats = df['chat_id'].nunique() if 'chat_id' in df.columns else 0
            st.metric("Unique Chats", f"{unique_chats:,}")
        
        # Pagination
        messages_per_page = st.slider("Messages per page", 10, 100, 50, 10)
        
        if len(df) > 0:
            total_pages = (len(df) - 1) // messages_per_page + 1
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page - 1) * messages_per_page
            end_idx = start_idx + messages_per_page
            page_df = df.iloc[start_idx:end_idx]
            
            st.info(f"Showing messages {start_idx + 1} to {min(end_idx, len(df))} of {len(df)}")
            
            # Display messages
            for idx, row in page_df.iterrows():
                with st.container():
                    # Message header
                    direction_icon = "📥" if row['direction'] == 'in' else "📤"
                    direction_text = "Received" if row['direction'] == 'in' else "Sent"
                    
                    col1, col2, col3 = st.columns([1, 3, 2])
                    
                    with col1:
                        st.markdown(f"**{direction_icon} {direction_text}**")
                    
                    with col2:
                        participant = row.get('participant', 'Unknown')
                        if show_raw_ids and 'raw_identifier' in row:
                            st.markdown(f"**From/To:** {participant} ({row['raw_identifier']})")
                        else:
                            st.markdown(f"**From/To:** {participant}")
                    
                    with col3:
                        if pd.notna(row.get('timestamp_local')):
                            timestamp = row['timestamp_local']
                            if isinstance(timestamp, datetime):
                                st.markdown(f"**{timestamp.strftime('%Y-%m-%d %H:%M:%S')}**")
                            else:
                                st.markdown(f"**{timestamp}**")
                        else:
                            st.markdown("**Date: N/A**")
                    
                    # Message text
                    message_text = row.get('text', '')
                    if message_text:
                        st.text_area(
                            "Message",
                            value=message_text,
                            height=100,
                            key=f"msg_{idx}",
                            disabled=True,
                            label_visibility="collapsed"
                        )
                    else:
                        st.markdown("*[No text content]*")
                    
                    # Additional info
                    if 'chat_name' in row and pd.notna(row.get('chat_name')):
                        st.caption(f"Chat: {row['chat_name']} | Group Size: {row.get('group_size', 'N/A')}")
                    
                    st.divider()
        else:
            st.warning("No messages match the current filters.")
        
        # Export option
        st.header("💾 Export")
        if st.button("Export Filtered Messages to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"messages_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Cleanup
        if 'temp_dir' in st.session_state and st.session_state.temp_dir:
            if st.button("🗑️ Delete Temporary Files"):
                import shutil
                try:
                    shutil.rmtree(st.session_state.temp_dir)
                    st.session_state.temp_dir = None
                    st.success("✅ Temporary files deleted")
                except Exception as e:
                    st.warning(f"⚠️ Could not delete temp files: {e}")


if __name__ == "__main__":
    main()

