"""Data loading, WAL merging, SQL queries, and time conversions."""

import sqlite3
import pandas as pd
import numpy as np
import tempfile
import os
import re
from typing import List, Optional, Tuple, Any, Union
from datetime import datetime, timezone

# Optional streamlit import for UI messages
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

from utils import create_participant_map

# Try to import biplist for parsing binary plists
try:
    import biplist
    BIPLIST_AVAILABLE = True
except ImportError:
    BIPLIST_AVAILABLE = False

# plistlib is in standard library (Python 3.4+ supports binary plists)
try:
    import plistlib
    PLISTLIB_AVAILABLE = True
except ImportError:
    PLISTLIB_AVAILABLE = False


def merge_wal_if_needed(db_path: str, wal_path: Optional[str] = None, 
                        shm_path: Optional[str] = None) -> str:
    """
    Merge WAL file into database if present, creating a merged temporary database.
    
    Args:
        db_path: Path to chat.db
        wal_path: Optional path to chat.db-wal
        shm_path: Optional path to chat.db-shm (not used but checked)
        
    Returns:
        Path to the database to use (merged or original)
    """
    if wal_path is None or not os.path.exists(wal_path):
        return db_path
    
    # Create a temporary merged database
    temp_dir = tempfile.gettempdir()
    merged_db_path = os.path.join(temp_dir, f"chat_merged_{os.getpid()}.db")
    merged_wal_path = f"{merged_db_path}-wal"
    
    try:
        import shutil
        
        # Copy the original database
        shutil.copy2(db_path, merged_db_path)
        
        # Copy the WAL file to the same location
        shutil.copy2(wal_path, merged_wal_path)
        
        # Open the merged database and checkpoint WAL
        con = sqlite3.connect(merged_db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        
        # Checkpoint the WAL into the main database
        con.execute("PRAGMA wal_checkpoint(FULL);")
        
        # Vacuum to ensure everything is merged
        con.execute("VACUUM;")
        con.close()
        
        # Remove the WAL file after checkpointing
        if os.path.exists(merged_wal_path):
            try:
                os.remove(merged_wal_path)
            except:
                pass  # Ignore if can't remove
        
        if STREAMLIT_AVAILABLE:
            st.info(f"✅ Merged WAL file. Using merged database.")
        else:
            print(f"✅ Merged WAL file. Using merged database.")
        return merged_db_path
        
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.warning(f"⚠️ Failed to merge WAL: {e}. Using original database.")
        else:
            print(f"⚠️ Failed to merge WAL: {e}. Using original database.")
        # Clean up on error
        if os.path.exists(merged_db_path):
            try:
                os.remove(merged_db_path)
            except:
                pass
        return db_path


def detect_columns(con: sqlite3.Connection, table: str) -> List[str]:
    """Detect which columns exist in a table."""
    try:
        cursor = con.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cursor.fetchall()]
    except:
        return []


def convert_apple_timestamp(apple_time: float) -> Tuple[datetime, datetime]:
    """
    Convert Apple timestamp to UTC and local datetime.
    
    Apple epoch is 2001-01-01 UTC.
    Timestamps can be in seconds or nanoseconds.
    
    Args:
        apple_time: Apple timestamp (seconds or nanoseconds since 2001-01-01)
        
    Returns:
        Tuple of (utc_datetime, local_datetime)
    """
    if pd.isna(apple_time) or apple_time == 0:
        return None, None
    
    # Detect if nanoseconds (if abs value > 1e12, likely nanoseconds)
    if abs(apple_time) > 1e12:
        seconds = apple_time / 1e9
    else:
        seconds = apple_time
    
    # Apple epoch: 2001-01-01 00:00:00 UTC
    apple_epoch = datetime(2001, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    # Convert to UTC
    from datetime import timedelta
    utc_dt = apple_epoch + timedelta(seconds=seconds)
    
    # Convert to local (naive for now, will handle timezone later if needed)
    local_dt = utc_dt.replace(tzinfo=None)
    
    return utc_dt, local_dt


def clean_rtf_content(text: str) -> str:
    r"""
    Clean RTF (Rich Text Format) control codes and formatting from text.
    
    RTF uses control codes like \i, \i0, \i1, \b, \b0, etc. and braces {} for formatting.
    This function removes RTF control sequences while preserving actual text content.
    
    Args:
        text: Text that may contain RTF formatting
        
    Returns:
        Cleaned text with RTF control codes removed
    """
    if not text or not isinstance(text, str):
        return text
    
    # Check if text contains RTF indicators
    # RTF typically starts with {\rtf or contains RTF control codes like \i, \b, \par, etc.
    has_rtf_indicators = (
        text.startswith('{\\rtf') or
        '\\rtf' in text or
        re.search(r'\\[a-zA-Z]+\d*', text) or  # RTF control codes like \i, \i0, \b1
        (text.count('{') > 0 or text.count('}') > 0)  # RTF uses braces for grouping
    )
    
    if not has_rtf_indicators:
        return text
    
    # Step 1: Remove RTF header if present ({\rtf1...)
    text = re.sub(r'\{?\\rtf\d*[^\}]*\}?', '', text, flags=re.IGNORECASE)
    
    # Step 2: Remove RTF control codes
    # Pattern: \ followed by letters and optional digits (e.g., \i, \i0, \i1, \b, \par, \tab)
    # But preserve URLs (http://, https://) and common patterns that might look like RTF
    # We'll be careful to not break URLs
    
    # First, protect URLs by temporarily replacing them
    url_pattern = r'(https?://[^\s]+)'
    urls = re.findall(url_pattern, text)
    url_placeholders = [f'__URL_PLACEHOLDER_{i}__' for i in range(len(urls))]
    for placeholder, url in zip(url_placeholders, urls):
        text = text.replace(url, placeholder)
    
    # Remove RTF control codes: \ followed by letters and optional digits
    # Examples: \i, \i0, \i1, \b, \b0, \par, \tab, \fs24, \f0, etc.
    text = re.sub(r'\\[a-zA-Z]+\d*', '', text)
    
    # Remove RTF escape sequences for special characters (e.g., \'e9 for é)
    # But be careful - we want to keep actual text, so only remove if it's clearly RTF
    # RTF escapes are like \' followed by hex digits
    text = re.sub(r"\\'[0-9a-fA-F]{2}", '', text)
    
    # Remove RTF braces {} - these are used for grouping in RTF
    # But we need to be careful not to remove braces that are part of actual text
    # RTF braces are typically balanced and contain formatting codes
    # We'll remove unmatched braces and braces that contain only formatting
    
    # Remove balanced braces that likely contain only formatting
    # This is a simplified approach - a full RTF parser would be better
    # But we'll remove braces that are clearly RTF formatting markers
    while True:
        # Find braces that contain only whitespace, control codes, or are empty
        old_text = text
        # Remove empty braces
        text = re.sub(r'\{\s*\}', '', text)
        # Remove braces containing only RTF-like content (backslashes, numbers, etc.)
        text = re.sub(r'\{[^}]*\\[^}]*\}', '', text)
        # Remove braces with only numbers/formatting
        text = re.sub(r'\{\s*[\d\s\\]+\s*\}', '', text)
        if text == old_text:
            break
    
    # Remove any remaining unmatched braces (but preserve if they look like actual content)
    # If a brace contains actual letters/words, keep it
    def should_keep_brace(match):
        content = match.group(1)
        # Keep if it has substantial text content (more than just formatting)
        if len(content) > 3 and re.search(r'[a-zA-Z]{3,}', content):
            return match.group(0)  # Keep the whole thing
        return ''  # Remove it
    
    text = re.sub(r'\{([^}]*)\}', should_keep_brace, text)
    
    # Restore URLs
    for placeholder, url in zip(url_placeholders, urls):
        text = text.replace(placeholder, url)
    
    # Step 3: Clean up any remaining RTF artifacts
    # Remove sequences of backslashes
    text = re.sub(r'\\+', '', text)
    
    # Remove sequences that look like RTF formatting remnants
    # Patterns like "iI", "bB", "i0", "i1" at word boundaries (common RTF italic/bold artifacts)
    text = re.sub(r'\b[iI][iI0-9]+\b', '', text)  # Remove "iI", "i0", "i1" etc.
    text = re.sub(r'\b[bB][bB0-9]+\b', '', text)  # Remove "bB", "b0", "b1" etc.
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing punctuation that might be RTF artifacts
    text = text.strip('.,;:!?{}[]()')
    
    return text.strip()


def extract_text_from_attributed_body(attributed_body: Union[bytes, str, None], debug: bool = False) -> str:
    """
    Extract human-readable text from iMessage attributedBody binary blob.
    
    CRITICAL: attributedBody contains NSKeyedArchiver binary plists or RTF data.
    DO NOT decode as UTF-8 text - this creates garbage artifacts like "ii", "bplist", "adhimu".
    
    Args:
        attributed_body: The attributedBody value from the database
        debug: If True, print debug information
        
    Returns:
        Extracted text string, or empty string if extraction fails
    """
    if attributed_body is None:
        return ''
    
    # Convert to bytes if it's a string
    if isinstance(attributed_body, str):
        try:
            attributed_body = attributed_body.encode('latin-1')
        except:
            return ''
    
    if not isinstance(attributed_body, bytes) or len(attributed_body) == 0:
        return ''
    
    # STEP 1: Check if this is a binary plist (starts with "bplist00")
    if attributed_body.startswith(b'bplist'):
        if debug:
            print(f"DEBUG: Binary plist detected (length: {len(attributed_body)})")
        
        # Try plistlib first (standard library)
        if PLISTLIB_AVAILABLE:
            try:
                plist_data = plistlib.loads(attributed_body)
                
                # Recursively extract all string values from the plist
                def extract_user_strings(obj: Any, depth: int = 0) -> List[str]:
                    """Extract only user-visible text, not metadata."""
                    if depth > 15:  # Prevent infinite recursion
                        return []
                    
                    strings = []
                    
                    if isinstance(obj, str):
                        # Skip plist metadata keys
                        if obj.startswith('$') or obj.startswith('NS') or obj.startswith('_'):
                            return []
                        # Skip class names and technical fields
                        if obj in ['root', 'objects', 'archiver', 'version', 'top', 'class']:
                            return []
                        # Skip very short strings (likely metadata)
                        if len(obj) < 2:
                            return []
                        # Keep strings that look like actual message text
                        obj_stripped = obj.strip()
                        if len(obj_stripped) >= 2:
                            strings.append(obj_stripped)
                    
                    elif isinstance(obj, dict):
                        # In NSKeyedArchiver plists, the actual text is often in specific keys
                        # Look for "NSString", "NSAttributedString", or numeric keys
                        for key, value in obj.items():
                            # Skip metadata keys
                            if isinstance(key, str) and (key.startswith('$') or key.startswith('NS') or key.startswith('_')):
                                continue
                            strings.extend(extract_user_strings(value, depth + 1))
                    
                    elif isinstance(obj, (list, tuple)):
                        for item in obj:
                            strings.extend(extract_user_strings(item, depth + 1))
                    
                    return strings
                
                extracted = extract_user_strings(plist_data)
                
                # Find the longest non-empty string (most likely the actual message)
                if extracted:
                    # Filter out obvious metadata
                    user_texts = []
                    for text in extracted:
                        text_lower = text.lower()
                        # Skip if it contains plist/RTF metadata
                        if any(x in text_lower for x in ['bplist', 'nskeyedarchiver', 'nsattributedstring', 'nsmutablestring', 'nsstring', 'rtf1', 'cocoartf']):
                            continue
                        # Skip class names
                        if text.startswith('NS') or text.startswith('_'):
                            continue
                        user_texts.append(text)
                    
                    if user_texts:
                        # Return the longest string (most likely the actual message content)
                        longest_text = max(user_texts, key=len)
                        
                        # If it's RTF, parse it
                        if longest_text.startswith('{\\rtf'):
                            longest_text = parse_rtf_to_plain_text(longest_text)
                        
                        if debug:
                            print(f"DEBUG: Extracted from plist: '{longest_text[:100]}'")
                        
                        return longest_text.strip()
            
            except Exception as e:
                if debug:
                    print(f"DEBUG: plistlib failed: {e}")
        
        # Try biplist as fallback
        if BIPLIST_AVAILABLE:
            try:
                plist_data = biplist.readPlistFromString(attributed_body)
                
                def extract_user_strings(obj: Any, depth: int = 0) -> List[str]:
                    if depth > 15:
                        return []
                    strings = []
                    if isinstance(obj, str):
                        if obj.startswith('$') or obj.startswith('NS') or obj.startswith('_'):
                            return []
                        if len(obj.strip()) >= 2:
                            strings.append(obj.strip())
                    elif isinstance(obj, dict):
                        for key, value in obj.items():
                            if isinstance(key, str) and (key.startswith('$') or key.startswith('NS') or key.startswith('_')):
                                continue
                            strings.extend(extract_user_strings(value, depth + 1))
                    elif isinstance(obj, (list, tuple)):
                        for item in obj:
                            strings.extend(extract_user_strings(item, depth + 1))
                    return strings
                
                extracted = extract_user_strings(plist_data)
                
                if extracted:
                    user_texts = [t for t in extracted if not any(x in t.lower() for x in ['bplist', 'nskeyedarchiver', 'nsattributedstring', 'rtf1']) and not t.startswith('NS')]
                    if user_texts:
                        longest_text = max(user_texts, key=len)
                        if longest_text.startswith('{\\rtf'):
                            longest_text = parse_rtf_to_plain_text(longest_text)
                        if debug:
                            print(f"DEBUG: Extracted from biplist: '{longest_text[:100]}'")
                        return longest_text.strip()
            
            except Exception as e:
                if debug:
                    print(f"DEBUG: biplist failed: {e}")
    
    # STEP 2: Check if this is RTF content (starts with "{\rtf1")
    elif attributed_body.startswith(b'{\\rtf'):
        try:
            # Decode RTF bytes to string first
            rtf_string = attributed_body.decode('utf-8', errors='ignore')
            plain_text = parse_rtf_to_plain_text(rtf_string)
            if debug:
                print(f"DEBUG: RTF converted: '{rtf_string[:100]}' -> '{plain_text[:100]}'")
            return plain_text
        except Exception as e:
            if debug:
                print(f"DEBUG: RTF parsing failed: {e}")
    
    # STEP 3: Handle older NSArchiver format (\x04\x0bstreamtyped...)
    # This format contains length-prefixed strings embedded in binary data
    # Format: [length_byte][text_bytes]...
    try:
        # Scan through the binary data looking for length-prefixed strings
        extracted_strings = []
        i = 0
        while i < len(attributed_body) - 2:
            # Check if this looks like a length prefix
            # Length byte followed by printable ASCII/UTF-8 text
            length = attributed_body[i]
            
            # Reasonable message length (1-255 chars for length-prefixed format)
            if 3 <= length <= 255 and i + length < len(attributed_body):
                # Try to decode the next 'length' bytes as text
                try:
                    candidate = attributed_body[i+1:i+1+length].decode('utf-8', errors='strict')
                    
                    # Check if this looks like actual text (not binary garbage)
                    # Must have mostly printable characters
                    printable_count = sum(1 for c in candidate if c.isprintable() or c in '\n\r\t')
                    if printable_count >= len(candidate) * 0.8:  # 80% printable
                        # Must have at least one letter
                        if any(c.isalpha() for c in candidate):
                            # Skip obvious metadata fields
                            if not any(x in candidate for x in ['kIMMess', 'NSMutable', 'NSAttributed', 'NSString', 'NSObject']):
                                extracted_strings.append(candidate)
                                if debug:
                                    print(f"DEBUG: Found length-prefixed string at offset {i}: '{candidate[:50]}'")
                except (UnicodeDecodeError, ValueError):
                    pass
            
            i += 1
        
        # Return the longest extracted string (most likely the actual message)
        if extracted_strings:
            # Filter to keep only USER TEXT, not metadata
            user_text_candidates = []
            metadata_keywords = ['NSDictionary', 'NSMutable', 'NSAttributed', 'NSString', 'NSObject', 'NSNumber', 'NSValue', 'NSArray', 'NSData', 'streamtyped', 'NSFont', 'NSColor', 'NSRange']
            
            for s in extracted_strings:
                # Skip if it's a metadata class name
                if any(meta in s for meta in metadata_keywords):
                    continue
                # Must be substantial (at least 2 chars) and have letters
                if len(s) >= 2 and any(c.isalpha() for c in s):
                    user_text_candidates.append(s)
            
            if user_text_candidates:
                # Return the LONGEST user text (actual messages are usually longer than artifacts)
                longest_text = max(user_text_candidates, key=len)
                if debug:
                    print(f"DEBUG: Best match from NSArchiver: '{longest_text[:100]}'")
                return longest_text.strip()
            elif extracted_strings:
                # Fallback: return longest of any extracted text
                longest_text = max(extracted_strings, key=len)
                if debug:
                    print(f"DEBUG: Fallback match: '{longest_text[:100]}'")
                return longest_text.strip()
    
    except Exception as e:
        if debug:
            print(f"DEBUG: Length-prefixed extraction failed: {e}")
    
    # STEP 4: Last resort - return empty
    if debug:
        print(f"DEBUG: Could not extract text")
    
    return ''


def parse_rtf_to_plain_text(rtf_content: str) -> str:
    """
    Convert RTF content to plain text.
    This is a simplified RTF parser - strips control codes and formatting.
    """
    if not rtf_content or not isinstance(rtf_content, str):
        return ''
    
    # Remove RTF header
    text = rtf_content
    
    # Remove RTF control words (backslash followed by letters/digits)
    text = re.sub(r'\\[a-zA-Z0-9]+', '', text)
    
    # Remove RTF control symbols (backslash followed by special char)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", '', text)  # Hex escapes
    text = re.sub(r'\\[\\{}]', '', text)  # Escaped backslashes and braces
    
    # Remove all remaining backslashes
    text = re.sub(r'\\', '', text)
    
    # Remove braces
    text = re.sub(r'[{}]', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def get_chat_participants(con: sqlite3.Connection, chat_id: int) -> int:
    """Get the number of unique participants in a chat."""
    try:
        # Try chat_handle_join first
        if 'chat_handle_join' in [t[0] for t in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            count = con.execute(
                "SELECT COUNT(DISTINCT handle_id) FROM chat_handle_join WHERE chat_id = ?",
                (chat_id,)
            ).fetchone()[0]
            return max(2, count)  # At least 2 (user + one other)
        
        # Fallback: count distinct handles in messages for this chat
        count = con.execute(
            """
            SELECT COUNT(DISTINCT m.handle_id) 
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            WHERE cmj.chat_id = ?
            """,
            (chat_id,)
        ).fetchone()[0]
        
        return max(2, count)
    except:
        return 2  # Default to 2 if we can't determine


def load_messages(db_files: List, deidentify: bool = True) -> pd.DataFrame:
    """
    Load messages from SQLite database files.
    
    Args:
        db_files: List of uploaded file objects (chat.db, optionally wal/shm)
        deidentify: Whether to pseudonymize participant identifiers
        
    Returns:
        DataFrame with columns: message_id, chat_id, chat_name, participant, 
        direction, text, timestamp_utc, timestamp_local, group_size
    """
    # Save uploaded files to temp directory
    temp_dir = tempfile.mkdtemp()
    db_path = None
    wal_path = None
    shm_path = None
    
    for uploaded_file in db_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if uploaded_file.name == "chat.db":
            db_path = file_path
        elif uploaded_file.name == "chat.db-wal":
            wal_path = file_path
        elif uploaded_file.name == "chat.db-shm":
            shm_path = file_path
    
    if db_path is None:
        raise ValueError("chat.db file not found in uploaded files")
    
    # Merge WAL if present
    final_db_path = merge_wal_if_needed(db_path, wal_path, shm_path)
    
    # Connect to database
    con = sqlite3.connect(final_db_path)
    
    # Detect available columns
    msg_cols = detect_columns(con, 'message')
    handle_cols = detect_columns(con, 'handle')
    chat_cols = detect_columns(con, 'chat')
    
    # Build query adaptively
    select_parts = [
        "m.ROWID AS message_id",
        "m.text",
        "m.is_from_me",
    ]
    
    if 'handle_id' in msg_cols:
        select_parts.append("m.handle_id")
    else:
        select_parts.append("NULL AS handle_id")
    
    if 'date' in msg_cols:
        select_parts.append("m.date AS apple_time")
    else:
        select_parts.append("NULL AS apple_time")
    
    if 'date_read' in msg_cols:
        select_parts.append("m.date_read")
    else:
        select_parts.append("NULL AS date_read")
    
    if 'date_delivered' in msg_cols:
        select_parts.append("m.date_delivered")
    else:
        select_parts.append("NULL AS date_delivered")
    
    # Add attributedBody if available
    if 'attributedBody' in msg_cols:
        select_parts.append("m.attributedBody")
    else:
        select_parts.append("NULL AS attributedBody")
    
    # Add fields for Tapback and threaded reply detection
    if 'guid' in msg_cols:
        select_parts.append("m.guid")
    else:
        select_parts.append("NULL AS guid")
    
    if 'associated_message_guid' in msg_cols:
        select_parts.append("m.associated_message_guid")
    else:
        select_parts.append("NULL AS associated_message_guid")
    
    if 'associated_message_type' in msg_cols:
        select_parts.append("m.associated_message_type")
    else:
        select_parts.append("NULL AS associated_message_type")
    
    if 'thread_originator_guid' in msg_cols:
        select_parts.append("m.thread_originator_guid")
    else:
        select_parts.append("NULL AS thread_originator_guid")
    
    if 'reply_to_guid' in msg_cols:
        select_parts.append("m.reply_to_guid")
    else:
        select_parts.append("NULL AS reply_to_guid")
    
    # Check for chat_message_join
    has_join = 'chat_message_join' in [t[0] for t in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    
    if has_join:
        select_parts.extend([
            "cmj.chat_id",
            "c.ROWID AS chat_rowid"
        ])
    else:
        select_parts.extend([
            "NULL AS chat_id",
            "c.ROWID AS chat_rowid"
        ])
    
    if 'display_name' in chat_cols:
        select_parts.append("c.display_name")
    else:
        select_parts.append("NULL AS display_name")
    
    if 'id' in handle_cols:
        select_parts.append("h.id AS handle_identifier")
    else:
        select_parts.append("NULL AS handle_identifier")
    
    query = f"""
    SELECT {', '.join(select_parts)}
    FROM message m
    LEFT JOIN handle h ON h.ROWID = m.handle_id
    """
    
    if has_join:
        query += """
        JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        JOIN chat c ON c.ROWID = cmj.chat_id
        """
    else:
        # Fallback: try to join via chat table directly if there's a chat_id column
        if 'chat_id' in msg_cols:
            query += """
            LEFT JOIN chat c ON c.ROWID = m.chat_id
            """
        else:
            query += """
            LEFT JOIN chat c ON 1=0
            """
    
    # No WHERE filter here; we will clean text later in Python
    
    try:
        df = pd.read_sql_query(query, con)
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.error(f"Error loading messages: {e}")
        else:
            print(f"❌ Error loading messages: {e}")
        con.close()
        raise
    
    con.close()
    
    # Normalize text column (ensure it exists, then strip)
    if 'text' in df.columns:
        df['text'] = df['text'].fillna('').astype(str).str.strip()
    else:
        df['text'] = ''
    
    # Extract text from attributedBody when text is empty
    # ONLY use attributedBody when message.text is empty
    # This properly parses NSKeyedArchiver binary plists and RTF data
    if 'attributedBody' in df.columns:
        mask = (df['text'] == '') & df['attributedBody'].notna()
        if mask.any():
            print(f"\n⚙️  Extracting text from attributedBody for {mask.sum():,} messages with empty text...")
            
            # Extract with debug enabled for first 3 samples
            extracted_texts = []
            debug_count = 0
            for idx, (i, row) in enumerate(df[mask].iterrows()):
                debug_mode = debug_count < 3
                extracted = extract_text_from_attributed_body(row['attributedBody'], debug=debug_mode)
                extracted_texts.append(extracted)
                
                if debug_mode and extracted:
                    print(f"  Sample {debug_count + 1}: Raw bytes (first 50): {row['attributedBody'][:50] if isinstance(row['attributedBody'], bytes) else 'N/A'}")
                    print(f"  Sample {debug_count + 1}: Extracted text: '{extracted[:100]}'")
                    debug_count += 1
            
            df.loc[mask, 'text'] = extracted_texts
            
            # Count successful extractions
            non_empty_extracted = sum(1 for t in extracted_texts if t and len(t.strip()) > 0)
            print(f"  ✅ Successfully extracted text from {non_empty_extracted:,} / {mask.sum():,} attributedBody fields")
    
    # Post-process: filter out any remaining Apple metadata artifacts from all text
    # This catches artifacts that might have leaked through or were in the original text field
    def clean_text(text: str) -> str:
        """Remove Apple metadata artifacts from text."""
        if not text or not isinstance(text, str):
            return ''
        
        # List of artifacts to remove (as whole words or substrings)
        artifacts = [
            'nsmutableattributedstring', 'nsmutablestring', 'nsattributedstring', 'nsstring',
            'kimfiletransferguidattributename', 'ddscannerresult', 'wversionyddresult',
            'classrarqtqprsrrvn', 'classnamex', 'znsobjects', 'nskeyedarchiver',
            'xversionyarchiver', 'nsdictionary', 'null', 'objects'
        ]
        
        # Split text into words and filter
        words = text.split()
        cleaned_words = []
        for word in words:
            word_lower = word.lower().strip('.,!?;:"()[]{}')
            # Skip if it's an artifact
            if any(artifact in word_lower for artifact in artifacts):
                continue
            # Skip very short words that match metadata patterns
            if len(word_lower) <= 2 and word_lower in ['ef', 'ar', 'qt', 'pr', 'rr', 'vn', 'dd', 'ns', 'cf', 'ui']:
                continue
            cleaned_words.append(word)
        
        return ' '.join(cleaned_words).strip()
    
    # Apply cleaning to all text
    df['text'] = df['text'].apply(clean_text)
    
    # Convert timestamps
    if 'apple_time' in df.columns and df['apple_time'].notna().any():
        timestamps = df['apple_time'].apply(convert_apple_timestamp)
        df['timestamp_utc'] = [t[0] for t in timestamps]
        df['timestamp_local'] = [t[1] for t in timestamps]
    else:
        df['timestamp_utc'] = None
        df['timestamp_local'] = None
    
    # Determine chat type and group size
    con = sqlite3.connect(final_db_path)
    if 'chat_id' in df.columns and df['chat_id'].notna().any():
        # Use a more efficient approach
        unique_chat_ids = df['chat_id'].dropna().unique()
        chat_size_map = {cid: get_chat_participants(con, int(cid)) for cid in unique_chat_ids}
        df['group_size'] = df['chat_id'].map(chat_size_map).fillna(2)
    else:
        df['group_size'] = 2
    con.close()
    
    # Map direction
    df['direction'] = df['is_from_me'].apply(lambda x: 'out' if x == 1 else 'in')
    
    # De-identify participants
    if deidentify:
        unique_handles = df['handle_identifier'].dropna().unique().tolist()
        participant_map = create_participant_map(unique_handles, deidentify=True)
        participant_map[None] = "UNKNOWN"
        df['participant'] = df['handle_identifier'].map(participant_map).fillna("UNKNOWN")
    else:
        df['participant'] = df['handle_identifier'].fillna("UNKNOWN")
    
    # Get chat names and de-identify if requested
    if 'display_name' in df.columns:
        df['chat_name'] = df['display_name'].fillna("Unknown Chat")
    else:
        df['chat_name'] = "Unknown Chat"
    
    # De-identify chat names (group chats)
    if deidentify:
        # Create mapping for unique chat names
        unique_chat_names = df['chat_name'].dropna().unique().tolist()
        # Filter out "Unknown Chat" from mapping
        unique_chat_names = [c for c in unique_chat_names if c != "Unknown Chat"]
        
        if unique_chat_names:
            chat_name_map = create_participant_map(unique_chat_names, deidentify=True)
            chat_name_map["Unknown Chat"] = "Unknown Chat"  # Keep this as-is
            df['chat_name'] = df['chat_name'].map(chat_name_map).fillna("Unknown Chat")
    
    # Select and order final columns
    final_columns = [
        'message_id', 'chat_id', 'chat_name', 'participant', 
        'direction', 'text', 'timestamp_utc', 'timestamp_local', 'group_size',
        'guid', 'associated_message_guid', 'associated_message_type',
        'thread_originator_guid', 'reply_to_guid'
    ]
    
    # Ensure all columns exist
    for col in final_columns:
        if col not in df.columns:
            df[col] = None
    
    df = df[final_columns].copy()
    
    # Sort by timestamp
    df = df.sort_values('timestamp_local').reset_index(drop=True)
    
    # Store temp directory path for cleanup
    df.attrs['temp_dir'] = temp_dir
    df.attrs['db_path'] = final_db_path
    
    return df

