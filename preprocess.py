"""Text cleaning and preprocessing functions."""

import re
import string
from typing import List, Optional
import pandas as pd

# Optional emoji library
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    emoji = None

# Lazy imports for optional dependencies
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except (ImportError, TypeError, AttributeError, ModuleNotFoundError):
    # Handle various import errors including pydantic compatibility issues
    SPACY_AVAILABLE = False


# Download NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        STOPWORDS = set(stopwords.words('english'))
        STEMMER = PorterStemmer()
        LEMMATIZER = WordNetLemmatizer()
    except:
        STOPWORDS = set()
        STEMMER = None
        LEMMATIZER = None
else:
    STOPWORDS = set()
    STEMMER = None
    LEMMATIZER = None

# Load spaCy model if available
SPACY_MODEL = None
if SPACY_AVAILABLE:
    try:
        SPACY_MODEL = spacy.load("en_core_web_sm")
    except (OSError, ImportError, TypeError, AttributeError) as e:
        # Handle various spaCy loading errors gracefully
        SPACY_MODEL = None
        import warnings
        warnings.warn(f"spaCy model could not be loaded: {e}. Lemmatization will use NLTK fallback.")

# Metadata/junk tokens to always filter out (from decode artifacts, NSKeyedArchiver, etc.)
METADATA_STOPLIST = {
    # Binary plist and archiver artifacts
    'bplist', 'bplist00', 'bplist01',
    'nskeyedarchiver', 'xversionyarchiver', 'nsdictionary',
    'nsattributedstring', 'nsmutableattributedstring', 'nsstring', 'nsmutablestring',
    # Decode junk tokens
    'rmsv', 'topx', 'emqrstux', 'tdate', 'tu', 'ydayofweek', 'ydaynumber', 'ijk', 'qv',
    # RTF/plist decode artifacts (u-prefixed, short sequences, etc.)
    'uhours', 'uvalue', 'utime', 'ttime', 'abc', 'de', 'ij', 'uv', 'ghi', 'adhimu', 'adhlu',
    # Malformed attributedBody decoding artifacts
    'pthat', 'amttime', 'pmttime', 'ajust', 'rjust', 'luhours',
    # Other common artifacts
    'ddscannerresult', 'wversionyddresult', 'classrarqtqprsrrvn', 'classnamex', 'znsobjects',
    'kimfiletransferguidattributename', 'kimmessagepartattributename', 'kimbasewritingdirectionattributename',
    'nsnumber', 'nsvalue', 'nsrange', 'nslocation', 'nslength', 'nsdata',
    'nsarray', 'nsmutablearray', 'nsset', 'nsmutableset',
    'nsfont', 'nsparagraphstyle', 'nscolor', 'nsunderline', 'nsstrikethrough',
    'nssuperscript', 'nslink', 'nsattachment',
    # Plist keys
    '$objects', '$archiver', '$version', '$top', '$class',
    # RTF artifacts
    'rtf1', 'cocoasubrtf',
    # Short metadata fragments
    'null', 'objects', 'ef', 'ar', 'qt', 'pr', 'rr', 'vn', 'dd', 'il'
}

# Regex patterns that indicate metadata tokens (not real words)
# Note: We're careful not to filter legitimate short words like "us", "go", "ok", "hi"
METADATA_PATTERNS = [
    r'^[a-z]+[0-9]+$',  # Lowercase followed by numbers (e.g., "bplist00", "tdate", "ydayofweek")
    r'^[0-9]+[a-z]+$',  # Numbers followed by lowercase
    r'^[a-z]+[A-Z][a-z]+$',  # CamelCase (Apple class names)
    r'^[A-Z][a-z]+[A-Z]',  # PascalCase
    r'^\$[a-z]+$',  # Starts with $ (plist keys)
    r'^[a-z]{12,}$',  # Very long lowercase words (often class names, but allow shorter ones)
    r'^[a-z]{2,4}[a-z]{2,4}[a-z]{2,4}[a-z]+$',  # Consecutive short segments (like "emqrstux")
    # RTF/plist decode artifacts
    r'^u[a-z]+$',  # Starts with 'u' followed by lowercase (uhours, uvalue, utime)
    r'^[a-z]{2,4}$',  # 2-4 letter sequences (will be filtered further by vowel check)
    r'^[a-z]{5,7}$',  # 5-7 letter sequences without English roots (will be checked)
    r'^t[a-z]{4,}$',  # Starts with 't' followed by 4+ letters (ttime, etc.)
]

# Common legitimate short words to preserve (even if they match some patterns)
LEGITIMATE_SHORT_WORDS = {
    'us', 'go', 'ok', 'hi', 'pm', 'am', 'no', 'so', 'to', 'do', 'be', 'me', 'we', 'he', 'it',
    'is', 'at', 'on', 'in', 'up', 'if', 'or', 'as', 'an', 'my', 'by', 'of', 'oh', 'ah', 'ha',
    'yeah', 'yes', 'one', 'got', 'get', 'let', 'see', 'like', 'love', 'loved', 'liked',
    'want', 'next', 'free', 'time', 'going', 'know', 'would', 'good', 'thanks', 'thank',
    'saturday', 'november', 'code', 'please', 'could', 'think', 'need', 'stop', 'https',
    'alan', 'today',  # Common names and words that might match patterns
}

# Common English word roots/substrings to help identify real words
# Used to filter out tokens that don't contain recognizable English patterns
ENGLISH_WORD_ROOTS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our',
    'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see',
    'two', 'way', 'who', 'boy', 'did', 'has', 'let', 'put', 'say', 'she', 'too', 'use', 'man',
    'any', 'ask', 'big', 'end', 'few', 'got', 'had', 'has', 'her', 'him', 'his', 'its', 'lot',
    'may', 'men', 'new', 'now', 'old', 'own', 'put', 'say', 'she', 'the', 'too', 'try', 'two',
    'use', 'way', 'who', 'why', 'yes', 'yet', 'you', 'able', 'back', 'been', 'best', 'call',
    'came', 'come', 'does', 'done', 'down', 'each', 'even', 'ever', 'find', 'first', 'from',
    'gave', 'give', 'goes', 'gone', 'good', 'great', 'hand', 'have', 'here', 'high', 'home',
    'hour', 'into', 'just', 'keep', 'kind', 'knew', 'know', 'last', 'late', 'left', 'life',
    'like', 'line', 'live', 'long', 'look', 'made', 'make', 'many', 'mean', 'meet', 'mind',
    'more', 'most', 'move', 'much', 'must', 'name', 'near', 'need', 'next', 'nice', 'once',
    'only', 'open', 'over', 'part', 'pass', 'past', 'pick', 'play', 'pull', 'push', 'put',
    'read', 'real', 'right', 'room', 'said', 'same', 'saw', 'say', 'see', 'seem', 'send',
    'sent', 'show', 'side', 'some', 'soon', 'sort', 'such', 'sure', 'take', 'talk', 'tell',
    'than', 'that', 'them', 'then', 'they', 'thin', 'this', 'those', 'though', 'thought',
    'three', 'through', 'time', 'told', 'took', 'turn', 'under', 'until', 'upon', 'used',
    'very', 'wait', 'walk', 'want', 'was', 'way', 'well', 'went', 'were', 'what', 'when',
    'where', 'which', 'while', 'white', 'who', 'whole', 'whom', 'whose', 'why', 'wide',
    'wife', 'will', 'wind', 'wish', 'with', 'within', 'without', 'woman', 'women', 'won',
    'wonder', 'wood', 'word', 'work', 'world', 'would', 'write', 'wrong', 'wrote', 'year',
    'years', 'yes', 'yet', 'you', 'young', 'your', 'yours', 'yourself', 'youth'
}


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub('', text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    email_pattern = re.compile(r'\S+@\S+')
    return email_pattern.sub('', text)


def remove_mentions(text: str) -> str:
    """Remove @mentions from text."""
    mention_pattern = re.compile(r'@\w+')
    return mention_pattern.sub('', text)


def remove_numbers(text: str) -> str:
    """Remove numbers from text."""
    return re.sub(r'\d+', '', text)


def remove_emojis(text: str) -> str:
    """Remove emojis from text."""
    if not EMOJI_AVAILABLE:
        # Fallback: use regex to remove emoji ranges (basic coverage)
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    return emoji.replace_emoji(text, replace='')


def extract_emojis(text: str) -> List[str]:
    """Extract emoji characters from text."""
    if not EMOJI_AVAILABLE:
        # Fallback: return empty list if emoji library not available
        return []
    return [c for c in text if c in emoji.EMOJI_DATA]


def tokenize(text: str) -> List[str]:
    """Basic tokenization - split on whitespace and punctuation."""
    # Remove punctuation and split
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [t.lower() for t in tokens if t.strip()]


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    remove_emojis_flag: bool = True,
    remove_urls_flag: bool = True,
    remove_emails_flag: bool = True,
    remove_mentions_flag: bool = True,
    remove_numbers_flag: bool = False,
    stem: bool = False,
    lemmatize: bool = False,
    min_length: int = 2,
    custom_stopwords: Optional[set] = None,
    disabled_metadata_stopwords: Optional[set] = None,
    remove_metadata_tokens: bool = True
) -> List[str]:
    """
    Preprocess a single text string.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_stopwords: Remove English stopwords
        remove_emojis_flag: Remove emoji characters
        remove_urls_flag: Remove URLs
        remove_emails_flag: Remove email addresses
        remove_mentions_flag: Remove @mentions
        remove_numbers_flag: Remove numbers
        stem: Apply stemming
        lemmatize: Apply lemmatization (takes precedence over stemming)
        min_length: Minimum token length
        
    Returns:
        List of preprocessed tokens
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    processed = text
    
    # Remove URLs, emails, mentions
    if remove_urls_flag:
        processed = remove_urls(processed)
    if remove_emails_flag:
        processed = remove_emails(processed)
    if remove_mentions_flag:
        processed = remove_mentions(processed)
    if remove_numbers_flag:
        processed = remove_numbers(processed)
    if remove_emojis_flag:
        processed = remove_emojis(processed)
    
    # Convert to lowercase
    if lowercase:
        processed = processed.lower()
    
    # Tokenize
    tokens = tokenize(processed)
    
    # Remove stopwords
    if remove_stopwords:
        # Combine NLTK stopwords with any custom stopwords passed in
        all_stopwords = STOPWORDS.copy() if STOPWORDS else set()
        if custom_stopwords:
            all_stopwords.update(custom_stopwords)
        tokens = [t for t in tokens if t not in all_stopwords]
    
    # Apply lemmatization or stemming
    if lemmatize:
        if SPACY_MODEL:
            # Use spaCy for lemmatization
            doc = SPACY_MODEL(" ".join(tokens))
            tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        elif LEMMATIZER:
            tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    elif stem and STEMMER:
        tokens = [STEMMER.stem(t) for t in tokens]
    
    # Filter by minimum length
    tokens = [t for t in tokens if len(t) >= min_length]
    
    # Filter out metadata/junk tokens
    # This removes decode artifacts, NSKeyedArchiver metadata, and other junk
    if disabled_metadata_stopwords is None:
        disabled_metadata_stopwords = set()
    
    def is_metadata_token(token: str) -> bool:
        """Check if a token is metadata/junk that should be filtered."""
        token_lower = token.lower()
        
        # Always preserve legitimate short words
        if token_lower in LEGITIMATE_SHORT_WORDS:
            return False
        
        # Check explicit stoplist (but skip if disabled)
        if token_lower in METADATA_STOPLIST and token_lower not in disabled_metadata_stopwords:
            return True
        
        # Filter tokens starting with 'u' followed by lowercase (uhours, uvalue, utime) - RTF/plist artifacts
        if re.match(r'^u[a-z]{3,}$', token_lower):
            return True
        
        # Filter 2-4 letter sequences with no vowels or only 1 vowel (e.g., abc, de, ij, uv, ghi)
        if 2 <= len(token_lower) <= 4:
            vowel_count = len(re.findall(r'[aeiou]', token_lower))
            if vowel_count <= 1:
                # Exception: keep legitimate words
                if token_lower not in LEGITIMATE_SHORT_WORDS:
                    return True
        
        # Filter 3-6 letter lowercase tokens that don't contain English word roots
        if 3 <= len(token_lower) <= 6 and token_lower.isalpha():
            # Check if it contains any English word root as a substring
            has_english_root = False
            for root in ENGLISH_WORD_ROOTS:
                if len(root) >= 3 and root in token_lower:
                    has_english_root = True
                    break
            
            if not has_english_root:
                # Check vowel pattern - if it has < 2 vowels, likely artifact
                vowel_count = len(re.findall(r'[aeiou]', token_lower))
                if vowel_count < 2:
                    return True
                # Check for improbable consonant clusters (adhimu, adhlu, ghi)
                if re.search(r'[bcdfghjklmnpqrstvwxyz]{3,}', token_lower):
                    return True
                # Check for specific artifact patterns
                if re.search(r'adh|ghi|him|hlu|himu', token_lower):
                    return True
        
        # Filter 5-7 letter sequences without English roots
        if 5 <= len(token_lower) <= 7 and token_lower.isalpha():
            has_english_root = False
            for root in ENGLISH_WORD_ROOTS:
                if len(root) >= 4 and root in token_lower:
                    has_english_root = True
                    break
            if not has_english_root:
                # Check for RTF/plist patterns
                if re.match(r'^[a-z]{5,7}$', token_lower):
                    vowel_count = len(re.findall(r'[aeiou]', token_lower))
                    if vowel_count < 2:
                        return True
        
        # Filter tokens starting with 't' followed by 4+ letters (ttime, etc.)
        # But allow legitimate words like 'think', 'thank', 'today', 'time'
        if re.match(r'^t[a-z]{4,}$', token_lower):
            if token_lower not in LEGITIMATE_SHORT_WORDS:
                # Check if it has reasonable vowel structure
                if not re.search(r'[aeiou]{2,}', token_lower):
                    return True
        
        # Check against regex patterns (but be lenient for longer words with English roots)
        for pattern in METADATA_PATTERNS:
            if re.match(pattern, token):
                # For 5-7 letter words, check if they contain English roots
                if 5 <= len(token_lower) <= 7:
                    if any(len(root) >= 4 and root in token_lower for root in ENGLISH_WORD_ROOTS):
                        continue  # Might be a real word, don't filter
                return True
        
        # Filter very short tokens (1-2 chars) that don't have vowels and aren't in legitimate list
        if len(token) <= 2 and not re.search(r'[aeiouAEIOU]', token):
            return True
        
        return False
    
    # Filter out metadata tokens (if enabled)
    if remove_metadata_tokens:
        tokens = [t for t in tokens if not is_metadata_token(t)]
    
    return tokens


def preprocess_documents(
    texts: pd.Series,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    remove_emojis_flag: bool = True,
    remove_urls_flag: bool = True,
    remove_emails_flag: bool = True,
    remove_mentions_flag: bool = True,
    remove_numbers_flag: bool = False,
    stem: bool = False,
    lemmatize: bool = False,
    min_length: int = 2,
    custom_stopwords: Optional[set] = None,
    disabled_metadata_stopwords: Optional[set] = None,
    remove_metadata_tokens: bool = True
) -> List[List[str]]:
    """
    Preprocess a series of text documents.
    
    Args:
        texts: Series of text strings
        ... (same as preprocess_text)
        custom_stopwords: Optional set of custom stopwords to add to the default list
        disabled_metadata_stopwords: Optional set of metadata stopwords to disable (not filter)
        
    Returns:
        List of lists of tokens
    """
    return [
        preprocess_text(
            text, lowercase, remove_stopwords, remove_emojis_flag,
            remove_urls_flag, remove_emails_flag, remove_mentions_flag,
            remove_numbers_flag, stem, lemmatize, min_length, custom_stopwords,
            disabled_metadata_stopwords, remove_metadata_tokens
        )
        for text in texts
    ]

