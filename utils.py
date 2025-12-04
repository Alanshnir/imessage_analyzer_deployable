"""Utility functions for hashing, configuration, and helpers."""

import hashlib
import base64
import os
from typing import Dict, Optional


def hash_identifier(identifier: str) -> str:
    """
    Create a stable hash-based pseudonym for an identifier.
    
    Args:
        identifier: Phone number or email address
        
    Returns:
        Short pseudonym like 'P001' or 'P002'
    """
    # Use SHA-256 for stable hashing
    hash_obj = hashlib.sha256(identifier.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    # Use first 6 characters of hash for pseudonym
    # Map to a readable format
    hash_int = int(hash_hex[:8], 16)
    pseudonym_id = hash_int % 10000  # Keep it reasonable
    
    return f"P{pseudonym_id:04d}"


def create_participant_map(identifiers: list, deidentify: bool = True) -> Dict[str, str]:
    """
    Create a mapping from identifiers to pseudonyms.
    
    Args:
        identifiers: List of unique participant identifiers
        deidentify: If True, use pseudonyms; if False, use original identifiers
        
    Returns:
        Dictionary mapping original identifier to display name
    """
    if not deidentify:
        return {ident: ident for ident in identifiers}
    
    mapping = {}
    seen_hashes = {}
    counter = 1
    
    for ident in sorted(set(identifiers)):
        if ident is None:
            mapping[None] = "UNKNOWN"
            continue
            
        # Check if we've seen this hash before
        hash_key = hashlib.sha256(str(ident).encode('utf-8')).hexdigest()[:8]
        if hash_key in seen_hashes:
            mapping[ident] = seen_hashes[hash_key]
        else:
            pseudonym = f"P{counter:04d}"
            mapping[ident] = pseudonym
            seen_hashes[hash_key] = pseudonym
            counter += 1
    
    return mapping


def get_mallet_home() -> Optional[str]:
    """Check if MALLET is available at MALLET_HOME environment variable."""
    mallet_home = os.environ.get('MALLET_HOME')
    if mallet_home and os.path.exists(mallet_home):
        mallet_path = os.path.join(mallet_home, 'bin', 'mallet')
        if os.path.exists(mallet_path) or os.path.exists(f"{mallet_path}.bat"):
            return mallet_home
    return None


def ensure_exports_dir():
    """Ensure the exports directory exists."""
    exports_dir = "exports"
    if not os.path.exists(exports_dir):
        os.makedirs(exports_dir)
    return exports_dir


def format_duration(minutes: float) -> str:
    """Format duration in minutes to human-readable string."""
    if pd.isna(minutes) or minutes < 0:
        return "N/A"
    
    if minutes < 60:
        return f"{int(minutes)}m"
    elif minutes < 1440:
        hours = minutes / 60
        return f"{hours:.1f}h"
    else:
        days = minutes / 1440
        return f"{days:.1f}d"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


# Import pandas for type checking
import pandas as pd

