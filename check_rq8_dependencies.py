#!/usr/bin/env python3
"""
Check if RQ8 dependencies are properly installed.
Run this before using RQ8 to verify your environment.

Usage:
    python check_rq8_dependencies.py
"""

import sys

def check_dependencies():
    """Check all RQ8 dependencies and their versions."""
    print("Checking RQ8 Dependencies...\n")
    print("=" * 60)
    
    # Show which Python is being used
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print("=" * 60 + "\n")
    
    all_ok = True
    
    # Check PyTorch
    try:
        import torch
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])
        
        if (major, minor) >= (2, 0):
            print(f"✅ PyTorch: {version} (OK - version 2.0+)")
        else:
            print(f"❌ PyTorch: {version} (TOO OLD - need 2.0+)")
            print("   Upgrade: pip install --upgrade torch>=2.0.0")
            all_ok = False
    except ImportError:
        print("❌ PyTorch: Not installed")
        print("   Install: pip install torch>=2.0.0")
        all_ok = False
    except Exception as e:
        print(f"⚠️  PyTorch: Error checking version - {e}")
        all_ok = False
    
    # Check sentence-transformers
    try:
        import sentence_transformers
        version = sentence_transformers.__version__
        print(f"✅ sentence-transformers: {version}")
    except ImportError:
        print("❌ sentence-transformers: Not installed")
        print("   Install: pip install sentence-transformers")
        all_ok = False
    except Exception as e:
        print(f"⚠️  sentence-transformers: Error - {e}")
        all_ok = False
    
    # Check UMAP
    try:
        import umap
        version = umap.__version__
        print(f"✅ umap-learn: {version}")
    except ImportError:
        print("❌ umap-learn: Not installed")
        print("   Install: pip install umap-learn")
        all_ok = False
    except Exception as e:
        print(f"⚠️  umap-learn: Error - {e}")
        all_ok = False
    
    # Check HDBSCAN
    try:
        import hdbscan
        version = hdbscan.__version__
        print(f"✅ hdbscan: {version}")
    except ImportError:
        print("❌ hdbscan: Not installed")
        print("   Install: pip install hdbscan")
        all_ok = False
    except Exception as e:
        print(f"⚠️  hdbscan: Error - {e}")
        all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("\n✅ All RQ8 dependencies are properly installed!")
        print("You can now use RQ8 in the iMessage Analyzer.")
        return 0
    else:
        print("\n❌ Some dependencies are missing or outdated.")
        print("\nQuick fix - install all at once:")
        print("pip install --upgrade torch>=2.0.0 sentence-transformers umap-learn hdbscan")
        print("\nAfter installing, restart the Streamlit app.")
        return 1

if __name__ == "__main__":
    sys.exit(check_dependencies())

