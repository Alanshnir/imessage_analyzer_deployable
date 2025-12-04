# Building a Standalone App with PyInstaller

## 1. Prepare Environment

**Install PyInstaller:**
```bash
pip install pyinstaller
```

**Download NLTK data (required):**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

This ensures NLTK data is available in your Python environment before building.

## 2. Build the executable

**On macOS:**

```bash
pyinstaller --onefile --noconsole run_analyzer.py
```

**On Windows:**

```bash
pyinstaller --onefile run_analyzer.py
```

## 3. Output

Your standalone app will appear in:

- `dist/run_analyzer` (macOS)
- `dist/run_analyzer.exe` (Windows)

Double-click to launch. It will open the Streamlit interface in the browser at:

**http://localhost:8501**

## Notes

- Users do not need Python installed.
- All processing happens locally.
- Users should place their `chat.db` files anywhere on their machine; the app's upload UI will access them normally.

## Troubleshooting

### Issue: App doesn't start

- **Solution**: Check that all required packages are installed before building:
  ```bash
  pip install -r requirements.txt
  ```

### Issue: "Module not found" errors in built app

- **Solution**: Add hidden imports to PyInstaller:
  ```bash
  pyinstaller --onefile --noconsole \
    --hidden-import=nltk \
    --hidden-import=gensim \
    --hidden-import=sklearn \
    --hidden-import=plotly \
    run_analyzer.py
  ```

### Issue: NLTK data not found

- **Solution**: Download NLTK data before building:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```

### Issue: Large file size

- The standalone app will be ~300-500MB due to included dependencies
- This is normal for Python apps with scientific libraries
- Consider distributing as a .zip file

## Distribution

After building:
1. Test the executable on your machine
2. Zip the `dist/run_analyzer` (or `dist/run_analyzer.exe`)
3. Upload to GitHub Releases
4. Users download, unzip, and double-click to run

