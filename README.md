# iMessage Analyzer

A privacy-first, local-only application for analyzing your iMessage conversations. This tool helps you understand your messaging patterns through nine research questions covering topic avoidance, conversation dynamics, and behavioral patterns.

## Research Questions

1. **RQ1**: Topics you tend not to engage with (broad LDA topics)
2. **RQ2**: Most commonly discussed topics
3. **RQ3**: Group vs one-to-one responsiveness
4. **RQ4**: Conversation starter topics (fine-grained)
5. **RQ5**: Conversation ender topics (fine-grained)
6. **RQ6**: Topics by closeness (close contacts vs acquaintances)
7. **RQ7**: Topics by time of day
8. **RQ8**: Fine-grained avoided topics (TF-IDF + K-Means)
9. **RQ9**: GPT-powered chatbot for querying your behavior

## Features

- 🔒 **Privacy-first**: All processing happens locally on your device. No data is sent over the network.
- 📊 **Comprehensive Analysis**: Topic modeling, response time analysis, and engagement metrics.
- 🎨 **Interactive Visualizations**: Beautiful charts and graphs using Plotly.
- 📁 **Export Reports**: Generate self-contained HTML reports and CSV exports.
- 🔐 **De-identification**: Automatically pseudonymizes participants by default.
- 🤖 **AI Chatbot**: Ask questions about your texting patterns using GPT (optional, requires API key)

## Local Application (No Python Needed)

### Download Standalone App

Download the standalone app from the [Releases](https://github.com/Alanshnir/imessage_analyzer_deployable/releases) section.

### Run the App

Double-click the file:
- `run_analyzer` on macOS
- `run_analyzer.exe` on Windows

Your browser will open automatically to:
**http://localhost:8501**

### Upload and Analyze

1. Accept the consent prompt
2. Upload your `chat.db` file (found at `~/Library/Messages/chat.db`)
   - Press **Cmd+Shift+G** in Finder and type `~/Library/Messages` to locate it
3. Configure analysis settings in the sidebar
4. Run any of the 9 research questions
5. Explore your messaging patterns!

**No Python installation required** - the app is fully self-contained.

## Requirements (For Developers)

- Python 3.10 or higher
- macOS (for accessing iMessage database)
- SQLite database files: `chat.db` (and optionally `chat.db-wal`, `chat.db-shm`)

## Installation

1. Clone or download this repository:
```bash
cd imessage_analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

   **Important**: If you encounter numpy/pandas compatibility errors, try:
   ```bash
   pip install --upgrade --force-reinstall numpy pandas
   pip install -r requirements.txt
   ```

3. Download NLTK data (if not already downloaded):
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

4. (Optional) Download spaCy English model for better lemmatization:
```bash
python -m spacy download en_core_web_sm
```
   
   **Note**: If you encounter compatibility errors with spaCy (especially with Python 3.12+), the app will automatically fall back to NLTK for lemmatization. You can skip this step if you encounter issues.

5. (Optional) Install MALLET for improved topic modeling:
   - Download MALLET from http://mallet.cs.umass.edu/
   - Set the `MALLET_HOME` environment variable to point to the MALLET directory
   - Example: `export MALLET_HOME=/path/to/mallet`

## Finding Your iMessage Database

On macOS, your iMessage database is located at:
```
~/Library/Messages/chat.db
```

You may also find:
- `chat.db-wal` (Write-Ahead Log)
- `chat.db-shm` (Shared Memory file)

**Note**: You may need to temporarily disable iMessage or close the Messages app to copy these files, as they may be locked while in use.

## Usage

### Main Analyzer App

1. Start the Streamlit analyzer app:
```bash
streamlit run app.py
```

2. The app will open in your web browser (usually at http://localhost:8501).

### Message Viewer App

To simply browse and view messages from your chat.db file:

1. Start the viewer app:
```bash
streamlit run viewer.py
```

2. Upload your chat.db file and browse messages with:
   - Search functionality to find specific messages
   - Filters by direction (sent/received) and date range
   - Pagination to browse through messages
   - Option to show raw participant identifiers
   - Export filtered messages to CSV

**Note**: Both apps are configured to accept files up to 2GB. If your file is larger, you may need to increase the limit in `.streamlit/config.toml`

## Building a Standalone Executable

Want to distribute the app as a single executable? See [build_instructions.md](build_instructions.md) for details.

**Quick build:**
```bash
pip install pyinstaller
pyinstaller --onefile --noconsole run_analyzer.py
```

The executable will be in `dist/run_analyzer` (or `dist/run_analyzer.exe` on Windows).

## Project Structure

```
imessage_analyzer/
├── app.py                    # Streamlit main analyzer application
├── run_analyzer.py           # Launcher script for standalone builds
├── viewer.py                 # Simple message viewer app
├── data_loader.py            # Database loading, WAL merging, SQL queries
├── preprocess.py             # Text cleaning and preprocessing
├── topics.py                 # Topic modeling with Gensim/MALLET
├── responses.py              # Response time calculations
├── analytics.py              # RQ1-8 aggregation logic
├── viz.py                    # Plotting and visualization
├── chatbot.py                # RQ9 GPT-powered chatbot
├── simple_topics.py          # RQ8 TF-IDF topic modeling
├── embedding_topics.py       # Optional BERT-based topics
├── utils.py                  # Utility functions (hashing, config)
├── requirements.txt          # Python dependencies
├── build_instructions.md     # PyInstaller build guide
└── README.md                 # This file
```

## How It Works

### Data Loading
- Handles both `chat.db` alone and with WAL/SHM files
- Automatically merges WAL files into a temporary database
- Defensive SQL queries that adapt to available schema columns
- Converts Apple timestamps (seconds or nanoseconds since 2001-01-01)

### Topic Modeling
- Uses Latent Dirichlet Allocation (LDA) via Gensim
- Optional MALLET integration for improved results
- Configurable preprocessing (stopwords, stemming, lemmatization)
- Computes topic coherence scores

### Response Time Analysis
- Tracks time from received message to your next sent message
- Filters outliers and configurable maximum gap windows
- Computes reluctance scores for engagement analysis

### Research Questions

**RQ1: Topics You Tend Not to Engage With**
- Identifies topics in received messages using LDA
- Weights topics by response time/reluctance
- Ranks topics by engagement reluctance
- Supports 1-30 topics (user-configurable)

**RQ2: Most Commonly Discussed Topics**
- Analyzes topic distribution across all conversations
- Shows topic prevalence per contact
- Tracks topic trends over time
- Supports 1-30 topics (user-configurable)

**RQ3: Group vs One-to-One Responsiveness**
- Compares reply rates and response times
- Analyzes by group size categories
- Identifies contacts you respond to more in groups
- Filterable by minimum message count

**RQ4: Conversation Starter Topics**
- Uses fine-grained 50-topic model
- Identifies topics that start conversations
- Measures reply likelihood and response speed
- Tracks session positioning

**RQ5: Conversation Ender Topics**
- Uses fine-grained 50-topic model
- Identifies topics that end conversations
- Measures no-reply rates and long response times
- Tracks conversation termination patterns

**RQ6: Topics by Closeness**
- Compares close contacts vs acquaintances
- Uses odds ratios to identify distinctive topics
- User-configurable closeness thresholds
- Fine-grained 50-topic analysis

**RQ7: Topics by Time of Day**
- Analyzes topics by time period (morning/afternoon/evening/night)
- Heatmap visualization
- Fine-grained 50-topic analysis
- Shows when different topics are discussed

**RQ8: Fine-Grained Avoided Topics**
- Uses TF-IDF + K-Means clustering
- 15-50 topics (user-configurable)
- More granular than RQ1
- No heavy dependencies required

**RQ9: GPT-Powered Chatbot**
- Ask questions about your texting patterns in plain English
- Uses aggregated statistics only (privacy-first)
- Requires OpenAI API key (user-provided)
- No raw messages sent to API

## Privacy & Security

- **Local Processing**: All analysis happens on your device
- **De-identification**: Participant identifiers are pseudonymized by default
- **No Network Calls**: The app works completely offline
- **Temporary Files**: WAL-merged databases are stored temporarily and can be deleted via the UI

## Troubleshooting

**"No module named 'nltk'" or similar errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Download NLTK data as described in Installation

**"Error loading messages" or SQL errors:**
- Ensure the database file is not locked (close Messages app)
- Check that you're using the correct `chat.db` file
- The app will warn you if certain columns are missing

**MALLET not found:**
- This is optional. The app works fine with Gensim LDA only
- If you want MALLET, set the `MALLET_HOME` environment variable

**numpy/pandas compatibility errors:**
- If you see "numpy.dtype size changed" errors, this is a binary incompatibility issue
- Fix by reinstalling numpy and pandas: `pip install --upgrade --force-reinstall numpy pandas`
- Then reinstall other dependencies: `pip install -r requirements.txt`
- This usually happens when packages were compiled against different numpy versions

**spaCy compatibility errors:**
- If you see pydantic/ForwardRef errors when installing spaCy, this is a known compatibility issue
- The app will automatically use NLTK for lemmatization instead
- You can skip the spaCy installation - it's optional and the app works without it
- If you need spaCy, try: `pip install "spacy<3.8.0" "pydantic<2.0.0"` then download the model

**Slow performance:**
- Large databases (100k+ messages) may take several minutes to process
- Consider filtering by date range if available in future versions
- Use fewer topics or passes for faster topic modeling

## Limitations

- Requires macOS to access the iMessage database
- Large databases may take time to process
- Topic modeling quality depends on message volume and diversity
- Some features may be limited if database schema differs from expected

## License

This project is provided as-is for personal use. Please respect privacy and use responsibly.

## Contributing

This is a personal project, but suggestions and improvements are welcome!

