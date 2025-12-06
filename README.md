# iMessage Analyzer

A privacy-first, local-only macOS application for analyzing your iMessage conversations. This tool helps you understand your messaging patterns through six research questions covering topic avoidance, conversation dynamics, sentiment analysis, and behavioral patterns.

**macOS only** - iMessage database access requires macOS.

## Research Questions

1. **RQ1**: Topics you tend not to engage with (LDA topic modeling)
2. **RQ2**: Most commonly discussed topics (topic distribution analysis)
3. **RQ3**: Group vs one-to-one responsiveness (reply rate comparison)
4. **RQ4**: Conversation starter topics (topics that initiate conversations)
5. **RQ5**: Sentiment analysis (positive/negative/neutral message analysis)
6. **RQ6**: GPT-powered chatbot for querying your behavior (AI-powered insights)

## Features

- 🔒 **Privacy-first**: All processing happens locally on your device. No data is sent over the network.
- 📊 **Comprehensive Analysis**: Topic modeling, response time analysis, and engagement metrics.
- 🎨 **Interactive Visualizations**: Beautiful charts and graphs using Plotly.
- 📁 **Export Data**: Export unprocessed text messages as CSV for custom analysis.
- 🔐 **De-identification**: Automatically pseudonymizes participants by default.
- 🤖 **AI Chatbot**: Ask questions about your texting patterns using GPT (optional, requires API key)

## Quick Start

### Installation

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/Alanshnir/imessage_analyzer_deployable.git
   cd imessage_analyzer
   ```

2. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. Your browser will open automatically to **http://localhost:8501**

**Requirements:**
- macOS 10.14 or later
- Python 3.10 or higher
- Internet connection (for first-time dependency installation)

### Upload and Analyze

1. **⚠️ IMPORTANT:** Close the Messages app before uploading (Cmd+Q on Mac)
2. Accept the consent prompt
3. **💡 Tip:** Adjust settings in the left-hand panel menu before loading your file
4. Upload your `chat.db` file (found at `~/Library/Messages/chat.db`)
   - Press **Cmd+Shift+G** in Finder and type `~/Library/Messages` to locate it
5. Configure analysis settings in the sidebar
6. Run any of the research questions
7. Explore your messaging patterns!

**Requirements:**
- macOS 10.14 or later
- ~500MB free disk space
- **No Python installation required** - the app is fully self-contained


## Requirements (For Developers)

- Python 3.10 or higher
- macOS (required - iMessage database is macOS-only)
- SQLite database file: `chat.db`

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

**⚠️ IMPORTANT**: You **must close the Messages app** before uploading your chat.db file! The database file is locked while Messages is running and cannot be read. Quit Messages from the menu bar (Messages → Quit Messages) or press Cmd+Q.

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

**Note**: The app is configured to accept files up to 5GB. If your file is larger, you may need to increase the limit in `.streamlit/config.toml`

### Key Features

- **Unified Topic Model**: RQ1, RQ2, and RQ4 all use the same topics learned from ALL your messages (sent + received), ensuring consistency across analyses
- **Per-Contact Analysis**: Explore individual contact behavior, topic distributions, and sentiment patterns
- **Exploratory Data Analysis**: Response times, text lengths, message counts, and daily trends
- **Customizable Settings**: Adjust topic count, preprocessing options, response time thresholds, and more


See [DEPLOYMENT.md](DEPLOYMENT.md) for complete build instructions and GitHub release setup.

## Project Structure

```
imessage_analyzer/
├── app.py                    # Streamlit main analyzer application
├── run_analyzer.py           # Launcher script for standalone builds
├── run_analyzer.spec         # PyInstaller configuration
├── viewer.py                 # Simple message viewer app (optional)
├── data_loader.py            # Database loading, SQL queries, attributedBody extraction
├── preprocess.py             # Text cleaning and preprocessing
├── topics.py                 # Topic modeling with Gensim/MALLET
├── responses.py              # Response time calculations, reluctance scores
├── analytics.py              # RQ1-6 aggregation logic
├── viz.py                    # Plotting and visualization (Plotly/Matplotlib)
├── chatbot.py                # RQ6 GPT-powered chatbot
├── sentiment.py              # RQ5 VADER sentiment analysis
├── utils.py                  # Utility functions (hashing, config)
├── requirements.txt          # Python dependencies
├── DEPLOYMENT.md             # Deployment and build guide
├── QUICK_START_DEPLOYMENT.md # Quick deployment reference
├── deploy.sh                 # Automated deployment script
├── .streamlit/config.toml    # Streamlit configuration (5GB upload limit)
└── README.md                 # This file
```

## How It Works

### Data Loading
- Loads `chat.db` file directly
- Defensive SQL queries that adapt to available schema columns
- Converts Apple timestamps (seconds or nanoseconds since 2001-01-01)

### Topic Modeling
- Uses Latent Dirichlet Allocation (LDA) via Gensim
- Optional MALLET integration for improved results
- **Unified Model**: RQ1, RQ2, and RQ4 all use the same topic model trained on ALL messages (sent + received)
- Configurable preprocessing (automatic: lowercase, emoji removal; optional: stopwords, stemming, lemmatization)
- User-configurable number of topics (1-30, default: 30)
- Computes topic coherence scores

### Response Time Analysis
- Tracks time from received message to your next sent message
- Supports Tapbacks and threaded replies (user-configurable)
- Filters outliers and configurable maximum gap windows
- Computes reluctance scores (0-1 scale) for engagement analysis
- Optimized for large datasets (vectorized operations, caching)

### Research Questions

**RQ1: Topics You Tend Not to Engage With**
- Uses LDA topic modeling on ALL aggregated messages (sent + received)
- Identifies topics in received messages with high reluctance scores
- Weights topics by response time and reply likelihood
- Ranks topics by engagement reluctance
- Supports 1-30 topics (user-configurable, default: 30)
- Includes per-contact analysis and high reluctance message examples

**RQ2: Most Commonly Discussed Topics**
- Uses the SAME topics from RQ1 (learned from all aggregated messages)
- Analyzes topic distribution across all conversations
- Shows topic prevalence per contact
- Tracks topic trends over time
- Per-contact topic analysis (pie charts and time series)
- Supports 1-30 topics (user-configurable, default: 30)

**RQ3: Group vs One-to-One Responsiveness**
- Compares reply rates and response times between group and one-on-one chats
- Analyzes by group size categories
- Identifies contacts you respond to more in groups vs individually
- Filterable by minimum message count
- Per-contact group vs one-on-one analysis

**RQ4: Conversation Starter Topics**
- Uses the SAME topics from RQ1 (learned from all aggregated messages)
- Identifies topics that start conversations
- Measures reply likelihood, response speed, and session positioning
- Ranks topics by starter score (combination of reply rate, speed, and position)

**RQ5: Sentiment Analysis**
- Uses VADER sentiment analysis on raw, untokenized text
- Analyzes positive/negative/neutral sentiment across all messages
- Shows sentiment trends over time
- Per-chat sentiment explorer
- Top contacts by positive sentiment proportion
- Separate analysis for sent vs received messages

**RQ6: GPT-Powered Chatbot**
- Ask questions about your texting patterns in plain English
- Uses aggregated statistics from RQ1-5 only (privacy-first)
- Requires OpenAI API key (user-provided, stored in session state)
- No raw messages, phone numbers, or PII sent to API
- Viewable/editable system context for transparency

## Privacy & Security

- **Local Processing**: All analysis happens on your device
- **De-identification**: Participant identifiers and chat names are pseudonymized by default (user-configurable)
- **No Network Calls**: The app works completely offline (except optional RQ6 chatbot which requires API key)
- **Privacy-First Chatbot**: RQ6 chatbot only sends aggregated statistics, never raw messages or PII

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

- **macOS only** - iMessage database is only available on macOS
- Large databases may take time to process
- Topic modeling quality depends on message volume and diversity
- Some features may be limited if database schema differs from expected

## License

This project is provided as-is for personal use. Please respect privacy and use responsibly.

## Contributing

This is a personal project, but suggestions and improvements are welcome!

