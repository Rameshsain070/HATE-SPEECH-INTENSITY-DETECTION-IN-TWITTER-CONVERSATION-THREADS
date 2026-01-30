import os
import glob
import re
import pickle
import numpy as np
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# PARAMETERS & SETTINGS
# -----------------------------
DATA_FOLDER = 'data_anti_racism'
LEXICON_PATH = 'expandedlexicon.txt'
FIXED_LENGTH = 300         # total timesteps per conversation
History_len = 25           # history length (th)
ROLLING_WINDOW = 10        # smoothing window
Future_len = FIXED_LENGTH - History_len   # future length (tf)
# Preprocessed file names
TS_FILE = "preprocessed_time_series.pkl"
SENT_FILE = "preprocessed_sentiments.pkl"
GRAPH_FILE = "preprocessed_graphs.pkl"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def preprocess_text(text):
    text = re.sub(r'@\w+', 'mention', text)
    text = re.sub(r'http\S+|www\.\S+', 'URL', text)
    return text

def load_lexicon(filepath):
    lex = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                score = float(parts[1])
                lex[word] = score
    return lex

def compute_lexicon_score(text, lex):
    words = re.findall(r'\w+', text.lower())
    scores = [lex[w] for w in words if w in lex]
    return np.mean(scores) if scores else 0.0

def compute_hate_intensity(text, hate_model, lex, w=0.6):
    res = hate_model(text)[0]
    hate_prob = res['score']
    lex_score = compute_lexicon_score(text, lex)
    return w * hate_prob + (1 - w) * lex_score

def rolling_average_win(values, window=ROLLING_WINDOW):
    out = []
    for i in range(len(values) - window + 1):
        out.append(np.mean(values[i:i+window]))
    return out

def process_conversation_csv(csv_file, hate_model, lex):
    df = pd.read_csv(csv_file, usecols=['username','userid','time','id','text'])
    df['text'] = df['text'].apply(preprocess_text)
    intensities = [compute_hate_intensity(txt, hate_model, lex) for txt in df['text']]
    if len(intensities) >= ROLLING_WINDOW:
        intensities = rolling_average_win(intensities, window=ROLLING_WINDOW)
    return np.array(intensities)

def load_all_conversations(folder, hate_model, lex):
    series_list = []
    files = glob.glob(os.path.join(folder, '*.csv'))
    for f in files:
        try:
            s = process_conversation_csv(f, hate_model, lex)
            if len(s) < FIXED_LENGTH:
                s = np.pad(s, (0, FIXED_LENGTH - len(s)), constant_values=0)
            else:
                s = s[:FIXED_LENGTH]
            series_list.append(s)
        except Exception as e:
            print(f"Error processing {f}: {e}")
    return np.array(series_list)

def build_graph(csv_file, fixed_len=FIXED_LENGTH):
    df = pd.read_csv(csv_file, usecols=['username','userid','time','id','text'])
    n_actual = len(df)
    n = min(n_actual, fixed_len)
    A = np.zeros((fixed_len, fixed_len))
    if n > 0:
        for i in range(n):
            A[i, i] = 1
        for i in range(1, n):
            A[0, i] = 1
            A[i, 0] = 1
    D = np.diag(np.sum(A, axis=1))
    D[D == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    return D_inv_sqrt @ A @ D_inv_sqrt

def load_all_graphs(folder, fixed_len=FIXED_LENGTH):
    graphs = []
    files = glob.glob(os.path.join(folder, '*.csv'))
    for f in files:
        try:
            g = build_graph(f, fixed_len)
            graphs.append(g)
        except Exception as e:
            print(f"Error processing graph for {f}: {e}")
    return np.array(graphs)

def compute_sentiment_series(csv_file, sent_model):
    df = pd.read_csv(csv_file, usecols=['username','userid','time','id','text'])
    df['text'] = df['text'].apply(preprocess_text)
    texts = df['text'].tolist()
    sentiments = []
    if len(texts) > 0:
        # Use the first tweet as the root
        root_emb = sent_model.encode(texts[0], convert_to_tensor=True)
        for txt in texts[:History_len]:
            emb = sent_model.encode(txt, convert_to_tensor=True)
            sim = float(util.cos_sim(root_emb, emb)[0][0])
            sentiments.append(sim)
    # Pad or truncate to exactly History_len elements
    if len(sentiments) < History_len:
        sentiments += [0.5] * (History_len - len(sentiments))
    else:
        sentiments = sentiments[:History_len]
    # Apply rolling average smoothing if possible
    if len(sentiments) >= ROLLING_WINDOW:
        sentiments = rolling_average_win(sentiments, window=ROLLING_WINDOW)
    return np.array(sentiments)

def load_all_sentiments(folder, sent_model):
    sent_list = []
    files = glob.glob(os.path.join(folder, '*.csv'))
    for f in files:
        try:
            s = compute_sentiment_series(f, sent_model)
            sent_list.append(s)
        except Exception as e:
            print(f"Error processing sentiments for {f}: {e}")
    return np.array(sent_list)

# -----------------------------
# MAIN DATA PREPROCESSING & SAVING/LOADING
# -----------------------------
def save_preprocessed_data(time_series, sentiments, graph_data):
    with open(TS_FILE, 'wb') as f:
        pickle.dump(time_series, f)
    with open(SENT_FILE, 'wb') as f:
        pickle.dump(sentiments, f)
    with open(GRAPH_FILE, 'wb') as f:
        pickle.dump(graph_data, f)
    print("Preprocessed data saved.")

def load_preprocessed_data():
    with open(TS_FILE, 'rb') as f:
        time_series = pickle.load(f)
    with open(SENT_FILE, 'rb') as f:
        sentiments = pickle.load(f)
    with open(GRAPH_FILE, 'rb') as f:
        graph_data = pickle.load(f)
    print("Preprocessed data loaded.")
    return time_series, sentiments, graph_data

# Load pretrained models and lexicon
print("Loading pretrained models and lexicon...")
hate_model = pipeline("text-classification", model="unitary/toxic-bert", tokenizer="unitary/toxic-bert")
sent_model = SentenceTransformer('all-MiniLM-L6-v2')
lexicon = load_lexicon(LEXICON_PATH)

# Check if preprocessed data exists; if not, process and save it.
if os.path.exists(TS_FILE) and os.path.exists(SENT_FILE) and os.path.exists(GRAPH_FILE):
    final_time_series, senti_time_series, graph_data = load_preprocessed_data()
else:
    print("Processing raw data to create time-series, sentiment, and graph data...")
    final_time_series = load_all_conversations(DATA_FOLDER, hate_model, lexicon)
    senti_time_series = load_all_sentiments(DATA_FOLDER, sent_model)
    graph_data = load_all_graphs(DATA_FOLDER, FIXED_LENGTH)
    print(f"Final time-series shape: {final_time_series.shape}")
    print(f"Sentiment time-series shape: {senti_time_series.shape}")
    print(f"Graph data shape: {graph_data.shape}")
    save_preprocessed_data(final_time_series, senti_time_series, graph_data)

# (At this point, the preprocessed data is ready and can be fed directly into the subsequent training pipeline)
print("Data preprocessing complete. You can now load these data into your training components.")
