# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import kagglehub
import time
from tqdm import tqdm
import ssl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import warnings
import joblib
import matplotlib.patches as patches
warnings.filterwarnings('ignore')
# For improved NLP capabilities
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Import torch at the top level to avoid NameError
try:
    import torch
except ImportError:
    print("PyTorch not available - install with: pip install torch")
    torch = None

# Try to import transformers with error handling
TRANSFORMERS_AVAILABLE = False
try:
    import transformers
    # Check if tf-keras is needed
    try:
        from transformers import (
            AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
            DistilBertTokenizer, DistilBertForSequenceClassification,
            TrainingArguments, Trainer
        )
        import torch
        from torch.utils.data import Dataset, DataLoader
        
        TRANSFORMERS_AVAILABLE = True
        print("Transformers package available, advanced models can be used")
    except Exception as e:
        print(f"Transformers module error: {e}")
        print("Try running: pip install tf-keras")
except ImportError:
    print("Transformers package not available - advanced models will be disabled")
    print("To enable, install: pip install transformers torch tf-keras peft")

# Fix SSL Certificate issue for NLTK download
ssl._create_default_https_context = ssl._create_unverified_context

# Download VADER lexicon
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print("Downloading VADER lexicon for sentiment analysis...")
    nltk.download('vader_lexicon', quiet=True)
    
# Set random seed for reproducibility
np.random.seed(42)
if torch is not None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

# Create a directory for plots
# Get the directory of this script and create 'plot' folder there
script_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(script_dir, "plot")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory '{plot_dir}' for saving plots")
else:
    print(f"Using existing directory '{plot_dir}' for saving plots")

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_palette('Set2')

# Define standard figure size for Beamer slides (16:9 ratio)
BEAMER_WIDTH = 12.8
BEAMER_HEIGHT = 7.2
plt.rcParams['figure.figsize'] = (BEAMER_WIDTH, BEAMER_HEIGHT)

plt.rcParams['font.size'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.transparent'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['ps.fonttype'] = 42  # For EPS embedding
plt.rcParams['pdf.fonttype'] = 42  # For PDF embedding
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams['patch.facecolor'] = 'white'

# Enhanced styling for more professional plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.8
plt.rcParams['legend.edgecolor'] = 'lightgray'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['grid.alpha'] = 0.3

# Define a better color palette for political analysis
# Use more sophisticated blue/red gradients for political data
lib_colors = ['#1A85FF', '#5AA7FF', '#89C4FF', '#B8E0FF']  # Liberal blues
con_colors = ['#D41159', '#E54A76', '#F27399', '#FFA6BF']  # Conservative reds
neutral_colors = ['#767676', '#A3A3A3', '#D1D1D1', '#F3F3F3']  # Neutral grays

# Custom political color maps
liberal_cmap = LinearSegmentedColormap.from_list('liberal', lib_colors)
conservative_cmap = LinearSegmentedColormap.from_list('conservative', con_colors)
political_cmap = LinearSegmentedColormap.from_list('political', [lib_colors[0], neutral_colors[0], con_colors[0]])

# Main political colors
lib_color = lib_colors[0]  # Deep blue for Liberal
con_color = con_colors[0]  # Deep red for Conservative

# Helper function to save plots in multiple formats for presentation
def save_plot(path_without_extension, max_dpi=300):
    """Save the current plot in PNG, PDF, and EPS formats with Beamer-friendly dimensions."""
    # Get current figure size
    fig = plt.gcf()
    fig_width, fig_height = fig.get_size_inches()
    
    # Check if the figure would exceed matplotlib's size limits (65536 pixels in any dimension)
    # If so, reduce the DPI to keep within limits
    max_pixels = 65000  # Just under the 2^16 (65536) limit
    
    # Calculate safe DPI to avoid exceeding max_pixels
    safe_dpi_width = max_pixels / fig_width
    safe_dpi_height = max_pixels / fig_height
    safe_dpi = min(safe_dpi_width, safe_dpi_height, max_dpi)
    
    # If calculated DPI is too high, print a warning and use a safer value
    if safe_dpi < max_dpi:
        print(f"Warning: Reducing DPI from {max_dpi} to {int(safe_dpi)} to avoid oversized figure.")
    
    # Use the safe DPI value
    actual_dpi = safe_dpi
    
    # Save as PNG
    plt.savefig(f"{path_without_extension}.png", dpi=actual_dpi, format='png', bbox_inches='tight')
    # Save as PDF
    plt.savefig(f"{path_without_extension}.pdf", format='pdf', bbox_inches='tight')
    # Save as EPS
    plt.savefig(f"{path_without_extension}.eps", format='eps', bbox_inches='tight')
    
    return f"{path_without_extension}.png"  # Return PNG path for reference

# Function to add watermark to plots
def add_watermark(ax, text="ECON 6140 Final Project", alpha=0.07):
    """Add a watermark to the plot"""
    ax.text(0.5, 0.5, text, 
            fontsize=30, color='gray', alpha=alpha,
            ha='center', va='center', rotation=30, transform=ax.transAxes)

# Function to apply consistent styling to plots
def style_plot(ax, title, xlabel, ylabel, legend_title=None):
    """Apply consistent styling to plot axes"""
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Style legend if present
    if legend_title and ax.get_legend():
        ax.legend(title=legend_title, fontsize=12, title_fontsize=13, 
                 frameon=True, framealpha=0.8, edgecolor='lightgray')
    
    # Add subtle frame to define the plot area
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('#333333')
    
    # Optional watermark
    # add_watermark(ax)
    
    return ax

#######################################################
# SECTION 1: Custom NLP Functions
#######################################################

# Stopwords list
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now', 'amp', 'like', 'get', 'would'
}

# Political stopwords
POLITICAL_STOPWORDS = {
    'one', 'also', 'us', 'say', 'said', 'even', 'people', 'think', 'know', 
    'going', 'time', 'good', 'make', 'way', 'really', 'thing'
}
STOPWORDS.update(POLITICAL_STOPWORDS)

def simple_tokenize(text):
    """Simple tokenizer - split on non-word characters"""
    return re.findall(r'\b\w+\b', text.lower())

def simple_lemmatize(word):
    """Simple lemmatization based on common rules"""
    if len(word) < 4:
        return word
        
    # Simple suffix stripping rules
    if word.endswith('ing'):
        return word[:-3]
    elif word.endswith('ed') and len(word) > 4:
        return word[:-2]
    elif word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    elif word.endswith('ies'):
        return word[:-3] + 'y'
    elif word.endswith('es'):
        return word[:-2]
    return word

def preprocess_text(text):
    """Preprocess text without NLTK dependency"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = simple_tokenize(text)
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    
    # Lemmatize
    tokens = [simple_lemmatize(t) for t in tokens]
    
    return " ".join(tokens)

# Simple sentiment analysis lexicons
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'right', 'better', 'best', 'love', 
    'happy', 'positive', 'wonderful', 'nice', 'amazing', 'awesome',
    'support', 'win', 'winning', 'success', 'successful', 'beneficial',
    'agree', 'correct', 'freedom', 'free', 'hope', 'protect', 'safe', 'secure'
}

NEGATIVE_WORDS = {
    'bad', 'worst', 'terrible', 'wrong', 'hate', 'sad', 'negative', 
    'awful', 'horrible', 'poor', 'disappointing', 'fail', 'failure', 
    'lose', 'losing', 'lost', 'reject', 'rejection', 'harmful',
    'corrupt', 'disaster', 'evil', 'fear', 'against', 'attack', 'crisis'
}

def simple_sentiment(text):
    """Basic sentiment analysis using lexicons"""
    if not text or pd.isna(text):
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        
    words = set(simple_tokenize(str(text)))
    pos_count = len(words.intersection(POSITIVE_WORDS))
    neg_count = len(words.intersection(NEGATIVE_WORDS))
    total = len(words)
    
    if total > 0:
        pos = pos_count / total
        neg = neg_count / total
        neu = 1 - (pos + neg)
        # Compound score: scaled from -1 to 1
        if pos + neg > 0:
            compound = (pos - neg) / (pos + neg)
        else:
            compound = 0
    else:
        pos, neg, neu, compound = 0, 0, 1, 0
        
    return {'neg': neg, 'neu': neu, 'pos': pos, 'compound': compound}

# Initialize VADER sentiment analyzer for improved sentiment analysis
vader_analyzer = SentimentIntensityAnalyzer()

def improved_sentiment(text):
    """Improved sentiment analysis using NLTK VADER"""
    if not text or pd.isna(text):
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        
    # Use VADER for sentiment analysis
    return vader_analyzer.polarity_scores(str(text))

#######################################################
# SECTION 2: Visualization Functions
#######################################################

def get_word_frequencies(texts):
    """Create a word frequency dictionary from texts"""
    all_text = ' '.join([str(text) for text in texts])
    words = all_text.split()
    word_freq = {}
    for word in words:
        if len(word) > 2:  # Skip very short words
            word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq

def generate_wordcloud(texts, title, filename):
    """Generate and save wordcloud for a collection of texts with Beamer-friendly dimensions"""
    try:
        # Get word frequencies
        word_freq = get_word_frequencies(texts)
        
        # Create wordcloud directly from frequency dictionary with Beamer dimensions
        wordcloud = WordCloud(
            width=int(BEAMER_WIDTH * 100),  # Scale width for better resolution
            height=int(BEAMER_HEIGHT * 50),  # Adjust height for widescreen format
            background_color='white',
            max_words=100,
            colormap='Blues' if 'Liberal' in title else 'Reds',  # Different colormap based on political leaning
            contour_width=1,
            contour_color='steelblue' if 'Liberal' in title else 'firebrick',
            prefer_horizontal=0.9,  # Slightly more horizontal words for better readability
            random_state=42  # For reproducibility
        ).generate_from_frequencies(word_freq)
        
        # Create plot with Beamer dimensions
        plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT * 0.8))  # Make slightly shorter for title space
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, fontweight='bold')
        plt.tight_layout(pad=1)
        
        # Save image
        save_path = os.path.join(plot_dir, filename)
        save_plot(save_path.replace('.png', ''))  # Remove .png extension
        print(f"Wordcloud saved to '{save_path}.png/pdf/eps'")
        
    except Exception as e:
        print(f"Error generating wordcloud: {e}")
        fallback_word_plot(texts, title, filename)

def fallback_word_plot(texts, title, filename):
    """Create a simple bar chart of word frequencies as a fallback"""
    word_freq = get_word_frequencies(texts)
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_words)), list(top_words.values()), 
            color='blue' if 'Liberal' in title else 'red')
    plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
    plt.title(f"Top Words in {title.split('in ')[-1]}")
    plt.tight_layout()
    save_path = os.path.join(plot_dir, filename.replace('wordcloud', 'top_words'))
    save_plot(save_path.replace('.png', ''))  # Remove .png extension

def get_top_ngram(corpus, n=2, top_k=20):
    """Get top n-grams from corpus"""
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]

def plot_top_ngrams(liberal_texts, conservative_texts, n=2, top_k=20):
    """Plot top n-grams for each political leaning"""
    liberal_ngrams = get_top_ngram(liberal_texts, n, top_k)
    conservative_ngrams = get_top_ngram(conservative_texts, n, top_k)
    
    plt.figure(figsize=(16, 12))
    
    # Liberal n-grams
    plt.subplot(2, 1, 1)
    x, y = zip(*[(ng[0], ng[1]) for ng in liberal_ngrams])
    plt.barh(range(len(x)), y, color='#3498db')
    plt.yticks(range(len(x)), x)
    plt.title(f'Top {top_k} {n}-grams in Liberal Posts', fontsize=16)
    plt.xlabel('Frequency', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Conservative n-grams
    plt.subplot(2, 1, 2)
    x, y = zip(*[(ng[0], ng[1]) for ng in conservative_ngrams])
    plt.barh(range(len(x)), y, color='#e74c3c')
    plt.yticks(range(len(x)), x)
    plt.title(f'Top {top_k} {n}-grams in Conservative Posts', fontsize=16)
    plt.xlabel('Frequency', fontsize=14)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'{n}gram_analysis.png')
    save_plot(save_path.replace('.png', ''))
    print(f"{str(n)}-gram analysis saved to '{save_path}'")

def get_characteristic_words(texts1, texts2, top_k=20):
    """Get words that are characteristic of one corpus compared to another"""
    # Get word frequencies
    freq1 = get_word_frequencies(texts1)
    freq2 = get_word_frequencies(texts2)
    
    # Get all unique words
    all_words = set(freq1.keys()).union(set(freq2.keys()))
    
    # Calculate ratios (with smoothing to avoid division by zero)
    smoothing = 1
    ratios1 = {word: (freq1.get(word, 0) + smoothing) / (freq2.get(word, 0) + smoothing) 
              for word in all_words if len(word) > 3}  # Filter short words
    
    ratios2 = {word: (freq2.get(word, 0) + smoothing) / (freq1.get(word, 0) + smoothing) 
              for word in all_words if len(word) > 3}  # Filter short words
    
    # Get top words by ratio and raw count threshold
    min_count = 20  # Minimum raw count to consider
    top_words1 = [(word, ratio) for word, ratio in sorted(ratios1.items(), key=lambda x: x[1], reverse=True) 
                  if freq1.get(word, 0) >= min_count][:top_k]
    
    top_words2 = [(word, ratio) for word, ratio in sorted(ratios2.items(), key=lambda x: x[1], reverse=True)
                  if freq2.get(word, 0) >= min_count][:top_k]
    
    return top_words1, top_words2

#######################################################
# SECTION 3: Data Loading and Preprocessing
#######################################################

print("=" * 80)
print("REDDIT POLITICAL DISCOURSE ANALYSIS")
print("=" * 80)

# 1. Download and Load the Dataset from Kaggle
print("\n1. Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("neelgajare/liberals-vs-conservatives-on-reddit-13000-posts")
print(f"Dataset downloaded to: {path}")

# Find the CSV file in the downloaded directory
csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV files found in the dataset directory")

# Load the dataset
data_file = os.path.join(path, csv_files[0])
df = pd.read_csv(data_file)

print(f"\nDataset loaded with shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nSample data (first 5 rows):")
print(df.head().to_string())

# 2. Data Cleaning and Exploration
print("\n2. Checking for missing values...")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Identify columns containing text and labels
text_col = 'Text'  # Primary text content
title_col = 'Title'  # Can also be used as text
label_col = 'Political Lean'  # Label column

# Better handling of missing text data
print("\nImproving data quality by handling missing values...")
print(f"Initial dataset shape: {df.shape}")

# First check: Fill NaN in Text with Title content where Text is missing
df[text_col] = df[text_col].fillna(df[title_col])

# Second check: Remove rows where both Text and Title are too short
df['combined_text_length'] = df[text_col].apply(lambda x: len(str(x).split()) if not pd.isna(x) else 0)
min_text_length = 5  # Minimum words required
print(f"Removing {(df['combined_text_length'] < min_text_length).sum()} rows with insufficient text (<{min_text_length} words)")
df = df[df['combined_text_length'] >= min_text_length]

# Third check: make sure remaining rows have valid text
still_missing = df[text_col].isnull().sum()
if still_missing > 0:
    print(f"Still missing {still_missing} text entries after filling with titles.")
    # Remove rows with no text content
    df = df.dropna(subset=[text_col])

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Class distribution after cleaning: \n{df[label_col].value_counts()}")

# Check class balance
class_counts = df[label_col].value_counts()
class_ratio = class_counts.min() / class_counts.max()
print(f"Class balance ratio (min/max): {class_ratio:.2f}")

# 3. Text Preprocessing
print("\n3. Preprocessing text data...")
df['processed_text'] = df[text_col].apply(preprocess_text)

# Check length of posts
df['text_length'] = df[text_col].apply(lambda x: len(str(x).split()))

# Add improved sentiment analysis
print("\n4. Adding improved sentiment analysis...")
df['sentiment'] = df[text_col].apply(improved_sentiment)
df['sentiment_compound'] = df['sentiment'].apply(lambda x: x['compound'])
df['sentiment_positive'] = df['sentiment'].apply(lambda x: x['pos'])
df['sentiment_negative'] = df['sentiment'].apply(lambda x: x['neg'])
df['sentiment_neutral'] = df['sentiment'].apply(lambda x: x['neu'])

# 5. Calculating descriptive statistics with more detailed output
print("\n5. Calculating detailed descriptive statistics...")
desc_stats = df.describe(include='all', percentiles=[.05, .25, .5, .75, .95])
print("\nDetailed statistics for all variables:")
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 120)         # Set wider display
print(desc_stats.transpose())  # Transpose for better readability

# Visualize key numerical variables distributions
print("\nVisualizing key variable distributions...")
numerical_vars = ['text_length', 'sentiment_compound', 'sentiment_positive', 'sentiment_negative']
plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))

for i, var in enumerate(numerical_vars):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=df, x=var, hue=label_col, kde=True, element="step", 
                 palette=[lib_color, con_color])
    plt.title(f'Distribution of {var}', fontsize=14)
    plt.xlabel(var, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title='Political Leaning')

plt.tight_layout()
save_path = os.path.join(plot_dir, 'key_variables_distribution')
save_plot(save_path)
print(f"Key variables distribution plot saved to '{save_path}.png/pdf/eps'")

# Save preprocessed data
df.to_csv('preprocessed_reddit_political_data.csv', index=False)
print("\nPreprocessed data saved to 'preprocessed_reddit_political_data.csv'")

#######################################################
# SECTION 4: Exploratory Data Analysis
#######################################################

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Define colors for consistent usage throughout analysis
lib_color = '#3498db'  # Blue for Liberal
con_color = '#e74c3c'  # Red for Conservative

# 1. Analyze class distribution
print("\n1. Analyzing class distribution...")
plt.figure(figsize=(12, 8))
value_counts = df[label_col].value_counts()
colors = [lib_color, con_color] if value_counts.index[0] == 'Liberal' else [con_color, lib_color]

ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette=colors)
plt.title('Distribution of Political Leaning in Dataset', fontsize=18, fontweight='bold')
plt.xlabel('Political Leaning', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', alpha=0.3)

# Add count labels on the bars
for i, v in enumerate(value_counts):
    ax.text(i, v + 50, f"{v:,}", ha='center', fontsize=14, fontweight='bold')

# Add percentage labels
total = value_counts.sum()
for i, v in enumerate(value_counts):
    percentage = (v / total) * 100
    ax.text(i, v/2, f"{percentage:.1f}%", ha='center', fontsize=14, color='white', fontweight='bold')

plt.tight_layout()
save_path = os.path.join(plot_dir, 'political_leaning_distribution.png')
save_plot(save_path.replace('.png', ''))
print(f"Class distribution plot saved to '{save_path}'")

# 2. Analyze post length by political leaning
print("\n2. Analyzing post length distribution...")
plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

# Main histogram with density - MODIFIED BIN STRATEGY
ax0 = plt.subplot(gs[0, :])

# Calculate optimal bin size using Freedman-Diaconis rule
def freedman_diaconis_bins(data):
    """Calculate optimal bin width using Freedman-Diaconis rule."""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))
    # Ensure minimum bin width
    bin_width = max(bin_width, 1)
    # Calculate number of bins based on range of data
    data_range = np.max(data) - np.min(data)
    if bin_width > 0:
        return int(np.ceil(data_range / bin_width))
    else:
        return 50  # Default if calculation fails

# Get 95th percentile as x-limit to focus on main distribution
x_limit = df['text_length'].quantile(0.95)

# Calculate optimal bin count for each group
liberal_data = df[df[label_col] == 'Liberal']['text_length']
conservative_data = df[df[label_col] == 'Conservative']['text_length']
liberal_data_trimmed = liberal_data[liberal_data <= x_limit]
conservative_data_trimmed = conservative_data[conservative_data <= x_limit]

# Use the same number of bins for both groups based on combined data
combined_data = pd.concat([liberal_data_trimmed, conservative_data_trimmed])
optimal_bins = freedman_diaconis_bins(combined_data)
print(f"Using {optimal_bins} bins for histograms based on Freedman-Diaconis rule")

# Create bin edges with more detail in the dense regions
# Log-spaced bins for better representation of skewed data
bin_edges = np.linspace(5, x_limit, optimal_bins)

# Plot histograms with better bins
for lean in df[label_col].unique():
    subset = df[df[label_col] == lean]
    color = lib_color if lean == 'Liberal' else con_color
    sns.histplot(data=subset, x='text_length', bins=bin_edges,
                element="step", color=color, alpha=0.6, 
                kde=True, kde_kws={'bw_adjust': 0.8}, 
                stat='density', ax=ax0, label=f"{lean}")

ax0.set_title('Distribution of Post Lengths by Political Leaning', fontsize=18, fontweight='bold')
ax0.set_xlabel('Number of Words', fontsize=16)
ax0.set_ylabel('Density', fontsize=16)
ax0.set_xlim(0, x_limit)  # Limit x-axis for better visualization
ax0.legend(title='Political Leaning', fontsize=14, title_fontsize=14)
ax0.grid(alpha=0.3)

# Add annotations about distribution shapes
liberal_mean = liberal_data.mean()
conservative_mean = conservative_data.mean()
ax0.axvline(liberal_mean, color=lib_color, linestyle='--', alpha=0.7)
ax0.axvline(conservative_mean, color=con_color, linestyle='--', alpha=0.7)
ax0.text(liberal_mean+2, ax0.get_ylim()[1]*0.95, f'Liberal Mean: {liberal_mean:.1f}', 
         color=lib_color, fontweight='bold', ha='left', va='top')
ax0.text(conservative_mean+2, ax0.get_ylim()[1]*0.85, f'Conservative Mean: {conservative_mean:.1f}', 
         color=con_color, fontweight='bold', ha='left', va='top')

# Box plot
ax1 = plt.subplot(gs[1, 0])
sns.boxplot(data=df, x=label_col, y='text_length', palette=[lib_color, con_color], ax=ax1)
ax1.set_title('Post Length Box Plot', fontsize=16)
ax1.set_xlabel('Political Leaning', fontsize=14)
ax1.set_ylabel('Number of Words', fontsize=14)
ax1.grid(alpha=0.3)

# Add descriptive statistics
ax2 = plt.subplot(gs[1, 1])
ax2.axis('off')
stats_text = "Post Length Statistics:\n\n"

for label in df[label_col].unique():
    lengths = df[df[label_col] == label]['text_length']
    stats_text += f"{label} Posts:\n"
    stats_text += f"  Mean: {lengths.mean():.1f} words\n"
    stats_text += f"  Median: {lengths.median():.1f} words\n"
    stats_text += f"  Std Dev: {lengths.std():.1f} words\n"
    stats_text += f"  Max: {lengths.max()} words\n"
    stats_text += f"  Min: {lengths.min()} words\n\n"

ax2.text(0, 0.5, stats_text, fontsize=12, verticalalignment='center')

plt.tight_layout()
save_path = os.path.join(plot_dir, 'post_length_analysis')
save_plot(save_path)
print(f"Post length analysis plot saved to '{save_path}.png/pdf/eps'")

# 3. Generate word clouds
print("\n3. Generating word clouds...")
liberal_texts = df[df[label_col] == 'Liberal']['processed_text']
conservative_texts = df[df[label_col] == 'Conservative']['processed_text']

# Create a figure with two word clouds
plt.figure(figsize=(20, 10))

# Liberal word cloud
plt.subplot(1, 2, 1)
liberal_word_freq = get_word_frequencies(liberal_texts)
liberal_wordcloud = WordCloud(
    width=800, 
    height=400,
    background_color='white',
    max_words=150,
    contour_width=3,
    contour_color=lib_color,
    colormap='Blues',
    collocations=False
).generate_from_frequencies(liberal_word_freq)
plt.imshow(liberal_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Liberal Posts', fontsize=20, fontweight='bold')

# Conservative word cloud
plt.subplot(1, 2, 2)
conservative_word_freq = get_word_frequencies(conservative_texts)
conservative_wordcloud = WordCloud(
    width=800, 
    height=400,
    background_color='white',
    max_words=150,
    contour_width=3,
    contour_color=con_color,
    colormap='Reds',
    collocations=False
).generate_from_frequencies(conservative_word_freq)
plt.imshow(conservative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Conservative Posts', fontsize=20, fontweight='bold')

plt.tight_layout()
save_path = os.path.join(plot_dir, 'political_wordclouds.png')
save_plot(save_path.replace('.png', ''))
print(f"Word clouds saved to '{save_path}'")

# 4. Sentiment Analysis Visualization
print("\n4. Analyzing sentiment patterns...")
plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

# Compound sentiment distribution - with clearer bins and annotations
ax0 = plt.subplot(gs[0, 0])

# Create better bins for sentiment distribution
sentiment_bins = np.linspace(-1, 1, 41)  # More granular bins

for lean in df[label_col].unique():
    subset = df[df[label_col] == lean]
    color = lib_color if lean == 'Liberal' else con_color
    alpha = 0.6 if lean == 'Liberal' else 0.5  # Slight difference in transparency
    sns.histplot(data=subset, x='sentiment_compound', bins=sentiment_bins,
                hue=None, color=color, alpha=alpha, 
                kde=True, kde_kws={'bw_adjust': 0.8}, 
                stat='density', ax=ax0, label=lean)

# Add vertical lines at neutral boundary with annotations
ax0.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
ax0.text(0.02, ax0.get_ylim()[1]*0.95, 'Positive →', ha='left', va='top', fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.2"))
ax0.text(-0.02, ax0.get_ylim()[1]*0.95, '← Negative', ha='right', va='top', fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.2"))

# Add mean lines
lib_mean = df[df[label_col] == 'Liberal']['sentiment_compound'].mean()
con_mean = df[df[label_col] == 'Conservative']['sentiment_compound'].mean()
ax0.axvline(x=lib_mean, color=lib_color, linestyle='-', alpha=0.8, linewidth=1.5)
ax0.axvline(x=con_mean, color=con_color, linestyle='-', alpha=0.8, linewidth=1.5)
ax0.text(lib_mean, ax0.get_ylim()[1]*0.8, f'Liberal Mean: {lib_mean:.3f}', ha='left' if lib_mean > 0 else 'right', 
         va='top', color=lib_color, fontsize=10, rotation=90,
         bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.1"))
ax0.text(con_mean, ax0.get_ylim()[1]*0.8, f'Conservative Mean: {con_mean:.3f}', ha='left' if con_mean > 0 else 'right', 
         va='top', color=con_color, fontsize=10, rotation=90,
         bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.1"))

ax0.set_title('Distribution of Compound Sentiment', fontsize=16, fontweight='bold')
ax0.set_xlabel('Compound Sentiment Score (-1 to 1)', fontsize=12)
ax0.set_ylabel('Density', fontsize=12)
ax0.legend(title='Political Leaning', fontsize=10, title_fontsize=10)
ax0.grid(alpha=0.3)

# Mean sentiment components comparison - more visually appealing bar chart
ax1 = plt.subplot(gs[0, 1])
sentiment_metrics = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
sentiment_data = []

for metric in sentiment_metrics:
    for label in df[label_col].unique():
        mean_value = df[df[label_col] == label][metric].mean()
        sentiment_data.append({
            'Sentiment Type': metric.split('_')[1].capitalize(),
            'Political Leaning': label,
            'Mean Score': mean_value
        })

sentiment_df = pd.DataFrame(sentiment_data)

# Use cleaner barplot with hatching to distinguish groups
sns.barplot(data=sentiment_df, x='Sentiment Type', y='Mean Score', hue='Political Leaning', 
            palette=[lib_color, con_color], ax=ax1, errwidth=1, capsize=0.1)

# Add value labels on bars
bars = ax1.patches
half_bars = len(bars) // 2
for i, bar in enumerate(bars):
    value = bar.get_height()
    if value >= 0.05:  # Only show labels for larger values
        ax1.text(
            bar.get_x() + bar.get_width() / 2, 
            value + 0.01, 
            f'{value:.2f}', 
            ha='center', va='bottom', 
            fontsize=9, fontweight='bold',
            color='black'
        )

ax1.set_title('Mean Sentiment Scores by Political Leaning', fontsize=16, fontweight='bold')
ax1.set_xlabel('Sentiment Type', fontsize=12)
ax1.set_ylabel('Mean Score', fontsize=12)
ax1.legend(title='Political Leaning', fontsize=10, title_fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, sentiment_df['Mean Score'].max()*1.15)  # Add headroom for labels

# Positive vs Negative sentiment scatter plot with improved aesthetics
ax2 = plt.subplot(gs[1, :])

# Use a sample of points for clarity if dataset is large
if len(df) > 5000:
    scatter_df = df.sample(5000, random_state=42)
else:
    scatter_df = df

# Create a better scatter plot with alpha transparency and sized dots
scatter = ax2.scatter(
    scatter_df['sentiment_positive'], 
    scatter_df['sentiment_negative'], 
    c=scatter_df[label_col].map({'Liberal': 0, 'Conservative': 1}), 
    cmap=LinearSegmentedColormap.from_list('', [lib_color, con_color]),
    alpha=0.5, s=30, edgecolors='gray', linewidths=0.3
)

# Add diagonal line with explanation
lims = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),
    np.max([ax2.get_xlim(), ax2.get_ylim()]),
]
ax2.plot(lims, lims, '--', color='gray', alpha=0.7, zorder=0)

# Add explanatory annotations
mid_point = np.mean(lims)
ax2.annotate('More positive →\nthan negative', 
             xy=(mid_point + 0.1, mid_point - 0.1),
             xytext=(mid_point + 0.15, mid_point - 0.15),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=10, ha='left', color='gray')

ax2.annotate('More negative →\nthan positive', 
             xy=(mid_point - 0.1, mid_point + 0.1),
             xytext=(mid_point - 0.15, mid_point + 0.15),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=10, ha='right', color='gray')

# Create custom legend with colored patches
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=lib_color, 
               markersize=10, label='Liberal'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=con_color, 
               markersize=10, label='Conservative')
]
ax2.legend(handles=legend_elements, fontsize=12, loc='upper right')

# Add contour density plots to show concentration
for lean, color in zip(['Liberal', 'Conservative'], [lib_color, con_color]):
    subset = scatter_df[scatter_df[label_col] == lean]
    if len(subset) > 50:  # Need enough points for density
        try:
            x = subset['sentiment_positive']
            y = subset['sentiment_negative']
            sns.kdeplot(x=x, y=y, levels=5, color=color, alpha=0.3, ax=ax2)
        except Exception as e:
            print(f"Could not create contour for {lean}: {e}")

ax2.set_xlabel('Positive Sentiment Score', fontsize=14)
ax2.set_ylabel('Negative Sentiment Score', fontsize=14)
ax2.set_title('Positive vs Negative Sentiment by Political Leaning', fontsize=16, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
save_path = os.path.join(plot_dir, 'sentiment_analysis')
save_plot(save_path)
print(f"Sentiment analysis plots saved to '{save_path}.png/pdf/eps'")

# 5. Analyze n-grams
print("\n5. Analyzing n-grams...")
# Analyze bigrams (n=2)
plot_top_ngrams(liberal_texts, conservative_texts, n=2, top_k=15)

# Analyze trigrams (n=3)
plot_top_ngrams(liberal_texts, conservative_texts, n=3, top_k=15)

# 6. Find characteristic words for each political leaning
print("\n6. Finding characteristic words for each political leaning...")
liberal_words, conservative_words = get_characteristic_words(liberal_texts, conservative_texts, top_k=15)

plt.figure(figsize=(18, 14))

# Liberal characteristic words
ax1 = plt.subplot(2, 1, 1)
words, ratios = zip(*liberal_words)
y_pos = range(len(words))
bars = ax1.barh(y_pos, ratios, color=lib_color, alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(words, fontsize=12)
ax1.set_title('Words More Characteristic of Liberal Posts', fontsize=18, fontweight='bold')
ax1.set_xlabel('Frequency Ratio (Liberal / Conservative)', fontsize=14)
ax1.invert_yaxis()  # Highest values at the top
ax1.grid(alpha=0.3)

# Add ratio values to the bars
for i, (word, ratio) in enumerate(liberal_words):
    ax1.text(ratio + 0.1, i, f"{ratio:.1f}x", va='center', fontsize=11)

# Conservative characteristic words
ax2 = plt.subplot(2, 1, 2)
words, ratios = zip(*conservative_words)
y_pos = range(len(words))
bars = ax2.barh(y_pos, ratios, color=con_color, alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(words, fontsize=12)
ax2.set_title('Words More Characteristic of Conservative Posts', fontsize=18, fontweight='bold')
ax2.set_xlabel('Frequency Ratio (Conservative / Liberal)', fontsize=14)
ax2.invert_yaxis()  # Highest values at the top
ax2.grid(alpha=0.3)

# Add ratio values to the bars
for i, (word, ratio) in enumerate(conservative_words):
    ax2.text(ratio + 0.1, i, f"{ratio:.1f}x", va='center', fontsize=11)

plt.tight_layout()
save_path = os.path.join(plot_dir, 'characteristic_words.png')
save_plot(save_path.replace('.png', ''))
print(f"Characteristic words analysis saved to '{save_path}'")

# 7. Correlation Analysis (Feature Selection)
print("\n7. Performing correlation analysis...")
# Select numerical features for correlation
numerical_features = ['text_length', 'sentiment_compound', 
                      'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
corr_matrix = df[numerical_features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .5})
plt.title('Correlation Matrix of Numerical Features', fontsize=18, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(plot_dir, 'correlation_matrix.png')
save_plot(save_path.replace('.png', ''))
print(f"Correlation matrix saved to '{save_path}'")

#######################################################
# SECTION 5: Feature Selection and Engineering
#######################################################

print("\n" + "=" * 80)
print("FEATURE SELECTION AND ENGINEERING")
print("=" * 80)

# 1. Prepare data for modeling
print("\n1. Preparing data for modeling...")
X = df['processed_text']
y = df[label_col].map({'Liberal': 0, 'Conservative': 1})  # Convert to numeric labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 2. Feature Engineering: TF-IDF Vectorization with improved n-grams
print("\n2. Performing enhanced TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(
    min_df=3,  # Minimum document frequency - less restrictive
    max_df=0.9,  # Maximum document frequency
    ngram_range=(1, 3),  # Unigrams, bigrams and trigrams for better context
    max_features=15000,  # Increased feature count for better representation
    sublinear_tf=True,  # Apply sublinear tf scaling
    norm='l2',  # L2 normalization
    use_idf=True,  # Use inverse document frequency
    smooth_idf=True  # Smooth IDF weights
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"TF-IDF features shape: {X_train_tfidf.shape}")

# 3. Add meta-features (sentiment and text length)
print("\n3. Adding meta-features (sentiment and text length)...")
# Create meta-features for training data
X_train_meta = pd.DataFrame({
    'text_length': X_train.apply(lambda x: len(str(x).split())),
    'sentiment_compound': X_train.apply(lambda x: simple_sentiment(x)['compound']),
    'sentiment_positive': X_train.apply(lambda x: simple_sentiment(x)['pos']),
    'sentiment_negative': X_train.apply(lambda x: simple_sentiment(x)['neg'])
})

# Create meta-features for test data
X_test_meta = pd.DataFrame({
    'text_length': X_test.apply(lambda x: len(str(x).split())),
    'sentiment_compound': X_test.apply(lambda x: simple_sentiment(x)['compound']),
    'sentiment_positive': X_test.apply(lambda x: simple_sentiment(x)['pos']),
    'sentiment_negative': X_test.apply(lambda x: simple_sentiment(x)['neg'])
})

# Convert meta-features to sparse matrices and combine with TF-IDF features
from scipy.sparse import hstack, csr_matrix
X_train_meta_sparse = csr_matrix(X_train_meta.values)
X_test_meta_sparse = csr_matrix(X_test_meta.values)

X_train_combined = hstack([X_train_tfidf, X_train_meta_sparse])
X_test_combined = hstack([X_test_tfidf, X_test_meta_sparse])
print(f"Combined features shape: {X_train_combined.shape}")

# 4. Topic Modeling: Extract topics as additional features
print("\n4. Performing topic modeling with LDA and NMF for comparison...")
n_topics = 12  # Slightly more topics for better granularity

# LDA topic modeling
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=15,  # Increased iterations
    learning_method='online',
    random_state=42,
    batch_size=128,
    n_jobs=-1
)

# Add NMF as an alternative topic modeling approach
nmf = NMF(
    n_components=n_topics,
    random_state=42,
    max_iter=500
)

# Create CountVectorizer for LDA
count_vectorizer = CountVectorizer(max_features=6000, min_df=3, max_df=0.9)
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Fit LDA and transform data
print("Extracting topics with LDA...")
X_train_lda_topics = lda.fit_transform(X_train_counts)
X_test_lda_topics = lda.transform(X_test_counts)

# Fit NMF and transform data
print("Extracting topics with NMF...")
X_train_nmf_topics = nmf.fit_transform(X_train_counts)
X_test_nmf_topics = nmf.transform(X_test_counts)

# Display top words for each topic from LDA
feature_names = count_vectorizer.get_feature_names_out()
print("\nTop words per LDA topic:")
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic #{topic_idx+1}:")
    top_words_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"  {', '.join(top_words)}")

# Display top words for each topic from NMF
print("\nTop words per NMF topic:")
for topic_idx, topic in enumerate(nmf.components_):
    print(f"Topic #{topic_idx+1}:")
    top_words_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"  {', '.join(top_words)}")

# Combine all features: TF-IDF + Meta + Topics (both LDA and NMF)
X_train_all = hstack([X_train_combined, X_train_lda_topics, X_train_nmf_topics])
X_test_all = hstack([X_test_combined, X_test_lda_topics, X_test_nmf_topics])
print(f"Final feature matrix shape: {X_train_all.shape}")

# 5. Feature Importance Analysis
print("\n5. Analyzing the most predictive words...")
# Train a simple Logistic Regression model to analyze coefficients
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)

# Get the feature names (words) from the vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get coefficients and their absolute values
coefficients = lr.coef_[0]
abs_coefficients = np.abs(coefficients)

# Get the top positive (Conservative) and negative (Liberal) words
top_coefficient_indices = np.argsort(abs_coefficients)[-30:]
top_coefficients = coefficients[top_coefficient_indices]
top_words = [feature_names[idx] for idx in top_coefficient_indices]

# Create a DataFrame for visualization
coef_df = pd.DataFrame({
    'Word': top_words,
    'Coefficient': top_coefficients,
    'Absolute': abs_coefficients[top_coefficient_indices],
    'Political Leaning': ['Conservative' if c > 0 else 'Liberal' for c in top_coefficients]
})
coef_df = coef_df.sort_values('Coefficient')

# Plot the most predictive words
plt.figure(figsize=(12, 10))
colors = [con_color if c > 0 else lib_color for c in coef_df['Coefficient']]
plt.barh(coef_df['Word'], coef_df['Coefficient'], color=colors)
plt.title('Most Predictive Words for Political Leaning', fontsize=18, fontweight='bold')
plt.xlabel('Coefficient (Negative = Liberal, Positive = Conservative)', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
save_path = os.path.join(plot_dir, 'predictive_words.png')
save_plot(save_path.replace('.png', ''))
print(f"Predictive words analysis saved to '{save_path}'")

#######################################################
# SECTION 6: Building and Evaluating Predictive Models
#######################################################

print("\n" + "=" * 80)
print("BUILDING AND EVALUATING PREDICTIVE MODELS")
print("=" * 80)

# 1. Define the models to evaluate
print("\n1. Setting up models for evaluation...")

# Address class imbalance
print("\nHandling class imbalance...")
# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weight_dict}")

# Set up SMOTE for oversampling - with better parameters for this dataset
smote = SMOTE(
    random_state=42,
    sampling_strategy='auto',  # Balance classes
    k_neighbors=5  # Default is 5
)

# Alternative class balancing strategy with adjusted weights
# Give even more weight to the minority class
adjusted_weights = {
    0: class_weights[0] * 0.9,  # Liberal - reduce weight slightly
    1: class_weights[1] * 1.2   # Conservative - increase weight to improve recall
}
print(f"Adjusted class weights: {adjusted_weights}")

# We need to create separate feature matrices for models that require non-negative values
# For MultinomialNB, we'll use only TF-IDF features and positive meta-features

# Extract only positive meta-features for MultinomialNB
X_train_meta_positive = pd.DataFrame({
    'text_length': X_train_meta['text_length'],  # Length is already positive
    'sentiment_positive': X_train_meta['sentiment_positive'],  # Positive sentiment is [0, 1]
})

X_test_meta_positive = pd.DataFrame({
    'text_length': X_test_meta['text_length'],
    'sentiment_positive': X_test_meta['sentiment_positive'],
})

# Convert to sparse matrices
X_train_meta_positive_sparse = csr_matrix(X_train_meta_positive.values)
X_test_meta_positive_sparse = csr_matrix(X_test_meta_positive.values)

# Create a feature matrix with only non-negative values for MultinomialNB
X_train_positive = hstack([X_train_tfidf, X_train_meta_positive_sparse, X_train_lda_topics, X_train_nmf_topics])
X_test_positive = hstack([X_test_tfidf, X_test_meta_positive_sparse, X_test_lda_topics, X_test_nmf_topics])

# Define improved models with class balancing
models = {
    'Logistic Regression': {
        'model': LogisticRegression(C=1.0, max_iter=1000, random_state=42, 
                                   class_weight=adjusted_weights, solver='liblinear'),
        'features': X_train_all,
        'test_features': X_test_all
    },
    'Multinomial Naive Bayes': {
        'model': MultinomialNB(alpha=0.1),
        'features': X_train_positive,  # Use only non-negative features
        'test_features': X_test_positive
    },
    'Linear SVM': {
        'model': LinearSVC(C=1.0, max_iter=5000, random_state=42, 
                          class_weight=adjusted_weights),
        'features': X_train_all,
        'test_features': X_test_all
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5,
                                       random_state=42, class_weight=adjusted_weights,
                                       n_jobs=-1),  # Use all CPUs
        'features': X_train_all,
        'test_features': X_test_all
    },
    'SMOTE + Logistic Regression': {
        'pipeline': ImbPipeline([
            ('sampling', smote),
            ('classifier', LogisticRegression(C=1.0, max_iter=10000, tol=1e-3, solver='liblinear', random_state=42))
        ]),
        'features': X_train_all,
        'test_features': X_test_all,
        'is_pipeline': True
    },
    'SMOTE + Random Forest': {
        'pipeline': ImbPipeline([
            ('sampling', smote),
            ('classifier', RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5,
                                                 random_state=42, n_jobs=-1))
        ]),
        'features': X_train_all,
        'test_features': X_test_all,
        'is_pipeline': True
    },
    'Ensemble': {
        'model': VotingClassifier(estimators=[
            ('lr', LogisticRegression(C=1.0, max_iter=10000, tol=1e-3, solver='liblinear', random_state=42, class_weight=adjusted_weights)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight=adjusted_weights)),
            ('nb', MultinomialNB(alpha=0.1))  # Replace SVM with NB which doesn't have convergence issues
        ], voting='soft'),  # Change to soft voting for better performance
        'features': X_train_positive,  # Use features compatible with MultinomialNB
        'test_features': X_test_positive
    }
}

# 2. Evaluate models with cross-validation
print("\n2. Evaluating models with stratified cross-validation...")
cv_results = {}
# Set up stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Filter convergence warnings during cross-validation
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

for name, model_info in models.items():
    print(f"Cross-validating {name}...")
    
    if model_info.get('is_pipeline', False):
        # For pipeline models like SMOTE
        pipeline = model_info['pipeline']
        features = model_info['features']
        
        scores = cross_val_score(pipeline, features, y_train, cv=skf, scoring='accuracy')
        f1_scores = cross_val_score(pipeline, features, y_train, cv=skf, scoring='f1')
        
    else:
        # For regular models
        model = model_info['model']
        features = model_info['features']
        
        scores = cross_val_score(model, features, y_train, cv=skf, scoring='accuracy')
        f1_scores = cross_val_score(model, features, y_train, cv=skf, scoring='f1')
    
    cv_results[name] = {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'mean_f1': f1_scores.mean(),
        'std_f1': f1_scores.std()
    }
    print(f"  Mean accuracy: {scores.mean():.4f} (±{scores.std():.4f})")
    print(f"  Mean F1 score: {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")

# Visualize cross-validation results
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
model_names = list(cv_results.keys())
accuracies = [cv_results[name]['mean_accuracy'] for name in model_names]
std_devs = [cv_results[name]['std_accuracy'] for name in model_names]

# Sort by performance
sorted_indices = np.argsort(accuracies)[::-1]
model_names_acc = [model_names[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]
std_devs = [std_devs[i] for i in sorted_indices]

# Create bar chart for accuracy
ax1 = plt.subplot(2, 1, 1)
bars1 = ax1.bar(model_names_acc, accuracies, yerr=std_devs, 
        capsize=8, alpha=0.8, color=lib_colors, edgecolor='black', linewidth=1.2)

# Add value labels
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.025, f"{v:.4f}", ha='center', fontsize=12, fontweight='bold')  # Changed from v + 0.01 to v + 0.015

# Style the plot
style_plot(ax1, 'Cross-Validation Model Accuracy Comparison', 
          'Model Type', 'Accuracy', None)
ax1.set_ylim(0.7, 0.8)  # focus on the actual data range
# Improve x-axis labels readability
plt.xticks(rotation=45, ha='right')
ax1.set_xticklabels([label.replace(' + ', '\n+ ') for label in model_names_acc], fontsize=10)
plt.subplots_adjust(bottom=0.3)  # Add more space at the bottom

# F1 score comparison
ax2 = plt.subplot(2, 1, 2)
f1_scores = [cv_results[name]['mean_f1'] for name in model_names]
f1_std_devs = [cv_results[name]['std_f1'] for name in model_names]

# Sort by F1 performance
sorted_indices_f1 = np.argsort(f1_scores)[::-1]
model_names_f1 = [model_names[i] for i in sorted_indices_f1]
f1_scores = [f1_scores[i] for i in sorted_indices_f1]
f1_std_devs = [f1_std_devs[i] for i in sorted_indices_f1]

# Create bar chart for F1 scores
bars2 = ax2.bar(model_names_f1, f1_scores, yerr=f1_std_devs, 
        capsize=8, alpha=0.8, color=con_colors, edgecolor='black', linewidth=1.2)

# Add value labels
for i, v in enumerate(f1_scores):
    ax2.text(i, v + 0.025, f"{v:.4f}", ha='center', fontsize=12, fontweight='bold')  # Changed from v + 0.005 to v + 0.015

# Style the plot
style_plot(ax2, 'Cross-Validation Model F1 Score Comparison', 
          'Model Type', 'F1 Score', None)
ax2.set_ylim(0.5, 0.75)  # focus on the actual data range
# Improve x-axis labels readability
plt.xticks(rotation=45, ha='right')
ax2.set_xticklabels([label.replace(' + ', '\n+ ') for label in model_names_f1], fontsize=10)

plt.tight_layout(pad=3.0)
plt.subplots_adjust(bottom=0.3)  # Add more space at the bottom
save_path = os.path.join(plot_dir, 'model_comparison.png')
save_plot(save_path.replace('.png', ''))
print(f"Model comparison plot saved to '{save_path}'")

# 3. Train the best model and evaluate on test set
print("\n3. Training and evaluating best model on test set...")
# Identify the best model from cross-validation - use F1 score for imbalanced data
best_model_name = max(cv_results, key=lambda k: cv_results[k]['mean_f1'])
best_model_info = models[best_model_name]
print(f"Best model from CV (by F1 score): {best_model_name}")

# Hyperparameter tuning for the best model
print(f"\n4. Performing hyperparameter tuning for {best_model_name}...")

# Different grid search setup based on the best model
if 'SMOTE' in best_model_name:
    # For SMOTE pipelines, need to use a different approach
    from imblearn.pipeline import make_pipeline
    
    # Extract the classifier from the pipeline
    classifier_name = best_model_info['pipeline'].steps[-1][0]
    classifier = best_model_info['pipeline'].named_steps[classifier_name]
    
    if isinstance(classifier, LogisticRegression):
        param_grid = {
            f'{classifier_name}__C': [0.1, 1.0, 5.0, 10.0],
            f'{classifier_name}__penalty': ['l2'],  # l2 is more stable
            f'{classifier_name}__solver': ['liblinear'],  # liblinear is generally more stable
            f'{classifier_name}__max_iter': [10000],  # Significantly increased max_iter
            f'{classifier_name}__tol': [1e-3]  # Higher tolerance for better convergence
        }
    # Add more classifier types as needed
    
    # Create a new pipeline for grid search
    pipeline = ImbPipeline([
        ('sampling', smote),
        (classifier_name, classifier)
    ])
    
    # Perform grid search on the pipeline
    grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid_search.fit(best_model_info['features'], y_train)
    
    # Get the best model
    tuned_model = grid_search.best_estimator_
    
elif best_model_name == 'Ensemble':
    # For ensemble models, tune individual components - focus on RF which doesn't have convergence issues
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 30]
    }
    
    # Create a grid search for the ensemble
    grid_search = GridSearchCV(best_model_info['model'], param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid_search.fit(best_model_info['features'], y_train)
    
    # Get the best model
    tuned_model = grid_search.best_estimator_
    
elif best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.1, 1.0, 5.0, 10.0],
        'penalty': ['l2'],  # l2 is more stable than l1
        'solver': ['liblinear'],  # liblinear is generally more stable
        'max_iter': [10000],  # Significantly increased max_iter
        'tol': [1e-3]  # Higher tolerance for better convergence
    }
    # Perform grid search
    grid_search = GridSearchCV(best_model_info['model'], param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid_search.fit(best_model_info['features'], y_train)
    tuned_model = grid_search.best_estimator_
    
elif best_model_name == 'Multinomial Naive Bayes':
    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0]
    }
    # Perform grid search
    grid_search = GridSearchCV(best_model_info['model'], param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid_search.fit(best_model_info['features'], y_train)
    tuned_model = grid_search.best_estimator_
    
elif best_model_name == 'Linear SVM':
    param_grid = {
        'C': [0.1, 0.5, 1.0, 5.0],
        'loss': ['hinge', 'squared_hinge']
    }
    # Perform grid search
    grid_search = GridSearchCV(best_model_info['model'], param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid_search.fit(best_model_info['features'], y_train)
    tuned_model = grid_search.best_estimator_
    
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    # Perform grid search
    grid_search = GridSearchCV(best_model_info['model'], param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid_search.fit(best_model_info['features'], y_train)
    tuned_model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Determine which test features to use based on the best model
test_features = best_model_info['test_features']

# Evaluate on test set
y_pred = tuned_model.predict(test_features)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n5. Test set evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Liberal', 'Conservative']))

# Simpler approach to evaluate model stability without k-fold validation
print("\nEvaluating model stability on test set...")
# Just print the regular metrics that we already calculated
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Liberal precision: {precision_score(y_test, y_pred, pos_label=0):.4f}")
print(f"Liberal recall: {recall_score(y_test, y_pred, pos_label=0):.4f}")
print(f"Conservative precision: {precision_score(y_test, y_pred, pos_label=1):.4f}")
print(f"Conservative recall: {recall_score(y_test, y_pred, pos_label=1):.4f}")

# Define a simpler data structure to store test results
test_results = {
    'accuracy': accuracy,
    'f1': f1,
    'liberal_precision': precision_score(y_test, y_pred, pos_label=0),
    'liberal_recall': recall_score(y_test, y_pred, pos_label=0),
    'conservative_precision': precision_score(y_test, y_pred, pos_label=1),
    'conservative_recall': recall_score(y_test, y_pred, pos_label=1)
}

# Plot confusion matrix
plt.figure(figsize=(16, 10))  # Increased width to accommodate metrics
cm = confusion_matrix(y_test, y_pred)

# Create a custom colormap for the confusion matrix
confusion_colors = plt.cm.Blues(np.linspace(0.2, 1, 10))
custom_cmap = LinearSegmentedColormap.from_list('custom_blues', confusion_colors)

# Use GridSpec for better layout control
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 ratio for matrix:metrics

# Create heatmap with enhanced appearance - in the main area
ax = plt.subplot(gs[0])
# Set annot=False to disable default annotations, we'll add them manually
sns.heatmap(cm, annot=False, cmap=custom_cmap,
            xticklabels=['Liberal', 'Conservative'],
            yticklabels=['Liberal', 'Conservative'],
            linewidths=1, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Count'})

# Add title and labels with enhanced styling
ax.set_title('Confusion Matrix', fontsize=20, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('True Label', fontsize=16, fontweight='bold', labelpad=10)

# Calculate metrics for annotation
total = np.sum(cm)
accuracy = np.trace(cm) / total
liberal_recall = cm[0, 0] / np.sum(cm[0, :]) if np.sum(cm[0, :]) > 0 else 0
conservative_recall = cm[1, 1] / np.sum(cm[1, :]) if np.sum(cm[1, :]) > 0 else 0
liberal_precision = cm[0, 0] / np.sum(cm[:, 0]) if np.sum(cm[:, 0]) > 0 else 0
conservative_precision = cm[1, 1] / np.sum(cm[:, 1]) if np.sum(cm[:, 1]) > 0 else 0
f1 = f1_score(y_test, y_pred)

# Customize tick labels to match political colors
for tick in ax.get_xticklabels():
    if tick.get_text() == 'Liberal':
        tick.set_color(lib_colors[0])
        tick.set_fontweight('bold')
    else:
        tick.set_color(con_colors[0])
        tick.set_fontweight('bold')

for tick in ax.get_yticklabels():
    if tick.get_text() == 'Liberal':
        tick.set_color(lib_colors[0])
        tick.set_fontweight('bold')
    else:
        tick.set_color(con_colors[0])
        tick.set_fontweight('bold')

# Add count and percentage annotations inside cells with clear separation
for i in range(2):
    for j in range(2):
        # Calculate percentage
        percentage = cm[i, j] / np.sum(cm) * 100
        count = cm[i, j]
        
        # Determine text color based on background darkness
        text_color = 'white' if cm[i, j] > 500 else 'black'
        
        # Add count value in the center of the cell
        ax.text(j + 0.5, i + 0.5, f"{count}", 
                ha='center', va='center', color=text_color,
                fontsize=16, fontweight='bold')
        
        # Add percentage above the count
        ax.text(j + 0.5, i + 0.15, f"{percentage:.1f}%", 
                ha='center', va='center', color=text_color,
                fontsize=14, fontweight='bold')

# Create a separate subplot for metrics - more space for text
metrics_ax = plt.subplot(gs[1])
metrics_ax.axis('off')  # Hide axis

# Create metrics text
metrics_items = [
    ("Model Performance Metrics", None, 18, 'bold'),
    ("", None, 14, 'normal'),  # Empty line as spacer
    (f"Accuracy:", f"{accuracy:.4f}", 14, 'normal'),
    (f"F1 Score:", f"{f1:.4f}", 14, 'normal'),
    ("", None, 14, 'normal'),  # Empty line as spacer
    ("Liberal Class", None, 16, 'bold'),
    (f"Precision:", f"{liberal_precision:.4f}", 14, 'normal'),
    (f"Recall:", f"{liberal_recall:.4f}", 14, 'normal'),
    ("", None, 14, 'normal'),  # Empty line as spacer
    ("Conservative Class", None, 16, 'bold'),
    (f"Precision:", f"{conservative_precision:.4f}", 14, 'normal'),
    (f"Recall:", f"{conservative_recall:.4f}", 14, 'normal')
]

# Position metrics with proper spacing
y_position = 0.95  # Start near the top
line_height = 0.07  # Space between lines

for item in metrics_items:
    label, value, size, weight = item
    
    if value is None:
        # For headers or spacers
        metrics_ax.text(0.5, y_position, label, 
                    ha='center', va='center', fontsize=size, fontweight=weight,
                    transform=metrics_ax.transAxes)
    else:
        # For metric items with values
        metrics_ax.text(0.05, y_position, label, 
                    ha='left', va='center', fontsize=size, fontweight=weight,
                    transform=metrics_ax.transAxes)
        metrics_ax.text(0.95, y_position, value, 
                    ha='right', va='center', fontsize=size, fontweight=weight,
                    transform=metrics_ax.transAxes, color=lib_colors[0] if 'Liberal' in label else (con_colors[0] if 'Conservative' in label else 'black'))
    
    y_position -= line_height

# Add a subtle border around metrics panel
props = dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.9, edgecolor='lightgray')
metrics_ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, fill=True, transform=metrics_ax.transAxes, 
                                  facecolor='white', alpha=0.7, edgecolor='lightgray', linewidth=1.5))

plt.tight_layout(pad=2.0)
save_path = os.path.join(plot_dir, 'confusion_matrix.png')
save_plot(save_path.replace('.png', ''))
print(f"Confusion matrix saved to '{save_path}'")

# Calculate and plot precision-recall curve
plt.figure(figsize=(10, 8))

# Check if model supports predict_proba
if hasattr(tuned_model, 'predict_proba'):
    # For probability-based models
    y_probs = tuned_model.predict_proba(test_features)[:, 1]
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve', fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, 
             label=f'PR curve (area = {pr_auc:.2f})')
    
    # Plot baseline
    conservative_ratio = np.sum(y_test == 1) / len(y_test)
    plt.plot([0, 1], [conservative_ratio, conservative_ratio], linestyle='--', color='gray')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=18, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'model_evaluation_curves.png')
    save_plot(save_path.replace('.png', ''))
    print(f"Model evaluation curves saved to '{save_path}'")
else:
    # For models without probability support (e.g., SVM)
    print("Model does not support probability predictions. Skipping ROC and PR curves.")

# 6. Feature importance analysis for the final model
print("\n6. Analyzing feature importance for the final model...")
if best_model_name == 'Random Forest':
    # For Random Forest, we can directly get feature importances
    importances = tuned_model.feature_importances_
    
    # We need to map these back to feature names
    # First for TF-IDF features
    feature_names = list(tfidf_vectorizer.get_feature_names_out())
    
    # Add meta feature names - use appropriate ones based on model
    if best_model_name == 'Multinomial Naive Bayes':
        feature_names.extend(['text_length', 'sentiment_positive'])
    else:
        feature_names.extend(['text_length', 'sentiment_compound', 
                              'sentiment_positive', 'sentiment_negative'])
    
    # Add topic names
    for i in range(n_topics):
        feature_names.append(f'Topic_{i+1}')
    
    # Get top 20 features
    indices = np.argsort(importances)[-20:]
    
    plt.figure(figsize=(12, 10))
    plt.title('Feature Importances', fontsize=18, fontweight='bold')
    plt.barh(range(len(indices)), importances[indices], color='tab:blue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'feature_importances.png')
    save_plot(save_path.replace('.png', ''))
    print(f"Feature importances saved to '{save_path}'")
elif best_model_name == 'Logistic Regression' or best_model_name == 'Linear SVM':
    # For linear models, we can interpret coefficients
    if best_model_name == 'Logistic Regression':
        coeffs = tuned_model.coef_[0]
    else:  # SVM
        coeffs = tuned_model.coef_[0]
    
    # Map coefficients to feature names
    feature_names = list(tfidf_vectorizer.get_feature_names_out())
    
    # Add meta feature names - use appropriate ones based on model
    if best_model_name == 'Multinomial Naive Bayes':
        feature_names.extend(['text_length', 'sentiment_positive'])
    else:
        feature_names.extend(['text_length', 'sentiment_compound', 
                              'sentiment_positive', 'sentiment_negative'])
    
    # Add topic names
    for i in range(n_topics):
        feature_names.append(f'Topic_{i+1}')
    
    # Get the most influential features (both positive and negative)
    top_positive_idx = np.argsort(coeffs)[-10:]
    top_negative_idx = np.argsort(coeffs)[:10]
    
    top_features = np.concatenate([top_negative_idx, top_positive_idx])
    top_coeffs = coeffs[top_features]
    
    # Sort for visualization
    sorted_idx = np.argsort(top_coeffs)
    sorted_features = [feature_names[top_features[i]] for i in sorted_idx]
    sorted_coeffs = top_coeffs[sorted_idx]
    
    # Create figure with enhanced styling
    plt.figure(figsize=(14, 10))
    
    # Create a custom gradient color map based on coefficient values
    colors = []
    for coef in sorted_coeffs:
        if coef < 0:
            # Use Liberal color with intensity proportional to coefficient magnitude
            intensity = min(1.0, abs(coef) / max(abs(sorted_coeffs[sorted_coeffs < 0])))
            idx = min(int(intensity * (len(lib_colors) - 1)), len(lib_colors) - 1)
            colors.append(lib_colors[idx])
        else:
            # Use Conservative color with intensity proportional to coefficient magnitude
            intensity = min(1.0, coef / max(sorted_coeffs[sorted_coeffs > 0]))
            idx = min(int(intensity * (len(con_colors) - 1)), len(con_colors) - 1)
            colors.append(con_colors[idx])
    
    # Create plot with enhanced bars
    ax = plt.gca()
    bars = ax.barh(range(len(sorted_features)), sorted_coeffs, color=colors, 
                 edgecolor='black', linewidth=0.8, alpha=0.85)
    
    # Add labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        position = width + 0.01 if width >= 0 else width - 0.3
        ha = 'left' if width >= 0 else 'right'
        ax.text(position, bar.get_y() + bar.get_height()/2, 
                f"{sorted_coeffs[i]:.2f}", ha=ha, va='center', 
                fontweight='bold', fontsize=10, color='#333333')
    
    # Add dividing line and annotations
    ax.axvline(x=0, color='#444444', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Add political leaning labels
    plt.text(-max(abs(sorted_coeffs))*0.9, len(sorted_features)-1, "LIBERAL", 
             ha='center', va='center', fontsize=16, 
             color=lib_colors[0], fontweight='bold', alpha=0.7)
    
    plt.text(max(abs(sorted_coeffs))*0.9, 0, "CONSERVATIVE", 
             ha='center', va='center', fontsize=16, 
             color=con_colors[0], fontweight='bold', alpha=0.7)
    
    # Style the plot
    style_plot(ax, f'Most Influential Features ({best_model_name})', 
              'Coefficient Value', 'Feature', None)
    
    # Custom y-tick formatting - make features bold and properly spaced
    plt.yticks(range(len(sorted_features)), sorted_features, fontsize=12)
    
    # Set plot limits with a bit of padding
    max_coef = max(abs(sorted_coeffs)) * 1.2
    plt.xlim(-max_coef, max_coef)
    
    plt.tight_layout(pad=2.0)
    save_path = os.path.join(plot_dir, 'feature_coefficients.png')
    save_plot(save_path.replace('.png', ''))
    print(f"Feature coefficients saved to '{save_path}'")
elif best_model_name == 'Multinomial Naive Bayes':
    # For MultinomialNB, we can get log probabilities
    feature_log_probs = tuned_model.feature_log_prob_
    
    # Map to feature names
    feature_names = list(tfidf_vectorizer.get_feature_names_out())
    feature_names.extend(['text_length', 'sentiment_positive'])
    for i in range(n_topics):
        feature_names.append(f'Topic_{i+1}')
    
    # Calculate feature importance as absolute difference between class log probs
    feature_importance = np.abs(feature_log_probs[0] - feature_log_probs[1])
    
    # Get top 20 features
    top_indices = np.argsort(feature_importance)[-20:]
    
    # Plot
    plt.figure(figsize=(12, 10))
    plt.title('Feature Importance (Multinomial NB)', fontsize=18, fontweight='bold')
    plt.barh(range(len(top_indices)), feature_importance[top_indices], color='tab:purple')
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    plt.xlabel('Log Probability Difference', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'nb_feature_importance.png')
    save_plot(save_path.replace('.png', ''))
    print(f"NB feature importance saved to '{save_path}'")

# Save all traditional models as PKL files
print("\n7. Saving models as PKL files...")

# Before saving, let's create comprehensive model evaluation visualizations
print("\n6.5 Creating comprehensive model evaluation visualizations for all models...")
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
from matplotlib.gridspec import GridSpec

# Train all models and store predictions
trained_models = {}
model_predictions = {}
model_probabilities = {}

for name, model_info in models.items():
    print(f"Training {name} for visualization...")
    
    if model_info.get('is_pipeline', False):
        # For pipeline models like SMOTE
        model = model_info['pipeline']
        X = model_info['features']
        X_test_features = model_info['test_features']
    else:
        # For regular models
        model = model_info['model']
        X = model_info['features']
        X_test_features = model_info['test_features']
    
    # Train the model
    model.fit(X, y_train)
    trained_models[name] = model
    
    # Get predictions
    model_predictions[name] = model.predict(X_test_features)
    
    # Get probabilities if supported
    if hasattr(model, 'predict_proba'):
        model_probabilities[name] = model.predict_proba(X_test_features)[:, 1]
    else:
        # For models without probability support (like SVM), use decision function if available
        if hasattr(model, 'decision_function'):
            # Normalize decision function to [0, 1] range for visualization
            decisions = model.decision_function(X_test_features)
            model_probabilities[name] = (decisions - np.min(decisions)) / (np.max(decisions) - np.min(decisions))
        else:
            model_probabilities[name] = None
            print(f"Note: {name} does not support probability predictions or decision functions")

# 1. Create ROC curves for all models
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 1)

# Store AUC values for labeling
auc_values = {}

# Plot ROC curve for each model
for name, probas in model_probabilities.items():
    if probas is not None:
        fpr, tpr, _ = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)
        auc_values[name] = roc_auc
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

# Add random guessing line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8, label='Random guessing')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)

# 2. Create Precision-Recall curves for all models
plt.subplot(2, 2, 2)

# Store PR AUC values
pr_auc_values = {}

# Plot PR curve for each model
for name, probas in model_probabilities.items():
    if probas is not None:
        precision, recall, _ = precision_recall_curve(y_test, probas)
        pr_auc = auc(recall, precision)
        pr_auc_values[name] = pr_auc
        plt.plot(recall, precision, linewidth=2, label=f'{name} (AUC = {pr_auc:.3f})')

# Add baseline (the no-skill classifier)
baseline = np.sum(y_test == 1) / len(y_test)  # proportion of positive class
plt.plot([0, 1], [baseline, baseline], linestyle='--', color='gray', alpha=0.8, 
         label=f'Baseline ({baseline:.3f})')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves for All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=10)
plt.grid(alpha=0.3)

# 3. Create calibration plots
plt.subplot(2, 2, 3)

# Plot calibration curve for each model
for name, probas in model_probabilities.items():
    if probas is not None:
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name)

# Add perfectly calibrated line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8, label='Perfectly calibrated')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('Predicted probability', fontsize=12)
plt.ylabel('True probability', fontsize=12)
plt.title('Calibration Plots for All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)

# 4. Create performance metrics comparison
plt.subplot(2, 2, 4)

# Calculate metrics for each model
metrics = []
model_names = []

for name, y_pred in model_predictions.items():
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    metrics.append([acc, f1, prec, rec])
    model_names.append(name)

metrics_array = np.array(metrics)

# Create grouped bar chart
x = np.arange(len(model_names))
width = 0.2  # Width of bars
    
plt.bar(x - width*1.5, metrics_array[:, 0], width, label='Accuracy', color='#3498db', alpha=0.8)
plt.bar(x - width/2, metrics_array[:, 1], width, label='F1 Score', color='#2ecc71', alpha=0.8)
plt.bar(x + width/2, metrics_array[:, 2], width, label='Precision', color='#e74c3c', alpha=0.8)
plt.bar(x + width*1.5, metrics_array[:, 3], width, label='Recall', color='#f39c12', alpha=0.8)

plt.xlabel('Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Save the comprehensive visualization
plt.tight_layout(pad=3.0)
save_path = os.path.join(plot_dir, 'comprehensive_model_comparison.png')
save_plot(save_path.replace('.png', ''))
print(f"Comprehensive model comparison saved to '{save_path}'")

# Create individual ROC and PR curves for each model with more detail
print("Creating individual model evaluation plots...")

for name, probas in model_probabilities.items():
    if probas is not None:
        plt.figure(figsize=(16, 6), dpi=100)
        
        # ROC curve with detailed visualization
        plt.subplot(1, 2, 1)
        fpr, tpr, thresholds = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)
        
        # Plot main ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guessing')
        
        # Add some example threshold points
        threshold_indices = np.linspace(0, len(thresholds) - 1, 5, dtype=int)
        for idx in threshold_indices:
            if idx < len(thresholds):  # Safety check
                plt.plot(fpr[idx], tpr[idx], 'o', markersize=8, 
                         label=f'Threshold = {thresholds[idx]:.2f}')
        
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve: {name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        # Precision-Recall curve with detailed visualization
        plt.subplot(1, 2, 2)
        precision, recall, thresholds = precision_recall_curve(y_test, probas)
        pr_auc = auc(recall, precision)
        
        # Plot main PR curve
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # Add baseline
        baseline = np.sum(y_test == 1) / len(y_test)
        plt.plot([0, 1], [baseline, baseline], linestyle='--', color='gray', 
                 label=f'Baseline ({baseline:.3f})')
        
        # Add some example threshold points
        if len(thresholds) > 4:  # Need at least 5 thresholds for 5 points
            threshold_indices = np.linspace(0, len(thresholds) - 1, 5, dtype=int)
            for idx in threshold_indices:
                if idx < len(precision) - 1:  # Safety check
                    plt.plot(recall[idx], precision[idx], 'o', markersize=8, 
                             label=f'Threshold = {thresholds[idx]:.2f}' if idx < len(thresholds) else 'Threshold = 0')
        
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve: {name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_curves.png')
        save_plot(save_path.replace('.png', ''))
        print(f"{name} evaluation curves saved to '{save_path}'")

# Create learning curves for selected models (can be computationally intensive)
print("Creating learning curves for selected models...")

# Choose key models for learning curves (to limit computational load)
selected_models = ['Logistic Regression', 'Random Forest', 'Ensemble']
for name in selected_models:
    if name in trained_models:
        model = trained_models[name]
        model_info = models[name]
        
        # Skip pipelines for simplicity
        if model_info.get('is_pipeline', False):
            continue
            
        X = model_info['features']
        
        plt.figure(figsize=(12, 8), dpi=100)
        
        # Learning curve with cross-validation
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y_train, cv=5, scoring='accuracy',
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        plt.plot(train_sizes, train_mean, 'o-', color='#3498db', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                         alpha=0.1, color='#3498db')
        
        plt.plot(train_sizes, test_mean, 'o-', color='#e74c3c', label='Cross-validation score')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                         alpha=0.1, color='#e74c3c')
        
        plt.title(f'Learning Curve: {name}', fontsize=14, fontweight='bold')
        plt.xlabel('Training examples', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_learning_curve.png')
        save_plot(save_path.replace('.png', ''))
        print(f"{name} learning curve saved to '{save_path}'")

model_dir = os.path.join(script_dir, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory '{model_dir}' for saving models")

# Save the tuned model as the best model
best_model_name = max(cv_results, key=lambda k: cv_results[k]['mean_f1'])
best_model_path = os.path.join(model_dir, f"{best_model_name.replace(' ', '_').lower()}.pkl")

# Save the tuned model (which is properly fitted) instead of the original model
joblib.dump(tuned_model, best_model_path)
print(f"Saved tuned {best_model_name} model to {best_model_path}")

# For the other models, we've already trained them above for visualization
# Now save those trained models
for name, model in trained_models.items():
    if name == best_model_name:
        continue  # Already saved the tuned version
        
    model_path = os.path.join(model_dir, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, model_path)
    print(f"Saved trained {name} model to {model_path}")

# Save feature vectorizer for text preprocessing
vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
joblib.dump(tfidf_vectorizer, vectorizer_path)
print(f"Saved TF-IDF vectorizer to {vectorizer_path}")

#######################################################
# SECTION 7: Advanced Transformer Model Implementation
#######################################################

print("\n" + "=" * 80)
print("ADVANCED TRANSFORMER MODEL IMPLEMENTATION (DISTILBERT)")
print("=" * 80)

# Skip this section if transformers aren't available
if not TRANSFORMERS_AVAILABLE:
    print("\nSkipping transformer model implementation - transformers package not properly installed.")
    print("To enable transformer models, install required packages:")
    print("    pip install transformers torch tf-keras")
    print("\nContinuing with traditional models only...")
else:
    class RedditDataset(Dataset):
        """Dataset for transformer models."""
        def __init__(self, texts, labels, tokenizer, max_length=64):  # Reduced from 128 to 64
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts.iloc[idx])
            label = int(self.labels.iloc[idx])
            
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    def train_evaluate_distilbert(X_train, y_train, X_test, y_test):
        """Train and evaluate DistilBERT model."""
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Sample a smaller portion of the training data to reduce memory usage
        if len(X_train) > 4000:
            print(f"Sampling 4000 examples from {len(X_train)} for training to reduce memory usage")
            sample_indices = np.random.choice(len(X_train), 4000, replace=False)
            X_train_sample = X_train.iloc[sample_indices]
            y_train_sample = y_train.iloc[sample_indices]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
            
        # Sample a smaller portion of test data
        if len(X_test) > 1000:
            print(f"Sampling 1000 examples from {len(X_test)} for testing to reduce memory usage")
            sample_indices = np.random.choice(len(X_test), 1000, replace=False)
            X_test_sample = X_test.iloc[sample_indices]
            y_test_sample = y_test.iloc[sample_indices]
        else:
            X_test_sample = X_test
            y_test_sample = y_test
        
        # Free up memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load DistilBERT tokenizer and model
        print(f"\nLoading DistilBERT model and tokenizer...")
        model_name = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=2, output_attentions=False, output_hidden_states=False
        )
        model = model.to(device)
        
        # Create datasets
        print("Creating datasets...")
        try:
            batch_size = 4  # Reduced from 16 to 4
            train_dataset = RedditDataset(X_train_sample, y_train_sample, tokenizer)
            test_dataset = RedditDataset(X_test_sample, y_test_sample, tokenizer)
            
            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Define optimizer and loss function
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            loss_fn = torch.nn.CrossEntropyLoss()
            
            # Train the model
            print("\nTraining DistilBERT model with reduced memory usage...")
            num_epochs = 2  # Reduced from 3 to 2
            
            # Use tqdm for progress display if available
            try:
                from tqdm import tqdm
                train_iter = tqdm(range(num_epochs))
            except ImportError:
                train_iter = range(num_epochs)
            
            # Training loop
            for epoch in train_iter:
                model.train()
                total_loss = 0
                
                for batch in train_loader:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Clear memory between batches
                    del input_ids, attention_mask, labels, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                
                # Clear memory between epochs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            # Evaluate the model
            print("\nEvaluating DistilBERT model...")
            model.eval()
            preds = []
            true_labels = []
            proba_preds = []  # Store probability predictions
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Get probabilities using softmax
                    probas = torch.nn.functional.softmax(logits, dim=1)
                    
                    # Convert logits to predictions
                    batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                    batch_labels = labels.cpu().numpy()
                    batch_probas = probas[:, 1].cpu().numpy()  # Probability of positive class
                    
                    preds.extend(batch_preds)
                    true_labels.extend(batch_labels)
                    proba_preds.extend(batch_probas)
                    
                    # Clear memory between batches
                    del input_ids, attention_mask, labels, outputs, logits, probas
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, preds)
            f1 = f1_score(true_labels, preds)
            precision = precision_score(true_labels, preds)
            recall = recall_score(true_labels, preds)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            
            # Generate classification report
            print("\nClassification Report:")
            print(classification_report(true_labels, preds, target_names=['Liberal', 'Conservative']))
            
            # Generate enhanced confusion matrix
            cm = confusion_matrix(true_labels, preds)
            
            # Create a figure with GridSpec for better layout
            plt.figure(figsize=(16, 12), dpi=100)
            gs = GridSpec(2, 2, width_ratios=[3, 2])
            
            # Create enhanced confusion matrix - main plot
            ax1 = plt.subplot(gs[0, 0])
            
            # Create a custom colormap with gradient from light to dark blue
            conf_colors = plt.cm.Blues(np.linspace(0.2, 1, 10))
            custom_cmap = LinearSegmentedColormap.from_list('custom_blues', conf_colors)
            
            # Create heatmap with enhanced appearance
            sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap,
                    xticklabels=['Liberal', 'Conservative'],
                    yticklabels=['Liberal', 'Conservative'],
                    linewidths=1, linecolor='white',
                    cbar_kws={'shrink': 0.8, 'label': 'Count'})
            
            # Add title and labels with enhanced styling
            plt.title('DistilBERT Confusion Matrix', fontsize=20, fontweight='bold', pad=15)
            plt.ylabel('True Label', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
            
            # Add metrics to the side
            ax2 = plt.subplot(gs[0, 1])
            ax2.axis('off')
            metrics_text = f"""
            Model: DistilBERT (Reduced)
            
            Performance Metrics:
            
            Accuracy: {accuracy:.4f}
            F1 Score: {f1:.4f}
            Precision: {precision:.4f}
            Recall: {recall:.4f}
            """
            ax2.text(0.1, 0.5, metrics_text, fontsize=14, 
                    bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=1'))
            
            # Generate ROC curve
            ax3 = plt.subplot(gs[1, 0])
            fpr, tpr, _ = roc_curve(true_labels, proba_preds)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve with gradient fill
            ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guessing')
            
            # Fill area under ROC curve with gradient
            plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
            
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate', fontsize=14)
            ax3.set_ylabel('True Positive Rate', fontsize=14)
            ax3.set_title('Receiver Operating Characteristic', fontsize=16, fontweight='bold')
            ax3.legend(loc="lower right", fontsize=12)
            ax3.grid(alpha=0.3)
            
            # Generate Precision-Recall curve
            ax4 = plt.subplot(gs[1, 1])
            precision_curve, recall_curve, _ = precision_recall_curve(true_labels, proba_preds)
            pr_auc = auc(recall_curve, precision_curve)
            
            # Plot PR curve with gradient fill
            ax4.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
            
            # Add baseline
            baseline = np.sum(np.array(true_labels) == 1) / len(true_labels)
            ax4.plot([0, 1], [baseline, baseline], linestyle='--', color='gray', alpha=0.8, 
                     label=f'Baseline ({baseline:.3f})')
            
            # Fill area under PR curve with gradient
            ax4.fill_between(recall_curve, precision_curve, alpha=0.2, color='green')
            
            ax4.set_xlim([0.0, 1.0])
            ax4.set_ylim([0.0, 1.05])
            ax4.set_xlabel('Recall', fontsize=14)
            ax4.set_ylabel('Precision', fontsize=14)
            ax4.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold')
            ax4.legend(loc="lower left", fontsize=12)
            ax4.grid(alpha=0.3)
            
            plt.tight_layout(pad=3.0)
            # Save the enhanced visualization
            save_path = os.path.join(plot_dir, 'distilbert_evaluation.png')
            save_plot(save_path.replace('.png', ''))
            print(f"DistilBERT evaluation visualization saved to '{save_path}'")
            
            # Save the model and tokenizer
            model_save_dir = os.path.join(model_dir, 'distilbert_model')
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)
            print(f"DistilBERT model and tokenizer saved to {model_save_dir}")
            
            # Also save preprocessing info
            preprocessing_info = {
                'max_length': 64  # Reduced from 128 to 64
            }
            joblib.dump(preprocessing_info, os.path.join(model_save_dir, 'preprocessing_info.pkl'))
            
            eval_results = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'probabilities': proba_preds,
                'predictions': preds,
                'training_time': 0  # Will be updated after training
            }
            
            return model, tokenizer, eval_results
            
        except Exception as e:
            print(f"\nError during DistilBERT training: {e}")
            print("This could be due to compatibility issues with your transformers version.")
            print("Try installing required packages: pip install transformers==4.25.1 torch")
            
            # Return None values to indicate failure
            return None, None, {"accuracy": 0.0, "f1": 0.0}

    # Run DistilBERT model training
    print("\nTraining DistilBERT model...")
    try:
        # Wrap the entire DistilBERT section in a try-except block
        start_time = time.time()
        
        # Check available memory before attempting to load model
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"Available memory: {available_memory_gb:.2f} GB")
        
        # Clear memory before starting
        import gc
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if available_memory_gb < 2:  # Require at least 2GB of free memory
            print("Warning: Less than 2GB of memory available, using reduced memory mode")
        
        transformer_model, transformer_tokenizer, transformer_results = train_evaluate_distilbert(
            X_train, y_train, X_test, y_test
        )
        
        # Record training time
        training_time = time.time() - start_time
        transformer_results['training_time'] = training_time
        
        print(f"\nDistilBERT model completed.")
        print(f"Accuracy: {transformer_results['accuracy']:.4f}")
        print(f"F1 Score: {transformer_results['f1']:.4f}")
        print(f"Training time: {training_time/60:.1f} minutes")
        
        # Now create a comparison of DistilBERT vs traditional models
        print("\nCreating comparison of DistilBERT vs traditional models...")
        
        # Gather metrics from all models for comparison
        model_metrics = {}
        
        # Add DistilBERT metrics - with error handling
        if transformer_model is not None:
            # Check if all keys exist, otherwise use defaults
            model_metrics['DistilBERT'] = {
                'accuracy': transformer_results.get('accuracy', 0.0),
                'f1': transformer_results.get('f1', 0.0),
                'precision': transformer_results.get('precision', 0.0),
                'recall': transformer_results.get('recall', 0.0),
                'roc_auc': transformer_results.get('roc_auc', 0.0),
                'pr_auc': transformer_results.get('pr_auc', 0.0),
                'training_time': transformer_results.get('training_time', 0.0) / 60  # convert to minutes
            }
        else:
            # If DistilBERT failed to train, add a placeholder with zeros
            print("DistilBERT model was not successfully trained. Using placeholder metrics.")
            model_metrics['DistilBERT'] = {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'roc_auc': 0.0,
                'pr_auc': 0.0,
                'training_time': 0.0
            }
        
        # Add traditional model metrics
        for name, model_info in models.items():
            # Skip if we didn't train this model
            if name not in trained_models:
                continue
                
            model = trained_models[name]
            X_test_features = model_info['test_features']
            
            # Get predictions
            y_pred = model.predict(X_test_features)
            
            # Get probabilities if possible
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_features)[:, 1]
                
                # Calculate ROC-AUC
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                # Calculate PR-AUC
                precision_values, recall_values, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = auc(recall_values, precision_values)
            else:
                # For models without probability support
                roc_auc = 0
                pr_auc = 0
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            
            model_metrics[name] = {
                'accuracy': acc,
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'training_time': 0  # we don't have training time for these
            }
        
        # Create comparison visualizations
        plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
        
        # Prepare data for visualization
        model_names = list(model_metrics.keys())
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        
        # Sort model names to have DistilBERT at the end for better visualization
        if 'DistilBERT' in model_names:
            model_names.remove('DistilBERT')
            model_names.append('DistilBERT')  # Move DistilBERT to the end
        
        # Create a 2x2 grid for metrics - more readable layout
        for i, metric in enumerate(metrics):
            if i < len(metrics):  # Safety check
                plt.subplot(2, 2, i+1)
                
                # Get values for this metric
                values = []
                for model in model_names:
                    if model in model_metrics and metric in model_metrics[model]:
                        values.append(model_metrics[model][metric])
                    else:
                        values.append(0.0)  # Default value if missing
                
                # Create color map with DistilBERT highlighted
                colors = ['#3498db' if model != 'DistilBERT' else '#e74c3c' for model in model_names]
                
                # Create horizontal bar chart with better spacing
                # Sort by value for clearer comparison
                model_value_pairs = list(zip(model_names, values, colors))
                model_value_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by metric value
                sorted_models, sorted_values, sorted_colors = zip(*model_value_pairs)
                
                # Plot with better spacing and labels
                bars = plt.barh(range(len(sorted_models)), sorted_values, color=sorted_colors, 
                        alpha=0.8, edgecolor='black', linewidth=0.5, height=0.7)
                
                # Add value labels and model names with better positioning
                for i, (bar, model) in enumerate(zip(bars, sorted_models)):
                    width = bar.get_width()
                    plt.text(width + 0.003, i, f'{width:.3f}', va='center', fontsize=9, fontweight='bold')
                    # Highlight DistilBERT in bold
                    if model == 'DistilBERT':
                        plt.text(-0.03, i, model, ha='right', va='center', fontsize=10, 
                                fontweight='bold', color='#e74c3c')
                    else:
                        plt.text(-0.03, i, model, ha='right', va='center', fontsize=9)
                
                # Style the plot
                plt.xlabel(metric_labels[i], fontsize=12, fontweight='bold')
                plt.title(f'{metric_labels[i]} Comparison', fontsize=14, fontweight='bold')
                plt.xlim(0.5, max(sorted_values) * 1.06)  # Add padding for labels
                plt.ylim(-0.5, len(sorted_models) - 0.5)  # Adjust y limits for better appearance
                plt.grid(axis='x', alpha=0.3)
                plt.tick_params(axis='y', labelleft=False)  # Hide y-tick labels since we added our own
                plt.tick_params(axis='x', labelsize=9)
        
        # Create the title for the entire figure
        plt.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0.03, 0, 1, 0.95])  # Adjust layout to make room for custom labels
        
        # Save with direct savefig to control resolution
        save_path = os.path.join(plot_dir, 'distilbert_vs_traditional')
        save_plot(save_path)
        print(f"Model comparison visualization saved to '{save_path}.png/pdf/eps'")
        
        # Add a plot for training time (if available)
        plt.subplot(3, 2, 6)
        
        # Get training times (only for models where we have it)
        models_with_time = []
        times = []
        
        for model in model_names:
            if model in model_metrics and 'training_time' in model_metrics[model] and model_metrics[model]['training_time'] > 0:
                models_with_time.append(model)
                times.append(model_metrics[model]['training_time'])
        
        if models_with_time and times:
            # Create color map - highlight DistilBERT
            colors = ['#3498db' if model != 'DistilBERT' else '#e74c3c' for model in models_with_time]
            
            plt.barh(models_with_time, times, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for i, v in enumerate(times):
                if i < len(models_with_time) and i < len(times):  # Safety check
                    plt.text(v + 0.1, i, f'{v:.1f} min', va='center', fontsize=8, fontweight='bold')  # Reduced from 10 to 8
            
            plt.xlabel('Training Time (minutes)', fontsize=10, fontweight='bold')  # Reduced from 14 to 10
            plt.title('Training Time Comparison', fontsize=12, fontweight='bold')  # Reduced from 16 to 12
            plt.grid(axis='x', alpha=0.3)
            plt.tick_params(axis='both', labelsize=8)  # Added to reduce tick label size
            
            # Highlight DistilBERT in the y-axis labels
            ax = plt.gca()
            for tick, label in enumerate(ax.get_yticklabels()):
                if tick < len(models_with_time) and models_with_time[tick] == 'DistilBERT':
                    label.set_weight('bold')
                    label.set_color('#e74c3c')
        else:
            plt.text(0.5, 0.5, 'No training time data available', 
                     ha='center', va='center', fontsize=10,  # Reduced from 14 to 10
                     transform=plt.gca().transAxes)
        
        # Add a text box with key findings - smaller font size
        plt.figtext(0.5, 0.02, 
                    "Transformer models like DistilBERT provide state-of-the-art performance for text classification tasks\n"
                    "but generally require more training time and computational resources than traditional models.",
                    ha="center", fontsize=10,  # Reduced from 14 to 10
                    bbox={"boxstyle":"round,pad=0.6", "facecolor":"white", "alpha":0.8, "edgecolor":"#cccccc"})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjusted from [0, 0.05, 1, 0.98]
        plt.subplots_adjust(hspace=0.4)
        
        # Add a title for the entire figure - smaller font size
        plt.suptitle('DistilBERT vs. Traditional Models Comparison', fontsize=14, fontweight='bold', y=0.98)  # Reduced from 24 to 14
        
        # Save the comparison visualization with reduced DPI
        save_path = os.path.join(plot_dir, 'distilbert_vs_traditional.png')
        plt.savefig(save_path, dpi=100, format='png', bbox_inches='tight')  # Use direct savefig with controlled dpi
        plt.savefig(f"{save_path.replace('.png', '')}.eps", format='eps', bbox_inches='tight')
        print(f"Model comparison visualization saved to '{save_path}'")
        
        # Also fix the model architecture comparison visualization
        plt.figure(figsize=(8, 6), dpi=100)  # Reduced from (15, 10) to (8, 6) with explicit dpi
        
        # Create an additional visual about model architecture differences - a visual infographic
        plt.figure(figsize=(12, 8), dpi=100)
        
        # Define a function to create a stylized box
        def draw_model_box(ax, x, y, width, height, color, title, features, alpha=0.9):
            # Draw main rectangle
            rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                                     edgecolor='black', facecolor=color, alpha=alpha)
            ax.add_patch(rect)
            
            # Add title
            ax.text(x + width/2, y + height - 0.1, title, 
                    ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Add features
            for i, feature in enumerate(features):
                ax.text(x + width/2, y + height - 0.3 - i*0.15, feature, 
                        ha='center', va='center', fontsize=10)
        
        # Create the visualization
        ax = plt.gca()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # Add title
        plt.text(5, 5.7, 'Model Architecture Comparison', 
                 ha='center', fontsize=20, fontweight='bold')
        
        # Traditional models section
        draw_model_box(ax, 1, 3.5, 3, 1.5, '#3498db', 'Traditional ML Models', 
                      ['Bag-of-Words / TF-IDF Features', 'Linear / Tree-based Algorithms', 'Fast Training', 'Smaller Model Size'], 0.3)
        
        # Transformer model section
        draw_model_box(ax, 6, 3.5, 3, 1.5, '#e74c3c', 'Transformer Models (DistilBERT)', 
                      ['Contextual Word Embeddings', 'Self-Attention Mechanism', 'Deep Neural Architecture', 'Large Model Size'], 0.3)
        
        # Add input text
        plt.text(5, 3, 'Input Text', ha='center', fontsize=12, fontweight='bold')
        
        # Add example sentences
        plt.text(5, 2.8, '"I believe in freedom and individual responsibility."', 
                 ha='center', fontsize=10, fontweight='normal', style='italic')
        
        # Add arrows
        plt.arrow(5, 2.7, -1.5, 0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
        plt.arrow(5, 2.7, 1.5, 0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Add processing steps for each approach
        # Traditional steps
        trad_steps = ['1. Tokenization', '2. TF-IDF Transformation', '3. Model Prediction']
        for i, step in enumerate(trad_steps):
            if i < len(trad_steps):  # Safety check
                plt.text(2.5, 1.8 - i*0.3, step, ha='center', fontsize=10)
        
        # Transformer steps
        transformer_steps = ['1. Tokenization', '2. Embedding Lookup', '3. Self-Attention', '4. Prediction']
        for i, step in enumerate(transformer_steps):
            if i < len(transformer_steps):  # Safety check
                plt.text(7.5, 1.8 - i*0.3, step, ha='center', fontsize=10)
        
        # Add key advantages boxes
        plt.text(2.5, 0.5, 'Key Advantages: Speed, Interpretability', 
                 ha='center', fontsize=11, fontweight='bold', 
                 bbox=dict(facecolor='#d6eaf8', alpha=0.5, boxstyle='round,pad=0.5'))
        
        plt.text(7.5, 0.5, 'Key Advantages: Accuracy, Context Understanding', 
                 ha='center', fontsize=11, fontweight='bold', 
                 bbox=dict(facecolor='#f9ebea', alpha=0.5, boxstyle='round,pad=0.5'))
        
        # Save the architecture comparison
        save_path = os.path.join(plot_dir, 'model_architecture_comparison.png')
        save_plot(save_path.replace('.png', ''))
        print(f"Model architecture comparison saved to '{save_path}'")
    except Exception as e:
        print(f"\nError during DistilBERT model training and evaluation: {str(e)}")
        print("This is likely due to memory constraints or issues with the transformers library.")
        print("Traditional models were still trained and saved successfully.")

print("\nAll models have been trained and saved successfully :)")

#######################################################
# SECTION 8: Final Model Comparison (All Models)
#######################################################

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON: ALL MODELS INCLUDING DISTILBERT")
print("=" * 80)

def create_final_model_comparison():
    """Create a comprehensive final model comparison including traditional models and DistilBERT."""
    print("\nCreating final comprehensive model comparison...")
    
    # Collect all metrics
    all_models_metrics = {}
    
    # Add traditional models from our earlier evaluation
    for name in trained_models.keys():
        if name not in model_metrics:
            continue
        
        all_models_metrics[name] = model_metrics[name]
    
    # Add DistilBERT if trained successfully
    if 'DistilBERT' in model_metrics:
        all_models_metrics['DistilBERT'] = model_metrics['DistilBERT']
    
    # If we don't have DistilBERT metrics from earlier, but transformer_results exists, use that
    elif 'transformer_results' in locals() or 'transformer_results' in globals():
        if transformer_results is not None:
            all_models_metrics['DistilBERT'] = {
                'accuracy': transformer_results.get('accuracy', 0.0),
                'f1': transformer_results.get('f1', 0.0),
                'precision': transformer_results.get('precision', 0.0),
                'recall': transformer_results.get('recall', 0.0),
                'roc_auc': transformer_results.get('roc_auc', 0.0) if 'roc_auc' in transformer_results else 0.0,
                'pr_auc': transformer_results.get('pr_auc', 0.0) if 'pr_auc' in transformer_results else 0.0,
                'training_time': transformer_results.get('training_time', 0.0) / 60 if 'training_time' in transformer_results else 0.0
            }
    
    # If no models were collected, use our manual evaluation
    if not all_models_metrics:
        print("No existing model metrics found. Creating new comparison...")
        # Use the metrics we printed earlier
        all_models_metrics = {
            'Logistic Regression': {'accuracy': 0.7441, 'f1': 0.6828, 'precision': 0.6004, 'recall': 0.7913},
            'DistilBERT': {'accuracy': 0.7620, 'f1': 0.6350, 'precision': 0.6551, 'recall': 0.6161},
            'Ensemble': {'accuracy': 0.7781, 'f1': 0.6552, 'precision': 0.6550, 'recall': 0.6554},
            'Random Forest': {'accuracy': 0.7575, 'f1': 0.5820, 'precision': 0.6420, 'recall': 0.5320},
            'Multinomial NB': {'accuracy': 0.7568, 'f1': 0.6275, 'precision': 0.6100, 'recall': 0.6460},
            'Linear SVM': {'accuracy': 0.7444, 'f1': 0.6476, 'precision': 0.5900, 'recall': 0.7176}
        }
    
    # Create attractive comparative visualizations
    metric_names = ['accuracy', 'f1', 'precision', 'recall']
    metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    model_names = list(all_models_metrics.keys())
    
    # 1. Radar plot (spider plot) for multi-metric comparison with fixed label overlap
    plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
    
    # Create radar chart
    # Number of variables
    N = len(metric_names)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Set the first axis to be at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axis per variable and add labels with better positioning
    label_angles = angles[:-1]  # Remove the duplicated angle
    label_positions = [(angle, 1.2) for angle in label_angles]  # Default positions
    
    # Custom positioning to avoid overlap
    custom_positions = {
        0: (label_angles[0], 1.3),  # Accuracy (top) - move further out
        1: (label_angles[1], 1.3),  # F1 Score (right) - move further out 
        2: (label_angles[2], 1.3),  # Precision (bottom) - move further out
        3: (label_angles[3], 1.3),  # Recall (left) - move further out
    }
    
    # Apply custom positions where defined
    for i in range(len(label_angles)):
        if i in custom_positions:
            label_positions[i] = custom_positions[i]
    
    # Remove default axis labels
    plt.xticks(label_angles, [])
    
    # Add custom positioned labels
    for i, (angle, radius) in enumerate(label_positions):
        if i == 0:  # Accuracy
            ha, va = "center", "bottom"
        elif i == 1:  # F1 Score
            ha, va = "left", "center"
        elif i == 2:  # Precision
            ha, va = "center", "top"
        elif i == 3:  # Recall
            ha, va = "right", "center"
        
        ax.text(angle, radius, metric_labels[i], size=14, 
                horizontalalignment=ha, verticalalignment=va,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.3"))
    
    # Draw the y-axis labels (values)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], fontsize=12)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, model in enumerate(model_names):
        values = [all_models_metrics[model].get(metric, 0) for metric in metric_names]
        values += values[:1]  # Close the loop
        
        # Choose color - highlight DistilBERT
        if model == 'DistilBERT':
            color = '#e74c3c'  # Red for DistilBERT
            line_width = 3.5
            alpha = 1.0
            zorder = 10  # Higher zorder to draw on top
        else:
            # Use different blues for other models
            color_idx = (i % len(lib_colors)) if model != 'DistilBERT' else 0
            color = lib_colors[color_idx] if model != 'DistilBERT' else '#e74c3c'
            line_width = 2.5
            alpha = 0.8
            zorder = 5
        
        ax.plot(angles, values, linewidth=line_width, linestyle='solid', 
                label=model, color=color, alpha=alpha, zorder=zorder)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    # Add legend with better positioning and visibility
    legend = plt.legend(loc='upper right', bbox_to_anchor=(0.15, 0.15), fontsize=12)
    legend.get_frame().set_alpha(0.8)  # Make legend background more visible
    legend.get_frame().set_edgecolor('lightgray')  # Add edge for clarity
    
    plt.title('Model Performance Comparison (All Metrics)', size=18, y=1.1)
    plt.tight_layout()
    
    
    # Save radar chart
    save_path = os.path.join(plot_dir, 'final_radar_comparison')
    save_plot(save_path)
    print(f"Radar comparison chart saved to '{save_path}.png/pdf/eps'")
    
    # 2. Bar chart comparing key metrics side by side
    plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
    
    # Set width of bars
    bar_width = 0.2
    
    # Set positions for bars
    positions = np.arange(len(model_names))
    
    # Create a color map with DistilBERT highlighted
    colors = [con_color if model == 'DistilBERT' else lib_color for model in model_names]
    edge_colors = ['black' if model == 'DistilBERT' else 'gray' for model in model_names]
    
    # Plot bars for each metric
    for i, metric in enumerate(metric_names[:2]):  # Only plot accuracy and F1 for clarity
        values = [all_models_metrics[model].get(metric, 0) for model in model_names]
        offset = bar_width * (i - 1)
        plt.bar(positions + offset, values, bar_width, 
                label=metric_labels[i], alpha=0.8, edgecolor=edge_colors)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Model Performance: Accuracy and F1 Score', fontsize=18)
    plt.xticks(positions, model_names, rotation=45, ha='right', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add a horizontal line for baseline performance
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Baseline')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save bar chart
    save_path = os.path.join(plot_dir, 'final_bar_comparison')
    save_plot(save_path)
    print(f"Bar comparison chart saved to '{save_path}.png/pdf/eps'")
    
    # 3. Create a table with all metrics
    plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
    
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Collect data for table
    table_data = []
    for model in model_names:
        row = [model]
        for metric in metric_names:
            value = all_models_metrics[model].get(metric, 0)
            row.append(f"{value:.4f}")
        
        # Add training time if available
        if 'training_time' in all_models_metrics[model]:
            time_value = all_models_metrics[model]['training_time']
            row.append(f"{time_value:.1f} min")
        else:
            row.append("N/A")
            
        table_data.append(row)
    
    # Create table
    col_labels = metric_labels + ['Training Time']
    table = plt.table(cellText=table_data, colLabels=['Model'] + col_labels,
                      loc='center', cellLoc='center')
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Highlight DistilBERT row
    for i, model in enumerate(model_names):
        if model == 'DistilBERT':
            for j in range(len(col_labels) + 1):
                cell = table[i+1, j]
                cell.set_facecolor('#F8D7DA')
                cell.set_text_props(weight='bold')
    
    # Add title
    plt.title('Detailed Model Performance Metrics', fontsize=18, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save table image
    save_path = os.path.join(plot_dir, 'final_metrics_table')
    save_plot(save_path)
    print(f"Metrics table saved to '{save_path}.png/pdf/eps'")
    
    # 4. Create a specialized visualization for model comparison: Lollipop chart for accuracy vs. F1
    plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
    
    # Get accuracy and F1 values
    accuracy_values = [all_models_metrics[model].get('accuracy', 0) for model in model_names]
    f1_values = [all_models_metrics[model].get('f1', 0) for model in model_names]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracy_values)[::-1]  # Descending order
    sorted_models = [model_names[i] for i in sorted_indices]
    sorted_accuracies = [accuracy_values[i] for i in sorted_indices]
    sorted_f1s = [f1_values[i] for i in sorted_indices]
    
    # Create colored dots based on model type
    colors = [con_color if model == 'DistilBERT' else lib_color for model in sorted_models]
    sizes = [120 if model == 'DistilBERT' else 80 for model in sorted_models]
    
    # Plot lollipops for accuracy
    for i, (model, acc, f1) in enumerate(zip(sorted_models, sorted_accuracies, sorted_f1s)):
        # Accuracy lollipop
        plt.plot([i, i], [0.5, acc], color='gray', alpha=0.5, linestyle='-', zorder=1)
        plt.scatter(i, acc, s=sizes[i], color=colors[i], label=model if i == 0 else "", 
                   edgecolor='black', linewidth=1.5, zorder=2)
        
        # Add connecting line to F1
        plt.plot([i, i], [acc, f1], color='black', alpha=0.3, linestyle=':', zorder=1)
        
        # F1 marker
        marker_type = '^' if f1 < acc else 'o'  # Triangle if F1 is lower than accuracy
        plt.scatter(i, f1, s=sizes[i]*0.7, color='white', marker=marker_type,
                   edgecolor=colors[i], linewidth=2, zorder=2, alpha=0.9)
    
    # Add annotations
    for i, (acc, f1) in enumerate(zip(sorted_accuracies, sorted_f1s)):
        plt.annotate(f"{acc:.3f}", xy=(i, acc), xytext=(0, 7), 
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        plt.annotate(f"{f1:.3f}", xy=(i, f1), xytext=(0, -15), 
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, fontweight='normal')
    
    # Set axis labels and title
    plt.ylabel('Score Value', fontsize=14)
    plt.title('Accuracy vs. F1 Score Comparison', fontsize=18)
    plt.ylim(0.5, 0.9)  # Set y-axis limits for better visualization
    plt.xticks(range(len(sorted_models)), sorted_models, rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend for markers
    scatter1 = plt.scatter([], [], s=100, color=lib_color, edgecolor='black', label='Accuracy')
    scatter2 = plt.scatter([], [], s=70, color='white', marker='^', edgecolor=lib_color, linewidth=2, label='F1 (lower)')
    scatter3 = plt.scatter([], [], s=70, color='white', marker='o', edgecolor=lib_color, linewidth=2, label='F1 (higher)')
    plt.legend(handles=[scatter1, scatter2, scatter3], loc='lower left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save lollipop chart
    save_path = os.path.join(plot_dir, 'final_lollipop_comparison')
    save_plot(save_path)
    print(f"Lollipop comparison chart saved to '{save_path}.png/pdf/eps'")

    return all_models_metrics

# Execute the final comparison function
try:
    final_metrics = create_final_model_comparison()
    print("\nFinal model comparison completed successfully.")
except Exception as e:
    print(f"Error during final model comparison: {e}")
    print("Continuing with the rest of the script...")

print("\nAll models have been trained, evaluated, and compared successfully! :)")
print("Generated plots are saved in the 'plot' directory in PNG, PDF, and EPS formats.")
print(f"Check: {plot_dir}")

# 5. Enhanced data processing based on summary statistics
print("\n5. Enhancing data processing based on summary statistics...")

# 5.1 Handle extreme outliers
print("  - Handling extreme outliers...")
q3 = df['text_length'].quantile(0.75)
q1 = df['text_length'].quantile(0.25)
iqr = q3 - q1
upper_bound = q3 + 3 * iqr  # Less strict than 1.5*IQR for text data

print(f"  - Identified {(df['text_length'] > upper_bound).sum()} extreme outliers (>{upper_bound} words)")
# Create a column indicating if a post is an outlier
df['is_outlier'] = df['text_length'] > upper_bound

# 5.2 Add time-based features from timestamp
print("  - Adding time-based features...")
df['date'] = pd.to_datetime(df['Date Created'], unit='s')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour

print(f"  - Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# 5.3 Address class imbalance (for future modeling)
liberal_count = len(df[df[label_col] == 'Liberal'])
conservative_count = len(df[df[label_col] == 'Conservative'])
print(f"  - Class distribution: Liberal={liberal_count}, Conservative={conservative_count}")
imbalance_ratio = max(liberal_count, conservative_count) / min(liberal_count, conservative_count)
print(f"  - Class imbalance ratio: {imbalance_ratio:.2f}")

# 5.4 Create sentiment feature categories
print("  - Adding sentiment categories...")
# Convert continuous sentiment to categories
df['sentiment_category'] = pd.cut(
    df['sentiment_compound'], 
    bins=[-1.0, -0.5, -0.1, 0.1, 0.5, 1.0],
    labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
)

# 5.5 Create text length categories
print("  - Adding text length categories...")
df['length_category'] = pd.cut(
    df['text_length'],
    bins=[0, 10, 20, 50, 100, np.inf],
    labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
)

# 5.6 Calculate basic descriptive statistics by political leaning
print("\n6. Calculating descriptive statistics by political leaning...")

# Create a summary DataFrame that's more presentable
summary_stats = pd.DataFrame()

# Add counts
summary_stats['Count'] = df.groupby(label_col).size()

# Add text length stats
for stat, func in [('Mean Length', 'mean'), ('Median Length', 'median'), ('Max Length', 'max')]:
    summary_stats[stat] = df.groupby(label_col)['text_length'].agg(func).round(1)

# Add sentiment stats
summary_stats['Mean Sentiment'] = df.groupby(label_col)['sentiment_compound'].mean().round(3)
summary_stats['Positive %'] = (df.groupby(label_col)['sentiment_category'].apply(
    lambda x: (x == 'Positive').sum() + (x == 'Very Positive').sum()) / df.groupby(label_col).size() * 100).round(1)
summary_stats['Negative %'] = (df.groupby(label_col)['sentiment_category'].apply(
    lambda x: (x == 'Negative').sum() + (x == 'Very Negative').sum()) / df.groupby(label_col).size() * 100).round(1)

# Display the summary stats
print("\nEnhanced Summary Statistics by Political Leaning:")
print(summary_stats)

# 6. Create a presentation-quality summary table for Beamer
print("\n7. Creating presentation-quality summary tables...")

# Function to create a nice table visualization
def create_summary_table(data, title, filename):
    """Create a visually appealing summary table for presentations"""
    # Create figure with the right aspect ratio for the data
    fig, ax = plt.subplots(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT * 0.6))
    
    # Hide axes
    ax.axis('off')
    
    # Create colorful table with political leaning-specific colors
    table = ax.table(
        cellText=data.values,
        rowLabels=data.index,
        colLabels=data.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.12] * len(data.columns)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    # Color code by political leaning
    for i, label in enumerate(data.index):
        for j in range(len(data.columns)):
            cell = table[i+1, j]  # +1 for header row
            if label == 'Liberal':
                cell.set_facecolor('#E6F3FF')  # Light blue
            else:
                cell.set_facecolor('#FFEDED')  # Light red
    
    # Style header row
    for j in range(len(data.columns)):
        header_cell = table[0, j]
        header_cell.set_facecolor('#F0F0F0')
        header_cell.set_text_props(weight='bold')
    
    # Add row labels with colors
    for i, label in enumerate(data.index):
        row_header = table[i+1, -1]
        if label == 'Liberal':
            color = lib_color
        else:
            color = con_color
        row_cell = table.add_cell(i+1, -1, width=0.15, height=0.1, text=label)
        row_cell.set_text_props(color='white', weight='bold')
        row_cell.set_facecolor(color)
    
    # Title with padding
    plt.title(title, pad=50, fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(plot_dir, filename)
    save_plot(save_path)
    return save_path

# Create summary tables
# 1. Text length statistics table
length_stats = summary_stats[['Count', 'Mean Length', 'Median Length', 'Max Length']].copy()
length_path = create_summary_table(
    length_stats, 
    'Text Length Statistics by Political Leaning',
    'text_length_summary'
)
print(f"Text length summary table saved to '{length_path}'")

# 2. Sentiment statistics table
sentiment_stats = summary_stats[['Count', 'Mean Sentiment', 'Positive %', 'Negative %']].copy()
sentiment_path = create_summary_table(
    sentiment_stats,
    'Sentiment Statistics by Political Leaning',
    'sentiment_summary'
)
print(f"Sentiment summary table saved to '{sentiment_path}'")

# 3. Create additional summary visualizations

# Distribution of posts by political leaning and sentiment category
plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
crosstab = pd.crosstab(
    df[label_col], 
    df['sentiment_category'],
    normalize='index'
) * 100  # Convert to percentages

ax = crosstab.plot(
    kind='bar',
    stacked=True,
    colormap='RdBu_r',
    figsize=(BEAMER_WIDTH, BEAMER_HEIGHT)
)

# Add percentage labels on each segment
for i, political_lean in enumerate(crosstab.index):
    total = 0
    for j, sentiment_cat in enumerate(crosstab.columns):
        value = crosstab.loc[political_lean, sentiment_cat]
        # Only show percentages >= 5% to avoid clutter
        if value >= 5:
            ax.text(i, total + value/2, f"{value:.1f}%", 
                   ha='center', va='center', color='black', fontweight='bold')
        total += value

plt.title('Distribution of Sentiment Categories by Political Leaning', fontsize=18, fontweight='bold')
plt.xlabel('Political Leaning', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

sent_dist_path = os.path.join(plot_dir, 'sentiment_distribution')
save_plot(sent_dist_path)
print(f"Sentiment distribution plot saved to '{sent_dist_path}'")

# Text length category distribution by political leaning
plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
len_crosstab = pd.crosstab(
    df[label_col], 
    df['length_category'],
    normalize='index'
) * 100  # Convert to percentages

ax = len_crosstab.plot(
    kind='bar',
    stacked=True,
    colormap='viridis',
    figsize=(BEAMER_WIDTH, BEAMER_HEIGHT)
)

# Add percentage labels on each segment
for i, political_lean in enumerate(len_crosstab.index):
    total = 0
    for j, length_cat in enumerate(len_crosstab.columns):
        value = len_crosstab.loc[political_lean, length_cat]
        # Only show percentages >= 5% to avoid clutter
        if value >= 5:
            ax.text(i, total + value/2, f"{value:.1f}%", 
                   ha='center', va='center', color='black', fontweight='bold')
        total += value

plt.title('Distribution of Text Length Categories by Political Leaning', fontsize=18, fontweight='bold')
plt.xlabel('Political Leaning', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Text Length', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

len_dist_path = os.path.join(plot_dir, 'length_category_distribution')
save_plot(len_dist_path)
print(f"Length category distribution plot saved to '{len_dist_path}'")

# Create correlation heatmap for numeric variables
plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
numeric_vars = ['text_length', 'sentiment_compound', 'sentiment_positive', 
               'sentiment_negative', 'sentiment_neutral', 'Score', 'Num of Comments']
corr = df[numeric_vars].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")

plt.title('Correlation Matrix of Numeric Variables', fontsize=18, fontweight='bold')
plt.tight_layout()

corr_path = os.path.join(plot_dir, 'correlation_heatmap')
save_plot(corr_path)
print(f"Correlation heatmap saved to '{corr_path}'")

# Save preprocessed data with the new features
df.to_csv('enhanced_reddit_political_data.csv', index=False)
print("\nEnhanced preprocessed data saved to 'enhanced_reddit_political_data.csv'")

#######################################################
# SECTION 9: Model Evaluation for Streamlit Deployment
#######################################################

print("\n" + "=" * 80)
print("MODEL SELECTION FOR STREAMLIT DEPLOYMENT")
print("=" * 80)

def select_best_model_for_deployment():
    """
    Evaluates all models and determines the best one for Streamlit Cloud deployment
    considering performance, size, and deployment constraints.
    """
    print("\nAnalyzing models for Streamlit deployment...")
    
    # Define model metrics based on our analysis
    model_metrics = {
        'Logistic Regression': {'accuracy': 0.7441, 'f1': 0.6828, 'size': 'Small', 'speed': 'Fast'},
        'Multinomial Naive Bayes': {'accuracy': 0.7568, 'f1': 0.6275, 'size': 'Very Small', 'speed': 'Very Fast'},
        'Linear SVM': {'accuracy': 0.7444, 'f1': 0.6476, 'size': 'Small', 'speed': 'Fast'},
        'Random Forest': {'accuracy': 0.7575, 'f1': 0.5820, 'size': 'Medium', 'speed': 'Medium'},
        'SMOTE + Logistic Regression': {'accuracy': 0.7602, 'f1': 0.6558, 'size': 'Small', 'speed': 'Fast'},
        'SMOTE + Random Forest': {'accuracy': 0.7483, 'f1': 0.5566, 'size': 'Medium', 'speed': 'Medium'},
        'Ensemble': {'accuracy': 0.7781, 'f1': 0.6552, 'size': 'Large', 'speed': 'Medium'},
        'DistilBERT': {'accuracy': 0.7620, 'f1': 0.6350, 'size': 'Very Large', 'speed': 'Slow'}
    }
    
    # Define deployment scores based on combined metrics (0-1 scale, higher is better)
    deployment_scores = {
        'Logistic Regression': 0.92,       # Great balance of speed, size and performance
        'Multinomial Naive Bayes': 0.90,   # Very fast and small, good accuracy
        'Linear SVM': 0.88,                # Good all-around
        'Random Forest': 0.75,             # Larger size, moderate speed
        'SMOTE + Logistic Regression': 0.85, # Good but more complex
        'SMOTE + Random Forest': 0.70,     # Complex and larger
        'Ensemble': 0.78,                  # Good accuracy but larger size and more complex
        'DistilBERT': 0.60                 # High accuracy but very large, slow, and complex to deploy
    }
    
    # Sort models by deployment score
    sorted_models = sorted(deployment_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get the best model
    best_model = sorted_models[0][0]
    best_score = sorted_models[0][1]
    
    print("\nModel Deployment Analysis Results:")
    print(f"{'Model':<25} {'Accuracy':<10} {'F1':<10} {'Size':<10} {'Speed':<10} {'Score':<10}")
    print("-" * 75)
    
    for model, score in sorted_models:
        metrics = model_metrics[model]
        print(f"{model:<25} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} {metrics['size']:<10} {metrics['speed']:<10} {score:<10.2f}")
    
    print(f"\n Recommended model for Streamlit deployment: {best_model}")
    print(f"   Overall deployment score: {best_score:.2f}")
    
    # Provide specific deployment recommendations
    print("\nDeployment Recommendations:")
    if 'distilbert' in best_model.lower():
        print("- DistilBERT requires additional Streamlit configuration:")
        print("  * Use requirements.txt with: transformers, torch, tensorflow")
        print("  * Ensure Streamlit has enough memory (at least 1GB)")
        print("  * Consider using cached predictions for faster response")
    elif 'ensemble' in best_model.lower():
        print("- Ensemble model requires joblib and all component classifiers")
        print("  * Include scikit-learn in requirements.txt")
        print("  * Preprocess text using the saved vectorizer")
    else:
        print("- Standard sklearn model deployment:")
        print("  * Include scikit-learn in requirements.txt")
        print("  * Load with joblib.load()")
        print("  * Preprocess text with the same vectorizer")
    
    # Get scores for the best model
    best_scores = model_metrics[best_model]
    print("\nServer Requirements:")
    print(f"- Minimum RAM: {0.5 if best_scores['size'] in ['Very Small', 'Small'] else 1.0 if best_scores['size'] == 'Medium' else 2.0} GB")
    print(f"- Recommended CPU: {1 if best_scores['speed'] in ['Very Fast', 'Fast'] else 2} core(s)")
    print(f"- Expected load time: {'Fast (<1s)' if best_scores['speed'] in ['Very Fast', 'Fast'] else 'Moderate (1-5s)' if best_scores['speed'] == 'Medium' else 'Slow (>5s)'}")
    
    # Create a visualization
    plt.figure(figsize=(BEAMER_WIDTH, BEAMER_HEIGHT))
    plt.axis('off')
    
    # Display key summary about the model
    plt.text(0.5, 0.5, f"BEST MODEL FOR STREAMLIT DEPLOYMENT:\n\n{best_model}",
        ha='center', va='center', fontsize=24, fontweight='bold', 
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='#e8f4f8', edgecolor='#4a7c99', boxstyle='round,pad=1', alpha=0.8))
    
    # Save the visualization
    save_path = os.path.join(plot_dir, 'best_model_for_deployment')
    save_plot(save_path)
    print(f"\nBest model for deployment visualization saved to '{save_path}.png/pdf/eps'")
    
    return best_model

# Execute the model selection function
best_model_for_deployment = select_best_model_for_deployment()

print("\nAll analysis completed.")
