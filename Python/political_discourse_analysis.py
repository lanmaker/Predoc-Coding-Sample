"""Political discourse analysis of Reddit posts.

This module covers the ECON 6140 final project:

* automated dataset retrieval from Kaggle
* data cleaning and text preprocessing
* exploratory visualisation of linguistic patterns
* feature engineering (TF-IDF, sentiment, topic modelling, meta features)
* traditional machine-learning model training and evaluation
* optional transformer-based fine-tuning (guarded behind availability checks)
"""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
import ssl
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import kagglehub
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import class_weight

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:  # Optional GPU stack
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore

try:  # Transformers are optional and can be heavy
    import transformers
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DistilBertForSequenceClassification,
                              DistilBertTokenizer, Trainer, TrainingArguments)
    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    transformers = None
    AutoModelForSequenceClassification = object  # type: ignore
    AutoTokenizer = object  # type: ignore
    DistilBertForSequenceClassification = object  # type: ignore
    DistilBertTokenizer = object  # type: ignore
    Trainer = object  # type: ignore
    TrainingArguments = object  # type: ignore
    TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PlotStyle:
    """Matplotlib configuration used across the project."""

    width: float = 12.8
    height: float = 7.2
    style: str = "fivethirtyeight"
    palette: str = "Set2"
    font_family: str = "sans-serif"
    font_choices: Tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")

    def apply(self) -> None:
        plt.style.use(self.style)
        sns.set_palette(self.palette)
        plt.rcParams.update(
            {
                "figure.figsize": (self.width, self.height),
                "font.size": 12,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.transparent": False,
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "ps.fonttype": 42,
                "pdf.fonttype": 42,
                "savefig.facecolor": "white",
                "patch.force_edgecolor": True,
                "patch.facecolor": "white",
                "font.family": self.font_family,
                "font.sans-serif": list(self.font_choices),
                "axes.titlesize": 16,
                "axes.titleweight": "bold",
                "axes.labelsize": 14,
                "axes.labelweight": "bold",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "xtick.major.size": 5,
                "ytick.major.size": 5,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "lines.linewidth": 2.5,
                "lines.markersize": 8,
                "legend.fontsize": 12,
                "legend.frameon": True,
                "legend.framealpha": 0.8,
                "legend.edgecolor": "lightgray",
                "grid.linestyle": "--",
                "grid.linewidth": 0.6,
                "grid.alpha": 0.3,
            }
        )


@dataclass
class ColorPalette:
    """Collection of colours used for political leaning comparisons."""

    liberals: Tuple[str, ...] = ("#1A85FF", "#5AA7FF", "#89C4FF", "#B8E0FF")
    conservatives: Tuple[str, ...] = ("#D41159", "#E54A76", "#F27399", "#FFA6BF")
    neutrals: Tuple[str, ...] = ("#767676", "#A3A3A3", "#D1D1D1", "#F3F3F3")

    @property
    def liberal_main(self) -> str:
        return self.liberals[0]

    @property
    def conservative_main(self) -> str:
        return self.conservatives[0]

    @property
    def liberal_cmap(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list("liberal", list(self.liberals))

    @property
    def conservative_cmap(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list("conservative", list(self.conservatives))

    @property
    def political_cmap(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(
            "political", [self.liberal_main, self.neutrals[0], self.conservative_main]
        )


@dataclass
class ProjectPaths:
    """Directory layout used throughout the project."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    dataset_dir: Path = field(init=False)
    plot_dir: Path = field(init=False)
    model_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.dataset_dir = self.root / "data"
        self.plot_dir = self.root / "plot"
        self.model_dir = self.root / "models"

        for path in (self.dataset_dir, self.plot_dir, self.model_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ProjectConfig:
    """High-level switches controlling the pipeline."""

    dataset: str = "neelgajare/liberals-vs-conservatives-on-reddit-13000-posts"
    text_column: str = "Text"
    title_column: str = "Title"
    label_column: str = "Political Lean"
    min_text_words: int = 5
    test_size: float = 0.25
    random_seed: int = 42
    n_topics: int = 12
    run_transformer: bool = False
    evaluation_sample_size: int = 5000
    figure_dpi_cap: int = 300

    def set_random_seeds(self) -> None:
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if torch is not None:
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():  # pragma: no cover - GPU path
                torch.cuda.manual_seed_all(self.random_seed)


# ---------------------------------------------------------------------------
# Logging and global helper state
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------


STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "amp",
    "like",
    "get",
    "would",
    "one",
    "also",
    "us",
    "say",
    "said",
    "even",
    "people",
    "think",
    "know",
    "going",
    "time",
    "good",
    "make",
    "way",
    "really",
    "thing",
}

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "right",
    "better",
    "best",
    "love",
    "happy",
    "positive",
    "wonderful",
    "nice",
    "amazing",
    "awesome",
    "support",
    "win",
    "winning",
    "success",
    "successful",
    "beneficial",
    "agree",
    "correct",
    "freedom",
    "free",
    "hope",
    "protect",
    "safe",
    "secure",
}

NEGATIVE_WORDS = {
    "bad",
    "worst",
    "terrible",
    "wrong",
    "hate",
    "sad",
    "negative",
    "awful",
    "horrible",
    "poor",
    "disappointing",
    "fail",
    "failure",
    "lose",
    "losing",
    "lost",
    "reject",
    "rejection",
    "harmful",
    "corrupt",
    "disaster",
    "evil",
    "fear",
    "against",
    "attack",
    "crisis",
}


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def simple_lemmatize(word: str) -> str:
    if len(word) < 4:
        return word
    if word.endswith("ing"):
        return word[:-3]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("es"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [simple_lemmatize(tok) for tok in simple_tokenize(text)]
    tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2]
    return " ".join(tokens)


def simple_sentiment(text: str) -> Dict[str, float]:
    if not text or pd.isna(text):
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    words = set(simple_tokenize(str(text)))
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    total = len(words) or 1
    pos = pos_count / total
    neg = neg_count / total
    neu = max(0.0, 1 - (pos + neg))
    compound = 0.0
    if pos + neg:
        compound = (pos - neg) / (pos + neg)
    return {"neg": neg, "neu": neu, "pos": pos, "compound": compound}


def load_vader() -> SentimentIntensityAnalyzer:
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


VADER = load_vader()


def improved_sentiment(text: str) -> Dict[str, float]:
    if not text or pd.isna(text):
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    return VADER.polarity_scores(str(text))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def save_plot(path_without_extension: Path, max_dpi: int, *, figure: Optional[plt.Figure] = None) -> None:
    fig = figure or plt.gcf()
    fig_width, fig_height = fig.get_size_inches()
    max_pixels = 65000
    safe_dpi_width = max_pixels / fig_width
    safe_dpi_height = max_pixels / fig_height
    safe_dpi = min(safe_dpi_width, safe_dpi_height, max_dpi)
    if safe_dpi < max_dpi:
        logging.warning("Reducing DPI from %s to %s to avoid oversized figure", max_dpi, int(safe_dpi))
    actual_dpi = safe_dpi
    plt.savefig(path_without_extension.with_suffix(".png"), dpi=actual_dpi, format="png", bbox_inches="tight")
    plt.savefig(path_without_extension.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(path_without_extension.with_suffix(".eps"), format="eps", bbox_inches="tight")


def style_plot(ax: plt.Axes, title: str, xlabel: str, ylabel: str, legend_title: Optional[str] = None) -> plt.Axes:
    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(alpha=0.3, linestyle="--")
    if legend_title and ax.get_legend():
        ax.legend(title=legend_title, fontsize=12, title_fontsize=13, frameon=True, framealpha=0.8, edgecolor="lightgray")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color("#333333")
    return ax


def get_word_frequencies(texts: Iterable[str]) -> Dict[str, int]:
    word_freq: Dict[str, int] = {}
    for text in texts:
        for word in str(text).split():
            if len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq


def generate_wordcloud(texts: Iterable[str], title: str, filename: str, paths: ProjectPaths, palette: ColorPalette, config: ProjectConfig) -> None:
    from wordcloud import WordCloud

    word_freq = get_word_frequencies(texts)
    colormap = "Blues" if "Liberal" in title else "Reds"
    contour_color = palette.liberal_main if "Liberal" in title else palette.conservative_main
    wordcloud = WordCloud(
        width=int(PlotStyle().width * 120),
        height=int(PlotStyle().height * 80),
        background_color="white",
        max_words=120,
        colormap=colormap,
        contour_width=1,
        contour_color=contour_color,
        prefer_horizontal=0.9,
        random_state=config.random_seed,
    ).generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(PlotStyle().width, PlotStyle().height * 0.8))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=20, fontweight="bold")
    plt.tight_layout(pad=1)
    save_plot(paths.plot_dir / filename, config.figure_dpi_cap, figure=fig)
    plt.close(fig)


def fallback_word_plot(texts: Iterable[str], title: str, filename: str, paths: ProjectPaths, palette: ColorPalette, config: ProjectConfig) -> None:
    word_freq = get_word_frequencies(texts)
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
    fig, ax = plt.subplots(figsize=(12, 6))
    color = palette.liberal_main if "Liberal" in title else palette.conservative_main
    ax.bar(range(len(top_words)), top_words.values(), color=color)
    ax.set_xticks(range(len(top_words)))
    ax.set_xticklabels(top_words.keys(), rotation=45, ha="right")
    ax.set_title(f"Top Words in {title.split('in ')[-1]}")
    fig.tight_layout()
    save_plot(paths.plot_dir / filename, config.figure_dpi_cap, figure=fig)
    plt.close(fig)


def get_top_ngrams(corpus: Iterable[str], n: int = 2, top_k: int = 20) -> List[Tuple[str, int]]:
    vec = CountVectorizer(ngram_range=(n, n)).fit(list(corpus))
    bag_of_words = vec.transform(list(corpus))
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]


def plot_top_ngrams(liberal_texts: Iterable[str], conservative_texts: Iterable[str], paths: ProjectPaths, palette: ColorPalette, config: ProjectConfig, n: int = 2, top_k: int = 20) -> None:
    liberal_ngrams = get_top_ngrams(liberal_texts, n, top_k)
    conservative_ngrams = get_top_ngrams(conservative_texts, n, top_k)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    for ax, ngrams, color, title in zip(
        axes,
        (liberal_ngrams, conservative_ngrams),
        (palette.liberal_main, palette.conservative_main),
        (f"Top {top_k} {n}-grams in Liberal Posts", f"Top {top_k} {n}-grams in Conservative Posts"),
    ):
        terms, freqs = zip(*ngrams)
        ax.barh(range(len(terms)), freqs, color=color)
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Frequency", fontsize=14)
    fig.tight_layout()
    save_plot(paths.plot_dir / f"{n}gram_analysis", config.figure_dpi_cap, figure=fig)
    plt.close(fig)


def get_characteristic_words(texts1: Iterable[str], texts2: Iterable[str], top_k: int = 20) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    freq1 = get_word_frequencies(texts1)
    freq2 = get_word_frequencies(texts2)
    all_words = set(freq1) | set(freq2)
    smoothing = 1
    ratios1 = {word: (freq1.get(word, 0) + smoothing) / (freq2.get(word, 0) + smoothing) for word in all_words if len(word) > 3}
    ratios2 = {word: (freq2.get(word, 0) + smoothing) / (freq1.get(word, 0) + smoothing) for word in all_words if len(word) > 3}
    min_count = 20
    top_words1 = [(word, ratio) for word, ratio in sorted(ratios1.items(), key=lambda x: x[1], reverse=True) if freq1.get(word, 0) >= min_count][:top_k]
    top_words2 = [(word, ratio) for word, ratio in sorted(ratios2.items(), key=lambda x: x[1], reverse=True) if freq2.get(word, 0) >= min_count][:top_k]
    return top_words1, top_words2


def plot_characteristic_word_ratios(liberal_words: List[Tuple[str, float]], conservative_words: List[Tuple[str, float]], paths: ProjectPaths, palette: ColorPalette, cfg: ProjectConfig) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    liberal_ax, conservative_ax = axes
    if liberal_words:
        words, ratios = zip(*liberal_words)
        liberal_ax.barh(range(len(words)), ratios, color=palette.liberal_main, alpha=0.85)
        liberal_ax.set_yticks(range(len(words)))
        liberal_ax.set_yticklabels(words, fontsize=12)
        liberal_ax.set_title("Words More Characteristic of Liberal Posts", fontsize=18, fontweight="bold")
        liberal_ax.set_xlabel("Frequency Ratio (Liberal / Conservative)", fontsize=14)
        liberal_ax.invert_yaxis()
        for idx, ratio in enumerate(ratios):
            liberal_ax.text(ratio + 0.1, idx, f"{ratio:.1f}x", va="center", fontsize=11)
        liberal_ax.grid(alpha=0.3)
    if conservative_words:
        words, ratios = zip(*conservative_words)
        conservative_ax.barh(range(len(words)), ratios, color=palette.conservative_main, alpha=0.85)
        conservative_ax.set_yticks(range(len(words)))
        conservative_ax.set_yticklabels(words, fontsize=12)
        conservative_ax.set_title("Words More Characteristic of Conservative Posts", fontsize=18, fontweight="bold")
        conservative_ax.set_xlabel("Frequency Ratio (Conservative / Liberal)", fontsize=14)
        conservative_ax.invert_yaxis()
        for idx, ratio in enumerate(ratios):
            conservative_ax.text(ratio + 0.1, idx, f"{ratio:.1f}x", va="center", fontsize=11)
        conservative_ax.grid(alpha=0.3)
    fig.tight_layout()
    save_plot(paths.plot_dir / "characteristic_words", cfg.figure_dpi_cap, figure=fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data ingestion and cleaning
# ---------------------------------------------------------------------------


def download_dataset(cfg: ProjectConfig, paths: ProjectPaths) -> Path:
    logging.info("Downloading dataset '%s'", cfg.dataset)
    dataset_path = Path(kagglehub.dataset_download(cfg.dataset))
    paths.dataset_dir.mkdir(exist_ok=True)
    return dataset_path


def locate_csv(dataset_path: Path) -> Path:
    csv_files = [file for file in dataset_path.iterdir() if file.suffix.lower() == ".csv"]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the dataset directory")
    return csv_files[0]


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    logging.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    logging.info("Loaded shape: %s", df.shape)
    return df


def clean_dataset(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    text_col, title_col, label_col = cfg.text_column, cfg.title_column, cfg.label_column
    df[text_col] = df[text_col].fillna(df[title_col])
    df["combined_text_length"] = df[text_col].fillna("").apply(lambda x: len(str(x).split()))
    df = df[df["combined_text_length"] >= cfg.min_text_words].copy()
    df = df.dropna(subset=[text_col, label_col])
    df["processed_text"] = df[text_col].apply(preprocess_text)
    df["text_length"] = df[text_col].apply(lambda x: len(str(x).split()))
    sentiments = df[text_col].apply(improved_sentiment)
    df["sentiment_compound"] = sentiments.apply(lambda x: x["compound"])
    df["sentiment_positive"] = sentiments.apply(lambda x: x["pos"])
    df["sentiment_negative"] = sentiments.apply(lambda x: x["neg"])
    df["sentiment_neutral"] = sentiments.apply(lambda x: x["neu"])
    df.to_csv("preprocessed_reddit_political_data.csv", index=False)
    logging.info("Preprocessed data saved to preprocessed_reddit_political_data.csv")
    return df


# ---------------------------------------------------------------------------
# Exploratory data analysis
# ---------------------------------------------------------------------------


def plot_class_distribution(df: pd.DataFrame, cfg: ProjectConfig, paths: ProjectPaths, palette: ColorPalette) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    value_counts = df[cfg.label_column].value_counts()
    colors = [palette.liberal_main if lbl == "Liberal" else palette.conservative_main for lbl in value_counts.index]
    sns.barplot(x=value_counts.index, y=value_counts.values, palette=colors, ax=ax)
    style_plot(ax, "Distribution of Political Leaning in Dataset", "Political Leaning", "Count")
    for index, value in enumerate(value_counts):
        ax.text(index, value + 50, f"{value:,}", ha="center", fontsize=14, fontweight="bold")
        ax.text(index, value / 2, f"{(value / value_counts.sum()) * 100:.1f}%", ha="center", fontsize=14, color="white", fontweight="bold")
    fig.tight_layout()
    save_plot(paths.plot_dir / "political_leaning_distribution", cfg.figure_dpi_cap, figure=fig)
    plt.close(fig)


def plot_key_variable_distributions(df: pd.DataFrame, cfg: ProjectConfig, paths: ProjectPaths, palette: ColorPalette) -> None:
    numerical_vars = ["text_length", "sentiment_compound", "sentiment_positive", "sentiment_negative"]
    fig, axes = plt.subplots(2, 2, figsize=(PlotStyle().width, PlotStyle().height))
    axes = axes.ravel()
    for idx, var in enumerate(numerical_vars):
        ax = axes[idx]
        sns.histplot(data=df, x=var, hue=cfg.label_column, kde=True, element="step", palette=[palette.liberal_main, palette.conservative_main], ax=ax)
        ax.set_title(f"Distribution of {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_plot(paths.plot_dir / "key_variables_distribution", cfg.figure_dpi_cap, figure=fig)
    plt.close(fig)


def plot_post_length_analysis(df: pd.DataFrame, cfg: ProjectConfig, paths: ProjectPaths, palette: ColorPalette) -> None:
    def freedman_diaconis_bins(data: np.ndarray) -> int:
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(data) ** (1 / 3)) if len(data) else 1
        bin_width = max(bin_width, 1)
        data_range = np.max(data) - np.min(data)
        return int(np.ceil(data_range / bin_width)) if bin_width else 50

    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    fig = plt.figure(figsize=(PlotStyle().width, PlotStyle().height))
    ax0 = plt.subplot(gs[0, :])
    x_limit = df["text_length"].quantile(0.95)
    liberal_data = df[df[cfg.label_column] == "Liberal"]["text_length"].values
    conservative_data = df[df[cfg.label_column] == "Conservative"]["text_length"].values
    trimmed = np.concatenate((liberal_data[liberal_data <= x_limit], conservative_data[conservative_data <= x_limit]))
    optimal_bins = freedman_diaconis_bins(trimmed)
    bin_edges = np.linspace(5, x_limit, optimal_bins)
    for lean in df[cfg.label_column].unique():
        subset = df[df[cfg.label_column] == lean]
        color = palette.liberal_main if lean == "Liberal" else palette.conservative_main
        sns.histplot(data=subset, x="text_length", bins=bin_edges, element="step", color=color, alpha=0.6, kde=True, kde_kws={"bw_adjust": 0.8}, stat="density", ax=ax0, label=lean)
    ax0.set_title("Distribution of Post Lengths by Political Leaning", fontsize=18, fontweight="bold")
    ax0.set_xlabel("Number of Words", fontsize=16)
    ax0.set_ylabel("Density", fontsize=16)
    ax0.legend(title="Political Leaning", fontsize=14, title_fontsize=14)
    liberal_mean = float(np.mean(liberal_data))
    conservative_mean = float(np.mean(conservative_data))
    ax0.axvline(liberal_mean, color=palette.liberal_main, linestyle="--", alpha=0.7)
    ax0.axvline(conservative_mean, color=palette.conservative_main, linestyle="--", alpha=0.7)
    ax0.text(liberal_mean + 2, ax0.get_ylim()[1] * 0.95, f"Liberal Mean: {liberal_mean:.1f}", color=palette.liberal_main, fontweight="bold", ha="left", va="top")
    ax0.text(conservative_mean + 2, ax0.get_ylim()[1] * 0.85, f"Conservative Mean: {conservative_mean:.1f}", color=palette.conservative_main, fontweight="bold", ha="left", va="top")
    ax1 = plt.subplot(gs[1, 0])
    sns.boxplot(data=df, x=cfg.label_column, y="text_length", palette=[palette.liberal_main, palette.conservative_main], ax=ax1)
    ax1.set_title("Post Length Box Plot", fontsize=16)
    ax1.set_xlabel("Political Leaning", fontsize=14)
    ax1.set_ylabel("Number of Words", fontsize=14)
    ax1.grid(alpha=0.3)
    ax2 = plt.subplot(gs[1, 1])
    ax2.axis("off")
    stats_text = "Post Length Statistics:\n\n"
    for label in df[cfg.label_column].unique():
        lengths = df[df[cfg.label_column] == label]["text_length"]
        stats_text += (
            f"{label} Posts:\n"
            f"  Mean: {lengths.mean():.1f} words\n"
            f"  Median: {lengths.median():.1f} words\n"
            f"  Std Dev: {lengths.std():.1f} words\n"
            f"  Max: {lengths.max()} words\n"
            f"  Min: {lengths.min()} words\n\n"
        )
    ax2.text(0, 0.5, stats_text, fontsize=12, verticalalignment="center")
    fig.tight_layout()
    save_plot(paths.plot_dir / "post_length_analysis", cfg.figure_dpi_cap, figure=fig)
    plt.close(fig)


def plot_sentiment_analysis(df: pd.DataFrame, cfg: ProjectConfig, paths: ProjectPaths, palette: ColorPalette) -> None:
    fig = plt.figure(figsize=(PlotStyle().width, PlotStyle().height))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    sentiment_summary = df.groupby(cfg.label_column)[["sentiment_positive", "sentiment_negative", "sentiment_neutral", "sentiment_compound"]].mean().reset_index().melt(id_vars=cfg.label_column, var_name="Sentiment", value_name="Mean Score")
    ax1 = plt.subplot(gs[0, :])
    sns.barplot(data=sentiment_summary, x="Sentiment", y="Mean Score", hue=cfg.label_column, palette=[palette.liberal_main, palette.conservative_main], ax=ax1)
    ax1.set_title("Mean Sentiment Scores by Political Leaning", fontsize=16, fontweight="bold")
    ax1.legend(title="Political Leaning", fontsize=10, title_fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    if len(df) > cfg.evaluation_sample_size:
        scatter_df = df.sample(cfg.evaluation_sample_size, random_state=cfg.random_seed)
    else:
        scatter_df = df
    ax2 = plt.subplot(gs[1, :])
    scatter = ax2.scatter(
        scatter_df["sentiment_positive"],
        scatter_df["sentiment_negative"],
        c=scatter_df[cfg.label_column].map({"Liberal": 0, "Conservative": 1}),
        cmap=LinearSegmentedColormap.from_list("", [palette.liberal_main, palette.conservative_main]),
        alpha=0.5,
        s=30,
        edgecolors="gray",
        linewidths=0.3,
    )
    lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]), max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims, lims, "--", color="gray", alpha=0.7)
    ax2.set_xlabel("Positive Sentiment Score", fontsize=14)
    ax2.set_ylabel("Negative Sentiment Score", fontsize=14)
    ax2.set_title("Positive vs Negative Sentiment by Political Leaning", fontsize=16, fontweight="bold")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    save_plot(paths.plot_dir / "sentiment_analysis", cfg.figure_dpi_cap, figure=fig)
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame, cfg: ProjectConfig, paths: ProjectPaths) -> None:
    numerical_features = ["text_length", "sentiment_compound", "sentiment_positive", "sentiment_negative", "sentiment_neutral"]
    corr_matrix = df[numerical_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=0.5, annot=True, fmt=".2f", cbar_kws={"shrink": 0.5}, ax=ax)
    ax.set_title("Correlation Matrix of Numerical Features", fontsize=18, fontweight="bold")
    fig.tight_layout()
    save_plot(paths.plot_dir / "correlation_matrix", cfg.figure_dpi_cap, figure=fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


@dataclass
class FeatureBundle:
    tfidf_vectorizer: TfidfVectorizer
    X_train_tfidf: Any
    X_test_tfidf: Any
    X_train_meta: pd.DataFrame
    X_test_meta: pd.DataFrame
    X_train_combined: Any
    X_test_combined: Any
    X_train_positive: Any
    X_test_positive: Any
    X_train_topics: Dict[str, Any]
    X_test_topics: Dict[str, Any]


from scipy.sparse import csr_matrix, hstack


def build_feature_bundle(X_train: pd.Series, X_test: pd.Series, cfg: ProjectConfig) -> FeatureBundle:
    tfidf_vectorizer = TfidfVectorizer(
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 3),
        max_features=15000,
        sublinear_tf=True,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    logging.info("TF-IDF feature shape: %s", X_train_tfidf.shape)
    X_train_meta = pd.DataFrame(
        {
            "text_length": X_train.apply(lambda x: len(str(x).split())),
            "sentiment_compound": X_train.apply(lambda x: simple_sentiment(x)["compound"]),
            "sentiment_positive": X_train.apply(lambda x: simple_sentiment(x)["pos"]),
            "sentiment_negative": X_train.apply(lambda x: simple_sentiment(x)["neg"]),
        }
    )
    X_test_meta = pd.DataFrame(
        {
            "text_length": X_test.apply(lambda x: len(str(x).split())),
            "sentiment_compound": X_test.apply(lambda x: simple_sentiment(x)["compound"]),
            "sentiment_positive": X_test.apply(lambda x: simple_sentiment(x)["pos"]),
            "sentiment_negative": X_test.apply(lambda x: simple_sentiment(x)["neg"]),
        }
    )
    X_train_meta_sparse = csr_matrix(X_train_meta.values)
    X_test_meta_sparse = csr_matrix(X_test_meta.values)
    X_train_combined = hstack([X_train_tfidf, X_train_meta_sparse])
    X_test_combined = hstack([X_test_tfidf, X_test_meta_sparse])
    logging.info("Combined feature shape: %s", X_train_combined.shape)
    topics: Dict[str, Tuple[Any, Any]] = {}
    lda = LatentDirichletAllocation(n_components=cfg.n_topics, max_iter=15, learning_method="online", random_state=cfg.random_seed)
    lda_topics_train = lda.fit_transform(X_train_tfidf)
    lda_topics_test = lda.transform(X_test_tfidf)
    topics["lda"] = (csr_matrix(lda_topics_train), csr_matrix(lda_topics_test))
    nmf = NMF(n_components=cfg.n_topics, random_state=cfg.random_seed, init="nndsvda", max_iter=400)
    nmf_topics_train = nmf.fit_transform(X_train_tfidf)
    nmf_topics_test = nmf.transform(X_test_tfidf)
    topics["nmf"] = (csr_matrix(nmf_topics_train), csr_matrix(nmf_topics_test))
    X_train_all = hstack([X_train_combined, topics["lda"][0], topics["nmf"][0]])
    X_test_all = hstack([X_test_combined, topics["lda"][1], topics["nmf"][1]])
    X_train_positive = hstack([X_train_tfidf, csr_matrix(X_train_meta[["text_length", "sentiment_positive"]].values), topics["lda"][0], topics["nmf"][0]])
    X_test_positive = hstack([X_test_tfidf, csr_matrix(X_test_meta[["text_length", "sentiment_positive"]].values), topics["lda"][1], topics["nmf"][1]])
    return FeatureBundle(
        tfidf_vectorizer=tfidf_vectorizer,
        X_train_tfidf=X_train_tfidf,
        X_test_tfidf=X_test_tfidf,
        X_train_meta=X_train_meta,
        X_test_meta=X_test_meta,
        X_train_combined=X_train_combined,
        X_test_combined=X_test_combined,
        X_train_positive=X_train_positive,
        X_test_positive=X_test_positive,
        X_train_topics={"lda": topics["lda"][0], "nmf": topics["nmf"][0], "all": X_train_all},
        X_test_topics={"lda": topics["lda"][1], "nmf": topics["nmf"][1], "all": X_test_all},
    )


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    estimator: Optional[ClassifierMixin] = None
    pipeline: Optional[ImbPipeline] = None
    features: Optional[Any] = None
    test_features: Optional[Any] = None

    def is_pipeline(self) -> bool:
        return self.pipeline is not None


@dataclass
class ModelResults:
    cross_validation: Dict[str, Dict[str, float]]
    trained_models: Dict[str, Any]
    evaluation: Dict[str, float]


def build_model_specs(bundle: FeatureBundle, y_train: pd.Series) -> Dict[str, ModelSpec]:
    class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weight_dict = {cls: class_weights[idx] for idx, cls in enumerate(sorted(np.unique(y_train)))}
    adjusted_weights = {0: weight_dict[0] * 0.9, 1: weight_dict[1] * 1.2}
    smote = SMOTE(random_state=42, sampling_strategy="auto", k_neighbors=5)
    specs = {
        "Logistic Regression": ModelSpec(
            name="Logistic Regression",
            estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="liblinear", class_weight=adjusted_weights),
            features=bundle.X_train_topics["all"],
            test_features=bundle.X_test_topics["all"],
        ),
        "Multinomial Naive Bayes": ModelSpec(
            name="Multinomial Naive Bayes",
            estimator=MultinomialNB(alpha=0.1),
            features=bundle.X_train_positive,
            test_features=bundle.X_test_positive,
        ),
        "Linear SVM": ModelSpec(
            name="Linear SVM",
            estimator=LinearSVC(C=1.0, max_iter=5000, random_state=42, class_weight=adjusted_weights),
            features=bundle.X_train_topics["all"],
            test_features=bundle.X_test_topics["all"],
        ),
        "Random Forest": ModelSpec(
            name="Random Forest",
            estimator=RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42, class_weight=adjusted_weights, n_jobs=-1),
            features=bundle.X_train_topics["all"],
            test_features=bundle.X_test_topics["all"],
        ),
        "SMOTE + Logistic Regression": ModelSpec(
            name="SMOTE + Logistic Regression",
            pipeline=ImbPipeline([
                ("sampling", smote),
                ("classifier", LogisticRegression(C=1.0, max_iter=10000, tol=1e-3, solver="liblinear", random_state=42)),
            ]),
            features=bundle.X_train_topics["all"],
            test_features=bundle.X_test_topics["all"],
        ),
        "SMOTE + Random Forest": ModelSpec(
            name="SMOTE + Random Forest",
            pipeline=ImbPipeline([
                ("sampling", smote),
                ("classifier", RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42, n_jobs=-1)),
            ]),
            features=bundle.X_train_topics["all"],
            test_features=bundle.X_test_topics["all"],
        ),
        "Ensemble": ModelSpec(
            name="Ensemble",
            estimator=VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(C=1.0, max_iter=10000, tol=1e-3, solver="liblinear", random_state=42, class_weight=adjusted_weights)),
                    ("rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight=adjusted_weights)),
                    ("nb", MultinomialNB(alpha=0.1)),
                ],
                voting="soft",
            ),
            features=bundle.X_train_positive,
            test_features=bundle.X_test_positive,
        ),
    }
    return specs


def evaluate_models(specs: Dict[str, ModelSpec], y_train: pd.Series, cfg: ProjectConfig) -> Dict[str, Dict[str, float]]:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_seed)
    results: Dict[str, Dict[str, float]] = {}
    for name, spec in specs.items():
        estimator = spec.pipeline if spec.is_pipeline() else spec.estimator
        features = spec.features
        if estimator is None or features is None:
            logging.warning("Skipping %s because estimator or features are missing", name)
            continue
        logging.info("Cross-validating %s", name)
        accuracy_scores = cross_val_score(estimator, features, y_train, cv=skf, scoring="accuracy")
        f1_scores = cross_val_score(estimator, features, y_train, cv=skf, scoring="f1")
        results[name] = {
            "mean_accuracy": float(accuracy_scores.mean()),
            "std_accuracy": float(accuracy_scores.std()),
            "mean_f1": float(f1_scores.mean()),
            "std_f1": float(f1_scores.std()),
        }
        logging.info(
            "%s - accuracy %.4f ± %.4f | F1 %.4f ± %.4f",
            name,
            results[name]["mean_accuracy"],
            results[name]["std_accuracy"],
            results[name]["mean_f1"],
            results[name]["std_f1"],
        )
    return results


def plot_cross_validation_summary(cv_results: Dict[str, Dict[str, float]], cfg: ProjectConfig, paths: ProjectPaths, palette: ColorPalette) -> None:
    if not cv_results:
        return
    ordered = sorted(cv_results.items(), key=lambda item: item[1]["mean_f1"], reverse=True)
    names = [name for name, _ in ordered]
    accuracies = [metrics["mean_accuracy"] for _, metrics in ordered]
    accuracy_std = [metrics["std_accuracy"] for _, metrics in ordered]
    f1_scores = [metrics["mean_f1"] for _, metrics in ordered]
    f1_std = [metrics["std_f1"] for _, metrics in ordered]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    acc_ax, f1_ax = axes
    acc_ax.bar(names, accuracies, yerr=accuracy_std, color=palette.liberal_main, alpha=0.85, capsize=6)
    acc_ax.set_ylabel("Accuracy", fontsize=14)
    acc_ax.set_title("Cross-Validation Accuracy (Mean ± Std)", fontsize=16, fontweight="bold")
    acc_ax.grid(axis="y", alpha=0.3)
    f1_ax.bar(names, f1_scores, yerr=f1_std, color=palette.conservative_main, alpha=0.85, capsize=6)
    f1_ax.set_ylabel("F1 Score", fontsize=14)
    f1_ax.set_title("Cross-Validation F1 Score (Mean ± Std)", fontsize=16, fontweight="bold")
    f1_ax.set_xticklabels(names, rotation=30, ha="right")
    f1_ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_plot(paths.plot_dir / "cross_validation_summary", cfg.figure_dpi_cap, figure=fig)
    plt.close(fig)


def train_models(specs: Dict[str, ModelSpec], y_train: pd.Series) -> Dict[str, Any]:
    trained: Dict[str, Any] = {}
    for name, spec in specs.items():
        estimator = spec.pipeline if spec.is_pipeline() else spec.estimator
        features = spec.features
        if estimator is None or features is None:
            logging.warning("Skipping training for %s because estimator or features are missing", name)
            continue
        logging.info("Training %s", name)
        fitted = clone(estimator)
        fitted.fit(features, y_train)
        trained[name] = fitted
    return trained


def select_best_model(cv_results: Dict[str, Dict[str, float]]) -> Optional[str]:
    if not cv_results:
        return None
    return max(cv_results.items(), key=lambda item: item[1]["mean_f1"])[0]


def evaluate_on_test(best_name: str, best_model: Any, spec: ModelSpec, y_test: pd.Series, cfg: ProjectConfig, paths: ProjectPaths, palette: ColorPalette) -> Dict[str, float]:
    if spec.test_features is None:
        raise ValueError(f"Test features not available for model {best_name}")
    logging.info("Evaluating %s on the held-out test set", best_name)
    y_pred = best_model.predict(spec.test_features)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision_liberal = precision_score(y_test, y_pred, pos_label=0)
    recall_liberal = recall_score(y_test, y_pred, pos_label=0)
    precision_conservative = precision_score(y_test, y_pred, pos_label=1)
    recall_conservative = recall_score(y_test, y_pred, pos_label=1)
    logging.info("Test accuracy: %.4f | F1: %.4f", accuracy, f1)
    logging.info("\n%s", classification_report(y_test, y_pred, target_names=["Liberal", "Conservative"]))
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=palette.liberal_cmap, xticklabels=["Liberal", "Conservative"], yticklabels=["Liberal", "Conservative"], ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    fig.tight_layout()
    save_plot(paths.plot_dir / f"{best_name.replace(' ', '_').lower()}_confusion_matrix", cfg.figure_dpi_cap, figure=fig)
    plt.close(fig)
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(spec.test_features)[:, 1]
        roc_fpr, roc_tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(roc_fpr, roc_tpr)
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(pr_recall, pr_precision)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        roc_ax, pr_ax = axes
        roc_ax.plot(roc_fpr, roc_tpr, color=palette.conservative_main, linewidth=2)
        roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        roc_ax.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.grid(alpha=0.3)
        pr_ax.plot(pr_recall, pr_precision, color=palette.liberal_main, linewidth=2)
        baseline = np.sum(y_test == 1) / len(y_test)
        pr_ax.plot([0, 1], [baseline, baseline], linestyle="--", color="gray")
        pr_ax.set_title(f"Precision-Recall Curve (AUC = {pr_auc:.3f})")
        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        pr_ax.grid(alpha=0.3)
        fig.tight_layout()
        save_plot(paths.plot_dir / f"{best_name.replace(' ', '_').lower()}_roc_pr", cfg.figure_dpi_cap, figure=fig)
        plt.close(fig)
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "liberal_precision": float(precision_liberal),
        "liberal_recall": float(recall_liberal),
        "conservative_precision": float(precision_conservative),
        "conservative_recall": float(recall_conservative),
    }


def save_artifacts(best_name: str, best_model: Any, bundle: FeatureBundle, cfg: ProjectConfig, paths: ProjectPaths) -> None:
    paths.model_dir.mkdir(exist_ok=True)
    model_path = paths.model_dir / f"{best_name.replace(' ', '_').lower()}.pkl"
    vectorizer_path = paths.model_dir / "tfidf_vectorizer.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(bundle.tfidf_vectorizer, vectorizer_path)
    preprocessing_info = {
        "text_column": cfg.text_column,
        "label_column": cfg.label_column,
        "min_text_words": cfg.min_text_words,
        "n_topics": cfg.n_topics,
    }
    joblib.dump(preprocessing_info, paths.model_dir / "preprocessing_info.pkl")
    logging.info("Saved best model to %s and vectorizer to %s", model_path, vectorizer_path)


def run_transformer_experiment(cfg: ProjectConfig, paths: ProjectPaths) -> None:
    if not cfg.run_transformer:
        return
    if not TRANSFORMERS_AVAILABLE or torch is None:
        logging.warning("Skipping transformer experiment - install transformers and torch to enable this step")
        return
    logging.info("Transformer experiment placeholder. Implement fine-tuning as needed.")


def run_pipeline(cfg: ProjectConfig, paths: ProjectPaths, palette: ColorPalette) -> ModelResults:
    cfg.set_random_seeds()
    dataset_path = download_dataset(cfg, paths)
    csv_path = locate_csv(dataset_path)
    local_csv = paths.dataset_dir / csv_path.name
    if not local_csv.exists():
        shutil.copy(csv_path, local_csv)
    df_raw = load_raw_data(local_csv)
    df = clean_dataset(df_raw, cfg)
    plot_class_distribution(df, cfg, paths, palette)
    plot_key_variable_distributions(df, cfg, paths, palette)
    plot_post_length_analysis(df, cfg, paths, palette)
    plot_sentiment_analysis(df, cfg, paths, palette)
    plot_correlation_matrix(df, cfg, paths)
    liberal_texts = df[df[cfg.label_column] == "Liberal"]["processed_text"]
    conservative_texts = df[df[cfg.label_column] == "Conservative"]["processed_text"]
    try:
        generate_wordcloud(liberal_texts, "Wordcloud for Liberal Posts", "liberal_wordcloud", paths, palette, cfg)
    except Exception as exc:  # pragma: no cover - only triggered when WordCloud fails
        logging.warning("Failed to generate liberal wordcloud: %s", exc)
        fallback_word_plot(liberal_texts, "Liberal Posts", "liberal_word_frequencies", paths, palette, cfg)
    try:
        generate_wordcloud(conservative_texts, "Wordcloud for Conservative Posts", "conservative_wordcloud", paths, palette, cfg)
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to generate conservative wordcloud: %s", exc)
        fallback_word_plot(conservative_texts, "Conservative Posts", "conservative_word_frequencies", paths, palette, cfg)
    plot_top_ngrams(liberal_texts, conservative_texts, paths, palette, cfg, n=2, top_k=15)
    plot_top_ngrams(liberal_texts, conservative_texts, paths, palette, cfg, n=3, top_k=15)
    liberal_words, conservative_words = get_characteristic_words(liberal_texts, conservative_texts, top_k=15)
    plot_characteristic_word_ratios(liberal_words, conservative_words, paths, palette, cfg)
    label_mapping = {"Liberal": 0, "Conservative": 1}
    labels = df[cfg.label_column].map(label_mapping)
    if labels.isnull().any():
        raise ValueError("Encountered labels outside of Liberal/Conservative")
    X_train, X_test, y_train, y_test = train_test_split(df["processed_text"], labels, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=labels)
    logging.info("Training samples: %s | Test samples: %s", X_train.shape[0], X_test.shape[0])
    bundle = build_feature_bundle(X_train, X_test, cfg)
    model_specs = build_model_specs(bundle, y_train)
    cv_results = evaluate_models(model_specs, y_train, cfg)
    plot_cross_validation_summary(cv_results, cfg, paths, palette)
    trained_models = train_models(model_specs, y_train)
    best_name = select_best_model(cv_results)
    if best_name is None:
        raise RuntimeError("No models were successfully trained")
    evaluation = evaluate_on_test(best_name, trained_models[best_name], model_specs[best_name], y_test, cfg, paths, palette)
    save_artifacts(best_name, trained_models[best_name], bundle, cfg, paths)
    run_transformer_experiment(cfg, paths)
    return ModelResults(cross_validation=cv_results, trained_models=trained_models, evaluation=evaluation)


def main() -> None:
    configure_logging()
    plot_style = PlotStyle()
    plot_style.apply()
    palette = ColorPalette()
    paths = ProjectPaths()
    cfg = ProjectConfig()
    results = run_pipeline(cfg, paths, palette)
    logging.info("Best model evaluation metrics: %s", results.evaluation)


if __name__ == "__main__":
    main()
