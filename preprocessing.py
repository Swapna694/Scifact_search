"""Text preprocessing shared by TF-IDF and BM25."""
import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "will", "with", "this", "these", "those", "than", "then", "their", "there",
    "which", "who", "not", "no", "can", "may", "also", "been", "into", "do", "does",
}


def tokenize(text):
    """Lowercase -> split into word tokens -> remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]