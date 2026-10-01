"""Simple extractive summarizer using TF-IDF and sentence scoring.

Functions:
 - extractive_summary(text, n_sentences=2)
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def extractive_summary(text, n_sentences=2):
    if not text or len(text.split()) < 20:
        return text
    sents = sent_tokenize(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    vect = TfidfVectorizer(stop_words='english')
    X = vect.fit_transform(sents)
    # Score sentences by sum of TF-IDF weights
    scores = X.sum(axis=1).A1
    top_idx = np.argsort(scores)[::-1][:n_sentences]
    top_idx_sorted = sorted(top_idx)
    chosen = [sents[i] for i in top_idx_sorted]
    return " ".join(chosen)

# Quick T5 summarizer demo (if you want neural summarization)
def t5_summarize(text, max_len_short=40, max_len_long=120, device=-1):
    """
    Returns (short_summary, long_summary) using t5-small.
    Requires: transformers
    """
    try:
        from transformers import pipeline
    except Exception:
        raise RuntimeError("Install transformers to use t5 summarizer: pip install transformers")
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=device)
    short = summarizer(text, max_length=max_len_short, min_length=10, do_sample=False)[0]['summary_text']
    long = summarizer(text, max_length=max_len_long, min_length=30, do_sample=False)[0]['summary_text']
    return short, long
