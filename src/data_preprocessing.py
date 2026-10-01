"""Data cleaning & preprocessing script.

Usage:
    python src/data_preprocessing.py --input data/raw/sample_feedback.csv --output data/cleaned/clean_feedback.csv
Requires: spacy (en_core_web_sm), nltk stopwords
"""
import argparse
import pandas as pd
import re
import os
import spacy
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# load spaCy model (download with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except Exception as e:
    raise RuntimeError("Please install spaCy model: python -m spacy download en_core_web_sm") from e

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r'http\S+|www.\S+', ' ', s)          # remove urls
    s = re.sub(r'\S+@\S+', ' ', s)                 # remove emails
    s = re.sub(r'[^0-9A-Za-z\s\.\,\'\-@]', ' ', s) # remove non-ASCII-ish chars
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def preprocess_text(s: str, lemmatize=True) -> str:
    s = clean_text(s)
    if not s:
        return ""
    doc = nlp(s.lower())
    tokens = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_ if lemmatize else tok.text
        if not lemma or lemma in STOPWORDS:
            continue
        if len(lemma) <= 1:
            continue
        tokens.append(lemma)
    return " ".join(tokens)

def load_and_clean(input_path, output_path):
    df = pd.read_csv(input_path)
    # Drop duplicates by text
    if 'text' in df.columns:
        df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    else:
        raise ValueError("Input CSV must contain 'text' column.")
    # Basic missing handling
    df['text'] = df['text'].fillna("").astype(str)
    df['clean_text'] = df['text'].apply(clean_text)
    # Remove extremely short feedbacks
    df = df[df['clean_text'].str.len() > 5].reset_index(drop=True)
    df['clean_text'] = df['clean_text'].apply(preprocess_text)
    # If label column exists, drop rows with missing labels
    if 'label' in df.columns:
        df = df[df['label'].notna()].reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/cleaned/clean_feedback.csv")
    args = parser.parse_args()
    df = load_and_clean(args.input, args.output)
    print(f"Cleaned dataset saved to {args.output} with shape {df.shape}")
