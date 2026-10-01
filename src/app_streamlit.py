"""Streamlit app for the Intelligent Customer Feedback Analysis System.

Features:
 - Upload CSV (expects 'text' column)
 - Preprocess uploaded data
 - Run sentiment model (transformers) if model saved in models/sentiment/
 - Summarize selected feedback (extractive or T5)
 - Show top keywords per category and a simple forecast (if dates present)

Run:
    streamlit run src/app_streamlit.py
"""
import streamlit as st
import pandas as pd
import os
from importlib import import_module
from pathlib import Path

st.set_page_config(page_title="Feedback AI", layout="wide")
st.title("Intelligent Customer Feedback Analysis")

# local imports (ensure src is in PYTHONPATH or run from repo root)
from summarizer_extractive import extractive_summary, t5_summarize  # local file in same folder if copied
from data_preprocessing import preprocess_text, clean_text  # or import functions if in package

# Attempt to import transformers pipeline for model inference (if model exists)
USE_TRANSFORMERS = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    USE_TRANSFORMERS = True
except Exception:
    USE_TRANSFORMERS = False

MODEL_DIR = "models/sentiment"

uploaded = st.file_uploader("Upload feedback CSV (must contain 'text' column)", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Uploaded preview")
    st.dataframe(df.head())

    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        with st.spinner("Cleaning & preprocessing..."):
            # Light cleanup & tokenization - uses the simple functions from data_preprocessing
            df['clean_text'] = df['text'].astype(str).apply(lambda x: clean_text(x)).apply(lambda x: preprocess_text(x))
        st.success("Preprocessing done.")
        st.write("Sample cleaned text:")
        st.write(df[['text','clean_text']].head())

        # Sentiment inference if model present
        if os.path.exists(MODEL_DIR) and USE_TRANSFORMERS:
            try:
                sentiment_pipe = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, truncation=True)
                st.info("Loaded local sentiment model.")
                if st.button("Run sentiment analysis on all rows"):
                    with st.spinner("Predicting..."):
                        # For speed, predict in batches
                        texts = df['clean_text'].astype(str).tolist()
                        preds = []
                        for t in texts:
                            if not t:
                                preds.append({'label':'NEUTRAL','score':0.0})
                                continue
                            out = sentiment_pipe(t[:512])[0]  # truncate for safety
                            preds.append(out)
                        df['sentiment_label'] = [p.get('label', '') for p in preds]
                        df['sentiment_score'] = [p.get('score', 0.0) for p in preds]
                    st.success("Predictions added to dataframe.")
                    st.bar_chart(df['sentiment_label'].value_counts())
            except Exception as e:
                st.error(f"Could not load sentiment model: {e}")
        else:
            st.warning("No local sentiment model found (models/sentiment/) or transformers not installed. You can still use extractive summarizer and insights.")

        # Summarization area
        st.subheader("Summarize a feedback entry")
        idx = st.number_input("Feedback index to summarize", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
        if st.button("Get summaries for selected index"):
            text = df.loc[int(idx),'text']
            st.write("Original:", text)
            short = extractive_summary(text, n_sentences=1)
            long = extractive_summary(text, n_sentences=3)
            st.write("Extractive short:", short)
            st.write("Extractive long:", long)
            if st.checkbox("Use T5 summarizer (may be slow)"):
                try:
                    s_short, s_long = t5_summarize(text)
                    st.write("T5 short:", s_short)
                    st.write("T5 long:", s_long)
                except Exception as e:
                    st.error("T5 summarizer not available. Install transformers and model. " + str(e))

        # Insights: top keywords per category
        if st.button("Generate insights (top keywords & optional forecast)"):
            st.info("Computing keywords...")
            try:
                import insights as ins
                df_local = df.copy()
                # ensure category column exists
                if 'category' not in df_local.columns:
                    df_local['category'] = 'unknown'
                keywords = ins.top_keywords_by_category(df_local, category_col='category', text_col='clean_text', top_n=8)
                for cat, terms in keywords.items():
                    st.write(f"**{cat}**: " + ", ".join([t for t,_ in terms]))
                # Forecast if date and label present
                if 'date' in df_local.columns and 'label' in df_local.columns:
                    st.info("Running simple forecast (Prophet). This may take a few seconds.")
                    forecast, model = ins.forecast_sentiment_timeseries(df_local, date_col='date', label_col='label', periods=30)
                    st.line_chart(forecast.set_index('ds')['yhat'].tail(60))
                else:
                    st.warning("Skipping forecast (needs 'date' and 'label' columns)."
                               )
            except Exception as e:
                st.error(f"Insights generation failed: {e}")
