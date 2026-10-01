# Intelligent Customer Feedback Analysis System

## Overview
End-to-end pipeline to clean, analyze, summarize, and forecast customer feedback.

## Quickstart
1. Create virtual env:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. Generate sample data:
   ```bash
   python src/generate_sample_data.py --out data/raw/sample_feedback.csv --n 2000
   ```
3. Clean:
   ```bash
   python src/data_preprocessing.py --input data/raw/sample_feedback.csv --output data/cleaned/clean_feedback.csv
   ```
4. Train sentiment model (optional / requires GPU recommended):
   ```bash
   python src/train_sentiment.py --data data/cleaned/clean_feedback.csv --out_dir models/sentiment
   ```
5. Run Streamlit UI:
   ```bash
   streamlit run src/app_streamlit.py
   ```

## Files
- `src/` : all scripts (preprocessing, training, summarization, insights, app)
- `data/` : raw and cleaned datasets
- `models/` : saved models
