"""Recurring issues and forecasting.

Functions:
 - top_keywords_by_category(df, category_col='category', text_col='clean_text', top_n=10)
 - forecast_sentiment_timeseries(df, date_col='date', label_col='label', agg='D', periods=30)

Requires: prophet (pip install prophet) or fbprophet depending on environment.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

def top_keywords_by_category(df, category_col='category', text_col='clean_text', top_n=10):
    result = {}
    for cat, group in df.groupby(category_col):
        texts = group[text_col].astype(str).tolist()
        if not texts:
            result[cat] = []
            continue
        vect = TfidfVectorizer(max_features=2000, stop_words='english')
        X = vect.fit_transform(texts)
        avg = np.array(X.mean(axis=0)).ravel()
        terms = vect.get_feature_names_out()
        top_idx = avg.argsort()[::-1][:top_n]
        result[cat] = [(terms[i], float(avg[i])) for i in top_idx]
    return result

def forecast_sentiment_timeseries(df, date_col='date', label_col='label', score_map=None, agg='D', periods=30):
    """
    Aggregates sentiment into a numeric score per date and forecasts next `periods` intervals using Prophet.
    Returns the forecast dataframe (with yhat) and the trained model.
    """
    try:
        from prophet import Prophet
    except Exception:
        raise RuntimeError("Install prophet: pip install prophet")

    if score_map is None:
        # default mapping
        score_map = {'negative':0, 'neutral':1, 'positive':2}

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['score'] = df[label_col].map(score_map).astype(float)
    # aggregate by date (resample)
    daily = df.set_index(date_col).resample(agg)['score'].mean().reset_index()
    daily = daily.rename(columns={date_col: 'ds', 'score': 'y'})
    daily = daily.dropna()
    model = Prophet()
    model.fit(daily)
    future = model.make_future_dataframe(periods=periods, freq=agg)
    forecast = model.predict(future)
    return forecast, model
