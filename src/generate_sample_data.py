"""Generate synthetic feedback dataset (>=1000 records).
Usage:
    python src/generate_sample_data.py --out data/raw/sample_feedback.csv --n 2000
"""
import csv
import random
import argparse
from faker import Faker

fake = Faker()

labels = ['positive', 'negative', 'neutral']
categories = ['billing', 'product', 'delivery', 'support', 'ui', 'pricing']
sources = ['email', 'chat', 'social']

BASE = {
    'positive': [
        "Very satisfied with the service. The product works as expected.",
        "Amazing support team. Resolved my issue quickly.",
        "Great user experience and fast delivery."
    ],
    'negative': [
        "The product stopped working after two days. Very frustrating.",
        "Delivery was delayed and the support didn't respond.",
        "Received damaged item and refund process is slow."
    ],
    'neutral': [
        "The product is okay. Nothing special to mention.",
        "I used the feature once. No strong opinion.",
        "The interface is fine but could be improved."
    ]
}

def make_feedback():
    sentiment = random.choices(labels, weights=[0.45,0.35,0.20])[0]
    cat = random.choice(categories)
    txt = random.choice(BASE[sentiment])
    # add noise and variability
    txt = f"{txt} {fake.sentence(nb_words=random.randint(6,12))}"
    return {
        'feedback_id': fake.uuid4(),
        'customer_id': fake.random_int(1000,9999),
        'date': fake.date_between(start_date='-1y', end_date='today').isoformat(),
        'source': random.choice(sources),
        'category': cat,
        'text': txt,
        'label': sentiment
    }

def main(out_path="data/raw/sample_feedback.csv", n=2000):
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=['feedback_id','customer_id','date','source','category','text','label'])
        w.writeheader()
        for _ in range(n):
            w.writerow(make_feedback())
    print(f"Generated {n} records -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/sample_feedback.csv")
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    main(args.out, args.n)
