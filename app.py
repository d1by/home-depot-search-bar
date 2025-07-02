import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

@st.cache_data
def load_data():
    train = pd.read_csv("train.csv", encoding="latin1")
    desc = pd.read_csv("product_descriptions.csv", encoding="latin1")
    merged = pd.merge(train, desc, on="product_uid", how="left")
    merged['combined'] = merged['product_title'].fillna('') + ' ' + merged['product_description'].fillna('')
    return merged

df = load_data()

@st.cache_data
def preprocess(text_series):
    sw = set(stopwords.words('english'))
    processed = []
    for text in text_series.fillna("").astype(str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]+', ' ', text)
        tokens = [w for w in text.split() if w not in sw and len(w) > 1]
        processed.append(" ".join(tokens))
    return processed

st.title("Home Depot Inventory Search")

df['processed'] = preprocess(df['combined'])

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['processed'])

query = st.text_input("Search Home Depot:")
top_n = st.slider("Top N results to show", 5, 50, 10)

if query:
    query_cleaned = preprocess(pd.Series([query]))[0]
    query_vector = tfidf.transform([query_cleaned])
    
    scores = (query_vector * tfidf_matrix.T).T.toarray().flatten()
    df['score'] = scores
    
    results = df.sort_values(by="score", ascending=False).head(top_n)
    
    st.subheader(f"Top {top_n} Results for: `{query}`")
    st.dataframe(results[['id', 'search_term', 'product_title', 'product_description', 'score']].reset_index(drop=True))