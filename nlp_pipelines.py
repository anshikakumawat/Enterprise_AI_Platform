import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import spacy
from gensim.models import Word2Vec
import gensim.downloader as api


nlp = spacy.load("en_core_web_sm")
glove_model = api.load("glove-wiki-gigaword-50")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def tokenize_text(text):
    """Tokenize input text"""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Remove stopwords"""
    return [word for word in tokens if word.lower() not in stop_words]

def stem_text(tokens):
    """Apply stemming"""
    return [stemmer.stem(word) for word in tokens]

def lemmatize_text(tokens):
    """Apply lemmatization"""
    return [lemmatizer.lemmatize(word) for word in tokens]

def pos_tagging(text):
    """Perform POS tagging"""
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def named_entity_recognition(text):
    """Extract named entities"""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def sentiment_analysis(text):
    """Analyze sentiment"""
    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"
    
def tfidf_features(text):
    """Generate TF-IDF features"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])

    return {
        "features": vectorizer.get_feature_names_out(),
        "values": tfidf_matrix.toarray()[0]
    }

def word2vec_model(tokens):
    """
    Generate Word2Vec embeddings
    """
    model = Word2Vec(
        sentences=[tokens],
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )

    return model

def glove_embedding(tokens):
    """
    Generate GloVe embeddings
    """
    vectors = []

    for token in tokens:
        token = token.lower()
        if token in glove_model:
            vectors.append(glove_model[token])

    if len(vectors) == 0:
        return []

    return np.mean(vectors, axis=0).tolist()

def compare_embeddings():
    """
    Compare embedding techniques
    """
    return {
        "TF-IDF": "Sparse statistical representation based on word frequency.",
        "Word2Vec": "Dense vector representation learned from context.",
        "GloVe": "Dense vector representation learned from global word co-occurrence."
    }
    
def analyze_text(text):
    """
    Complete NLP analysis
    """
    tokens = tokenize_text(text)
    filtered_tokens = remove_stopwords(tokens)
    stemmed_tokens = stem_text(filtered_tokens)
    lemmatized_tokens = lemmatize_text(filtered_tokens)

    return {
        "Tokens": tokens,
        "Filtered Tokens": filtered_tokens,
        "Stemmed Tokens": stemmed_tokens,
        "Lemmatized Tokens": lemmatized_tokens,
        "POS Tags": pos_tagging(text),
        "Named Entities": named_entity_recognition(text),
        "Sentiment": sentiment_analysis(text),
        "TF-IDF": tfidf_features(text),
        "Word2Vec": word2vec_model(lemmatized_tokens),
        "GloVe": glove_embedding(lemmatized_tokens),
        "Comparison": compare_embeddings()
    }

# Testing the NLP Pipeline
if __name__ == "__main__":
    text = "Apple is launching a new product in California. I love this amazing technology."

    result = analyze_text(text)

    for key, value in result.items():
        print("\n", key, ":", value)