from flask import Flask, jsonify
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from flask_cors import CORS


# Download necessary NLTK data if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load SpaCy model and other resources
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

# Custom stop words
custom_stop_words = set(['uh', 'um', 'well', 'also', 'like', 'just', 'really', 'us'])
stop_words = set(stopwords.words('english')).union(custom_stop_words)

# Load your CSV file (update the path accordingly)
csv_filename = 'all_youtube_transcripts_2471.csv'
df = pd.read_csv(csv_filename)

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]

# Word frequency function
def word_frequency(words):
    return Counter(words).most_common(20)

# Sentiment analysis function
def sentiment_analysis(text):
    return sia.polarity_scores(str(text))

# Named entity recognition function
def named_entity_recognition(text):
    doc = nlp(str(text))
    return [(ent.text, ent.label_) for ent in doc.ents]

# Preprocess all texts and create additional columns in DataFrame
df['processed_text'] = df['transcript'].apply(preprocess_text)
all_words = [word for words in df['processed_text'] for word in words]
df['sentiment'] = df['transcript'].apply(sentiment_analysis)
df['sentiment_category'] = df['sentiment'].apply(lambda x: 'Positive' if x['compound'] > 0 else ('Negative' if x['compound'] < 0 else 'Neutral'))
df['entities'] = df['transcript'].apply(named_entity_recognition)

@app.route('/api/word-frequency', methods=['GET'])
def get_word_frequency():
    word_freq = word_frequency(all_words)
    return jsonify(word_freq)

@app.route('/api/sentiment-distribution', methods=['GET'])
def get_sentiment_distribution():
    sentiment_counts = df['sentiment_category'].value_counts(normalize=True).to_dict()
    return jsonify(sentiment_counts)

@app.route('/api/named-entities', methods=['GET'])
def get_named_entities():
    all_entities = [ent for entities in df['entities'] for ent in entities]
    most_common_entities = Counter(all_entities).most_common(5)
    return jsonify(most_common_entities)

@app.route('/api/average-sentiment', methods=['GET'])
def get_average_sentiment():
    avg_sentiment = df['sentiment'].apply(pd.Series).mean().to_dict()
    return jsonify(avg_sentiment)

@app.route('/api/topic-modeling', methods=['GET'])
def get_topic_modeling():
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=list(stop_words))
    doc_term_matrix = vectorizer.fit_transform(df['transcript'])
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(doc_term_matrix)

    def print_topics(model, feature_names, n_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        return topics

    feature_names = vectorizer.get_feature_names_out()
    topics_info = print_topics(lda, feature_names, 5)
    
    return jsonify(topics_info)

@app.route('/api/transcripts', methods=['GET'])
def get_all_transcripts():
    # Assuming 'transcript' is the column name in your DataFrame
    transcripts = df[['video_id', 'transcript']].to_dict(orient='records')
    return jsonify(transcripts)

if __name__ == '__main__':
    app.run(port=5328)