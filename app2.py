import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch

# Load your labeled climate change tweets CSV with 'tweet' and 'sentiment' columns
data = pd.read_csv('climate_tweets.csv')

# Preprocessing function for tweets
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'httpS+', '', tweet)  # remove URLs
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)  # remove @mentions
    tweet = re.sub(r'[^a-zs]', '', tweet)  # remove special chars and numbers
    tweet = re.sub(r's+', ' ', tweet).strip()  # remove extra spaces
    return tweet

data['cleaned_tweet'] = data['tweet'].apply(preprocess_tweet)

# Mapping sentiment labels to numerical classes
label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
data['label'] = data['sentiment'].map(label_mapping)

# Load ClimateBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/climatebert-base')
model = AutoModel.from_pretrained('yiyanghkust/climatebert-base')

# Function to convert text to embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token representation as embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

# Create embeddings for dataset (may take time depending on dataset size)
embeddings = np.array([get_bert_embedding(t) for t in data['cleaned_tweet']])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, data['label'], test_size=0.3, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))