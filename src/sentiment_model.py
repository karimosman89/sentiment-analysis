import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

def train_model(data_path):
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    joblib.dump(model, 'sentiment_model.pkl')
    return model

if __name__ == "__main__":
    train_model('data/reviews.csv')
