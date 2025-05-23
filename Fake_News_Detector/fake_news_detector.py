import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load datasets
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

# Add label column
true_news['label'] = 'REAL'
fake_news['label'] = 'FAKE'

# Combine datasets
df = pd.concat([true_news, fake_news])

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Drop missing values if any
df = df.dropna(subset=['text'])

# Split data (assuming the news content column is 'text')
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluate model
preds = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, preds))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
