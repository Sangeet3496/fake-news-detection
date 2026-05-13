import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0   # FAKE
true["label"] = 1   # REAL

# Combine datasets
data = pd.concat([fake, true])

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Features and labels
X = data["text"]
y = data["label"]

# Convert text into numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy check
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy * 100)

# Save model + vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")