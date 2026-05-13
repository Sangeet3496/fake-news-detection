import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Shuffle dataset
data = data.sample(frac=1)

# Input and output
x = data["text"]
y = data["label"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Convert text into vectors
vectorizer = TfidfVectorizer(stop_words="english")

x_train_vector = vectorizer.fit_transform(x_train)
x_test_vector = vectorizer.transform(x_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)

model.fit(x_train_vector, y_train)

# Prediction
y_pred = model.predict(x_test_vector)

# Accuracy
score = accuracy_score(y_test, y_pred)

print("Accuracy:", score * 100)