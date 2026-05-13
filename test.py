import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine data
data = pd.concat([fake, true])

x = data["text"]
y = data["label"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Model
model = SGDClassifier(loss="hinge", max_iter=50)
model.fit(x_train_vec, y_train)

# Prediction
y_pred = model.predict(x_test_vec)

# Accuracy
score = accuracy_score(y_test, y_pred)
print("Accuracy:", round(score * 100, 2))

# User input prediction
news = input("Enter news text: ")
news_vec = vectorizer.transform([news])
prediction = model.predict(news_vec)

if prediction[0] == 1:
    print("REAL NEWS")
else:
    print("FAKE NEWS")

# Save model (IMPORTANT - OUTSIDE if-else)
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")