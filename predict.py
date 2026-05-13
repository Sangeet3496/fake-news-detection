import joblib

# Load saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# User input
news = input("Enter news text: ")

# Convert text into vector
news_vec = vectorizer.transform([news])

# Prediction
prediction = model.predict(news_vec)

# Output result
if prediction[0] == 1:
    print("REAL NEWS")
else:
    print("FAKE NEWS")