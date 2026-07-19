import pandas as pd
import pickle
import nltk
import string

from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")

# Load dataset
data = pd.read_csv("news.csv")

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing function
def clean_text(text):
    text = str(text).lower()

    # Remove punctuation
    for p in string.punctuation:
        text = text.replace(p, "")

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = text.split()

    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)

print("\n================================")
print(" AI FAKE NEWS DETECTION TOOL")
print("================================")
print("Dataset Loaded Successfully!")
print("Total News Records:", len(data))


# Take input directly
news = input("\nEnter News Text: ")

clean_news = clean_text(news)

news_vector = vectorizer.transform([clean_news])

prediction = model.predict(news_vector)[0]

confidence = max(model.predict_proba(news_vector)[0]) * 100

print("\n========== RESULT ==========")
print("Prediction       :", prediction)
print("Confidence Score :", round(confidence, 2), "%")
print("============================")
