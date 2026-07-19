import pickle
import nltk
import string

from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing function
def clean_text(text):
    text = text.lower()

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


# Menu
while True:
    print("\n===================================")
    print(" AI Based Fake News Detection Tool ")
    print("===================================")
    print("1. Predict News")
    print("2. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":
        news = input("\nEnter News Text:\n")

        clean_news = clean_text(news)

        news_vector = vectorizer.transform([clean_news])

        prediction = model.predict(news_vector)[0]

        confidence = max(model.predict_proba(news_vector)[0]) * 100

        print("\n========== Prediction ==========")
        print("Result           :", prediction)
        print("Confidence Score :", round(confidence, 2), "%")
        print("================================")

    elif choice == "2":
        print("\nThank you for using the AI Based Fake News Detection Tool!")
        break

    else:
        print("\nInvalid choice! Please try again.")
