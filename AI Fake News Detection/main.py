import pickle

# Load trained model and vectorizer
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

print("=" * 50)
print(" AI Based Fake News Detection Tool ")
print("=" * 50)

while True:
    print("\nMenu")
    print("1. Predict News")
    print("2. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        news = input("\nEnter News Text:\n")

        # Convert text into TF-IDF features
        news_vector = vectorizer.transform([news])

        # Prediction
        prediction = model.predict(news_vector)[0]

        # Confidence score
        confidence = max(model.predict_proba(news_vector)[0]) * 100

        print("\nPrediction:", prediction)
        print(f"Confidence Score: {confidence:.2f}%")

    elif choice == "2":
        print("\nThank you for using the AI Based Fake News Detection Tool.")
        break

    else:
        print("\nInvalid choice. Please try again.")
