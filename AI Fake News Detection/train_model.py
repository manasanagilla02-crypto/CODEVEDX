import pandas as pd
import pickle
import string
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download("stopwords")

# Load dataset
data = pd.read_csv("news.csv")
data.columns = data.columns.str.strip()

# Select required columns
data = data[["text", "label"]]

# Remove missing values
data = data.dropna()

# Stopwords
stop_words = set(stopwords.words("english"))

# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Clean text
data["text"] = data["text"].apply(clean_text)

# Features and labels
X = data["text"]
y = data["label"]

# Convert text into numerical values
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, prediction)
accuracy_percentage = accuracy * 100

# Print accuracy
print("Model Accuracy:", round(accuracy_percentage, 2), "%")

# Accuracy Graph
plt.figure(figsize=(5, 5))
plt.bar(["Accuracy"], [accuracy_percentage], width=0.5)

plt.title("Fake News Detection Model Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.show()

# Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("model.pkl created successfully")
print("vectorizer.pkl created successfully")
