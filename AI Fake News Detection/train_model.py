import pandas as pd
import pickle
import string
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download("stopwords")

# Load dataset
data = pd.read_csv("news.csv")

# Check dataset columns
print(data.head())
print(data.columns)

# Select text and label columns
data = data[["text", "label"]]

# Remove missing values
data = data.dropna()

# Text preprocessing
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Apply cleaning
data["text"] = data["text"].apply(clean_text)

# Input and output
X = data["text"]
y = data["label"]

# Convert text into numerical values
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Model Accuracy:", accuracy)

# Save trained model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("model.pkl created successfully")
print("vectorizer.pkl created successfully")
