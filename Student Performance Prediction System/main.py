import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv")

# Dataset Information
print("\nDataset Information:")
data.info()

# Check Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove Missing Values
data = data.dropna()

# Dataset Statistics
print("\nDataset Statistics:")
print(data.describe())

# Input and Output
X = data[["attendance", "marks", "study_hours"]]
y = data["final_score"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Accuracy
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

print("\n========== Student Performance Prediction System ==========")
print("Model Accuracy:", round(accuracy, 2))

# User Input
attendance = float(input("\nEnter Attendance (%): "))
marks = float(input("Enter Marks: "))
study_hours = float(input("Enter Study Hours per Day: "))

# Prediction
input_data = pd.DataFrame({
    "attendance": [attendance],
    "marks": [marks],
    "study_hours": [study_hours]
})

result = model.predict(input_data)

print("\nPredicted Final Score:", round(result[0], 2))

# ==========================================
# Student Performance Analysis
# ==========================================

plt.figure(figsize=(8,5))

# Scatter Plot
plt.scatter(
    data["study_hours"],
    data["final_score"],
    color="blue",
    s=80,
    label="Students"
)

# Trend Line
plt.plot(
    data["study_hours"],
    model.predict(data[["attendance", "marks", "study_hours"]]),
    color="red",
    linewidth=2,
    label="Trend"
)

plt.title("Student Performance Analysis")
plt.xlabel("Study Hours per Day")
plt.ylabel("Final Score")
plt.grid(True)
plt.legend()

plt.show()
