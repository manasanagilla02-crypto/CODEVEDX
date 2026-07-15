import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Input and Output
X = data[["Attendance", "Study_Hours", "Internal_Marks"]]
y = data["Final_Performance"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
print("----- Model Evaluation -----")
print("R2 Score:", round(r2_score(y_test, y_pred), 2))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))

# User input
print("\n----- Student Performance Prediction -----")
attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours per day: "))
internal_marks = float(input("Enter Internal Marks: "))

prediction = model.predict([[attendance, study_hours, internal_marks]])

print(f"\nPredicted Final Performance: {prediction[0]:.2f}")

# Visualization
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Performance")
plt.ylabel("Predicted Performance")
plt.title("Actual vs Predicted Performance")
plt.show()
