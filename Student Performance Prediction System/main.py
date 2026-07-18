import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Display first few records
print("\n===== Student Performance Dataset =====")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Input and Output
X = data[["Attendance", "Study_Hours", "Internal_Marks"]]
y = data["Final_Performance"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

# Evaluation
print("\n===== Model Performance =====")
print("R2 Score:", round(r2_score(y_test, y_pred), 2))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))

# User Prediction
print("\n===== Student Performance Prediction =====")
attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours per day: "))
internal_marks = float(input("Enter Internal Marks: "))

# Create DataFrame to avoid feature name warning
new_student = pd.DataFrame({
    "Attendance": [attendance],
    "Study_Hours": [study_hours],
    "Internal_Marks": [internal_marks]
})

prediction = model.predict(new_student)

print("\nPredicted Final Performance:", round(prediction[0], 2))

# Visualization 1: Actual vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Performance")
plt.ylabel("Predicted Performance")
plt.title("Actual vs Predicted Performance")
plt.grid(True)
plt.show()

# Visualization 2: Attendance vs Final Performance
plt.figure(figsize=(6, 5))
plt.scatter(data["Attendance"], data["Final_Performance"])
plt.xlabel("Attendance")
plt.ylabel("Final Performance")
plt.title("Attendance vs Final Performance")
plt.grid(True)
plt.show()

# Visualization 3: Study Hours vs Final Performance
plt.figure(figsize=(6, 5))
plt.scatter(data["Study_Hours"], data["Final_Performance"])
plt.xlabel("Study Hours")
plt.ylabel("Final Performance")
plt.title("Study Hours vs Final Performance")
plt.grid(True)
plt.show()

# Visualization 4: Internal Marks vs Final Performance
plt.figure(figsize=(6, 5))
plt.scatter(data["Internal_Marks"], data["Final_Performance"])
plt.xlabel("Internal Marks")
plt.ylabel("Final Performance")
plt.title("Internal Marks vs Final Performance")
plt.grid(True)
plt.show()
