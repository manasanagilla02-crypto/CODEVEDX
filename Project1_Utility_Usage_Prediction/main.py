import pandas as pd
import csv
import pickle
from sklearn.linear_model import LinearRegression

while True:
    print("\n===== UTILITY USAGE PREDICTION TOOL =====")
    print("1. Add Usage Data")
    print("2. Train Model")
    print("3. Predict Usage")
    print("4. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        try:
            temp = float(input("Enter Temperature: "))
            users = int(input("Enter Number of Users: "))
            usage = float(input("Enter Utility Usage: "))

            with open("usage_data.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([temp, users, usage])

            print("Data added successfully!")

        except ValueError:
            print("Invalid input!")

    elif choice == "2":
        try:
            data = pd.read_csv("usage_data.csv")

            X = data[["Temperature", "Users"]]
            y = data["Usage"]

            model = LinearRegression()
            model.fit(X, y)

            with open("model.pkl", "wb") as file:
                pickle.dump(model, file)

            print("Model trained successfully!")

        except Exception as e:
            print("Error:", e)

    elif choice == "3":
        try:
            with open("model.pkl", "rb") as file:
                model = pickle.load(file)

            temp = float(input("Enter Temperature: "))
            users = int(input("Enter Number of Users: "))

            prediction = model.predict([[temp, users]])

            print("Predicted Utility Usage =", round(prediction[0], 2))

        except FileNotFoundError:
            print("Please train the model first!")

        except ValueError:
            print("Invalid input!")

    elif choice == "4":
        print("Thank You!")
        break

    else:
        print("Invalid Choice")
