"""
Student Performance Prediction System using Decision Tree Classifier.
Predicts Pass/Fail from the supplied student performance dataset.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_and_preprocess_data():
    data_path = Path(__file__).resolve().parent / "Student_Performance_DT.csv"
    df = pd.read_csv(data_path)

    features = [
        "Study_Hours", "Attendance", "Previous_Score",
        "Assignments_Completed", "Sleep_Hours", "Participation",
        "Internet_Access", "Extracurricular"
    ]
    target = "Final_Result"

    missing = [col for col in features + [target] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    data = df[features + [target]].copy()
    for col in ["Internet_Access", "Extracurricular"]:
        data[col] = data[col].map({"Yes": 1, "No": 0})

    X = data[features]
    y = data[target]

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def main():
    print("Loading Student Performance dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    print(f"\nModel Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    sample_student = pd.DataFrame([{
        "Study_Hours": 6.0,
        "Attendance": 85.0,
        "Previous_Score": 75.0,
        "Assignments_Completed": 8,
        "Sleep_Hours": 7.0,
        "Participation": 7,
        "Internet_Access": 1,
        "Extracurricular": 1
    }])
    print("Example student predicted result:", model.predict(sample_student)[0])


if __name__ == "__main__":
    main()
