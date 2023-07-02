import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from joblib import dump


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def extract_features(df):
    # Extract the necessary features
    feature_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    target_column = "Machine failure"

    # Split the data into X and y
    X = df[feature_columns]
    y = df[target_column]

    return X, y


def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = make_pipeline(StandardScaler(), RandomForestClassifier())
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")


def save_model(model, filename):
    dump(model, filename)


def run_random_forest(filepath, output_model_path):
    # Load the dataframe
    df = load_data(filepath)

    # Create feature and target columns
    X, y = extract_features(df)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # Create a model pipeline and train the model
    model = train_model(X_train, y_train)

    # Evaluate the model with different metrics
    evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, output_model_path)


if __name__ == "__main__":
    run_random_forest("data/ai4i2020.csv", "models/random_forest_model.joblib")
