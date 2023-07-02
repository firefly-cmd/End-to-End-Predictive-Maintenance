import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    fbeta_score,
    make_scorer,
    cohen_kappa_score,
    balanced_accuracy_score,
    average_precision_score,
)
from joblib import dump
import optuna
from functools import partial
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


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


def save_model(model, filename):
    dump(model, filename)


def objective(trial, X, y, beta):
    C = trial.suggest_float("C", 1e-10, 1e10, log=True)

    # define solver and penalty together to avoid incompatible combinations
    solver_penalty_choice = trial.suggest_categorical(
        "solver_penalty",
        [
            ("newton-cg", "l2"),
            ("lbfgs", "l2"),
            ("liblinear", "l1"),
            ("liblinear", "l2"),
        ],
    )
    solver = solver_penalty_choice[0]
    penalty = solver_penalty_choice[1]

    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C, solver=solver, penalty=penalty, class_weight=class_weight
        ),
    )

    score = cross_val_score(
        model, X, y, cv=5, scoring=make_scorer(fbeta_score, beta=beta)
    ).mean()
    return score


# Defining a function to compute metrics and save them to a CSV file
def compute_and_save_metrics(y_true, y_pred, beta=1, save_path=""):
    """
    Function to compute metrics and save them in a CSV file

    Parameters:
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    model_name : string
        The name of the model for which metrics are being computed.
    beta : float, default=1
        The strength of recall versus precision in the F-beta score.

    Returns:
    None
    """

    # Computing metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fbeta = fbeta_score(y_true, y_pred, beta=beta)
    kappa = cohen_kappa_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)

    # Storing metrics in a dictionary
    metrics = {}
    metrics["Precision"] = precision
    metrics["Recall"] = recall
    metrics["Fbeta Score"] = fbeta
    metrics["Cohen Kappa"] = kappa
    metrics["Balanced Accuracy"] = balanced_accuracy
    metrics["Average Precision"] = average_precision

    # Converting dictionary to DataFrame
    metrics_df = pd.DataFrame([metrics])

    # Writing DataFrame to CSV file
    file_name = save_path
    metrics_df.to_csv(file_name, index=False)


def calculate_and_save_feature_importances(model, feature_names, save_path):
    # Extract the logistic regression estimator from the pipeline
    log_reg = model.named_steps["logisticregression"]

    # Get the feature importances
    importances = log_reg.coef_[0]

    # Convert importances to percentages
    importances = importances / np.sum(np.abs(importances)) * 100

    # Map the feature importances to the corresponding feature names
    feature_importances = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    )

    # Sort by the absolute value of the importance of the feature
    feature_importances = feature_importances.sort_values(
        "Importance", key=lambda x: x.abs(), ascending=False
    )

    feature_importances.to_csv(save_path, index=False)


def save_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(filename, cm)


# Create new features from the original data
def create_exp2_features(data):
    # Sort the data
    data = data.sort_values("UDI")

    # Create the new features
    power = data["Rotational speed [rpm]"] * data["Torque [Nm]"]
    strain = data["Tool wear [min]"] * data["Torque [Nm]"]
    temp_diff = data["Process temperature [K]"] - data["Air temperature [K]"]
    machine_failure = data["Machine failure"]

    # Combine the new features into a new DataFrame
    new_features = pd.DataFrame(
        {"power": power, "strain": strain, "temp_diff": temp_diff}
    )

    target = pd.DataFrame({"target": machine_failure})

    return new_features, target


def create_exp3_features(data):
    # Create a new dataframe
    data_new = data.copy()

    # Add the new features
    data_new["power"] = data_new["Rotational speed [rpm]"] * data_new["Torque [Nm]"]
    data_new["strain"] = data_new["Tool wear [min]"] * data_new["Torque [Nm]"]
    data_new["temp_diff"] = (
        data_new["Process temperature [K]"] - data_new["Air temperature [K]"]
    )

    feature_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "power",
        "strain",
        "temp_diff",
    ]

    X = data_new[feature_columns]
    y = data_new["Machine failure"]

    return X, y


def logistic_regression_experiment(
    X_train, y_train, X_test, y_test, beta, experiment_dictionary
):
    # Search for the hyperparameters
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, X=X_train, y=y_train, beta=beta), n_trials=100)

    print(
        f"Best trial recall: {study.best_trial.value}, Params: {study.best_trial.params}"
    )

    # Retrain the model with the best parameters and evaluate on the test set
    best_params = study.best_params
    print("BEST PARAMS")
    print(best_params)

    best_hyperparameters = {}
    best_hyperparameters["C"] = best_params["C"]
    best_hyperparameters["solver"] = best_params["solver_penalty"][0]
    best_hyperparameters["penalty"] = best_params["solver_penalty"][1]
    best_hyperparameters["class_weight"] = best_params["class_weight"]

    # Save hyper parameters
    # Converting dictionary to DataFrame
    hyperparametrs_df = pd.DataFrame([best_hyperparameters])

    # Writing DataFrame to CSV file
    file_name = f"results/{experiment_dictionary}/hyperparameters.csv"

    hyperparametrs_df.to_csv(file_name, index=False)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=best_params["C"],
            solver=best_params["solver_penalty"][0],
            penalty=best_params["solver_penalty"][1],
            class_weight=best_params["class_weight"],
        ),
    )

    # Train the model with best hyperparameters
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute and record the evaluation metrics
    compute_and_save_metrics(
        y_true=y_test,
        y_pred=y_pred,
        beta=beta,
        save_path=f"results/{experiment_dictionary}/metrics.csv",
    )

    # Calculate and save the feature importances
    calculate_and_save_feature_importances(
        model,
        feature_names=X_train.columns,
        save_path=f"results/{experiment_dictionary}/feature_importances.csv",
    )

    # Calculate and save the confusion matrix
    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        filename=f"results/{experiment_dictionary}/confusion_matrix.txt",
    )


from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


def perform_kernel_pca(X_train, X_test, n_components=2, kernel="rbf"):
    """
    Apply Kernel PCA on the data

    Parameters:
    X_train: np.array or pd.DataFrame
        The training data
    X_test: np.array or pd.DataFrame
        The test data
    n_components: int, optional (default=2)
        Number of components to keep
    kernel: string, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm

    Returns:
    X_kpca_train: np.array
        Training data transformed using Kernel PCA
    X_kpca_test: np.array
        Test data transformed using Kernel PCA
    """

    # Standardize the features to have mean=0 and variance=1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(
        X_test
    )  # Use the same scaler fit to the training set

    # Perform Kernel PCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    X_kpca_train = kpca.fit_transform(X_train_scaled)
    X_kpca_test = kpca.transform(
        X_test_scaled
    )  # Use the same kpca fit to the training set

    # Converting numpy arrays to dataframes
    X_train_df = pd.DataFrame(
        X_kpca_train,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=X_train.index,
    )
    X_test_df = pd.DataFrame(
        X_kpca_test,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=X_test.index,
    )

    return X_train_df, X_test_df


def run_logistic_regression(filepath, output_model_path):
    ## Initial data loading
    # Load the dataframe
    df = load_data(filepath)

    # Create feature and target columns
    X, y = extract_features(df)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # Initiate the beta value for the f beta score
    # Since the detection of machine failures are more important than the non-machine
    # failure case we can give more importance to the recall than the precision
    # This value should change according to the stakeholders and the cost of
    # machine failures vs cost of maintanence. For ease of use, this value will be set to 2
    # throughout the course of this classification task
    beta = 2

    ## Experiment 1
    # In this experiment raw metrics will be directly used and a baseline will be created.
    logistic_regression_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        beta,
        experiment_dictionary="logistic_regression_experiment1",
    )

    ## Experiment 2
    # In this experiment new extracted features will be used
    # New features will include the power, strain, air temperature difference
    X2, y2 = create_exp2_features(df)

    # Split the data into train and test
    X2_train, X2_test, y2_train, y2_test = split_data(X2, y2, test_size=0.2)

    logistic_regression_experiment(
        X2_train,
        y2_train,
        X2_test,
        y2_test,
        beta,
        experiment_dictionary="logistic_regression_experiment2",
    )

    ## Experiment 3
    # In this experiment we will use both old and new features to try to capture new information
    X3, y3 = create_exp3_features(df)

    # Split the data into train and test
    X3_train, X3_test, y3_train, y3_test = split_data(X3, y3, test_size=0.2)

    logistic_regression_experiment(
        X3_train,
        y3_train,
        X3_test,
        y3_test,
        beta,
        experiment_dictionary="logistic_regression_experiment3",
    )

    ## Experiment 4
    # In this experiment we will use kernel PCA to try to improve the model performance with the raw sensory data
    X4, y4 = extract_features(df)

    # Split the data into train and test
    X4_train, X4_test, y4_train, y4_test = split_data(X4, y4, test_size=0.2)

    # Apply Kernel PCA to both train and test set
    X4_train_pca, X4_test_pca = perform_kernel_pca(
        X4_train, X4_test, n_components=2, kernel="rbf"
    )

    logistic_regression_experiment(
        X4_train_pca,
        y3_train,
        X4_test_pca,
        y3_test,
        beta,
        experiment_dictionary="logistic_regression_experiment4",
    )


if __name__ == "__main__":
    run_logistic_regression(
        "data/ai4i2020.csv", "models/logistic_regression_model.joblib"
    )
