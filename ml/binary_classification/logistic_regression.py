import pandas as pd
from sklearn.model_selection import cross_val_score
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
from sklearn.preprocessing import StandardScaler

# Import dataset creation functions
from experiment_dataset_creators import (
    create_experiment1_dataset,
    create_experiment2_dataset,
    create_experiment3_dataset,
    create_experiment4_dataset,
)


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


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


def logistic_regression_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    beta,
    experiment_dictionary,
    model_directory,
    experiment_number,
):
    # Search for the hyperparameters
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, X=X_train, y=y_train, beta=beta), n_trials=100)

    print(
        f"Best trial recall: {study.best_trial.value}, Params: {study.best_trial.params}"
    )

    # Retrain the model with the best parameters and evaluate on the test set
    best_params = study.best_params

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

    # Save the best model
    model_save_filepath = f"models/{model_directory}/{experiment_number}.joblib"

    save_model(model, model_save_filepath)


def run_logistic_regression(filepath):
    ## Initial data loading
    # Load the dataframe
    df = load_data(filepath)

    # Initiate the beta value for the f beta score
    # Since the detection of machine failures are more important than the non-machine
    # failure case we can give more importance to the recall than the precision
    # This value should change according to the stakeholders and the cost of
    # machine failures vs cost of maintanence. For ease of use, this value will be set to 2
    # throughout the course of this classification task
    beta = 2

    # Create experiment 1 features
    X_train, X_test, y_train, y_test = create_experiment1_dataset(df)

    ## Experiment 1
    # In this experiment raw metrics will be directly used and a baseline will be created.
    logistic_regression_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        beta,
        experiment_dictionary="logistic_regression_experiment1",
        model_directory="logistic_regression",
        experiment_number="1",
    )

    # ## Experiment 2
    # # In this experiment new extracted features will be used
    # # New features will include the power, strain, air temperature difference

    # Create experiment 2 features
    X2_train, X2_test, y2_train, y2_test = create_experiment2_dataset(df)

    logistic_regression_experiment(
        X2_train,
        y2_train,
        X2_test,
        y2_test,
        beta,
        experiment_dictionary="logistic_regression_experiment2",
        model_directory="logistic_regression",
        experiment_number="2",
    )

    # ## Experiment 3
    # # In this experiment we will use both old and new features to try to capture new information
    X3_train, X3_test, y3_train, y3_test = create_experiment3_dataset(df)

    logistic_regression_experiment(
        X3_train,
        y3_train,
        X3_test,
        y3_test,
        beta,
        experiment_dictionary="logistic_regression_experiment3",
        model_directory="logistic_regression",
        experiment_number="3",
    )

    # ## Experiment 4
    # # In this experiment we will use kernel PCA to try to improve the model performance with the raw sensory data
    X4_train, X4_test, y4_train, y4_test = create_experiment4_dataset(df)

    logistic_regression_experiment(
        X4_train,
        y4_train,
        X4_test,
        y4_test,
        beta,
        experiment_dictionary="logistic_regression_experiment4",
        model_directory="logistic_regression",
        experiment_number="4",
    )


if __name__ == "__main__":
    run_logistic_regression("data/ai4i2020.csv")
