import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
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

# Import dataset creation functions
from experiment_dataset_creators import (
    create_experiment1_dataset,
    create_experiment2_dataset,
    create_experiment3_dataset,
    create_experiment4_dataset,
)


def load_data(filepath):
    return pd.read_csv(filepath)


def save_model(model, path):
    dump(model, path)


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


def save_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(filename, cm)


def objective(trial, X, y, beta):
    n_estimators = trial.suggest_int("n_estimators", 2, 150)
    max_depth = trial.suggest_int("max_depth", 1, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    subsample = trial.suggest_float("subsample", 0.1, 1)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1)
    gamma = trial.suggest_float("gamma", 0, 1)
    scale_pos_weight = trial.suggest_float("scale_pos_weight", 1, 100)

    model = make_pipeline(
        StandardScaler(),
        XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,  # Avoids warning, ensure you have latest xgboost
            eval_metric="logloss",  # Avoids warning, ensure you have latest xgboost
        ),
    )

    score = cross_val_score(
        model, X, y, cv=5, scoring=make_scorer(fbeta_score, beta=beta)
    ).mean()
    return score


def xgboost_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    beta,
    experiment_dictionary,
    model_directory,
    experiment_number,
):
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, X=X_train, y=y_train, beta=beta), n_trials=100)

    print(
        f"Best trial recall: {study.best_trial.value}, Params: {study.best_trial.params}"
    )

    best_params = study.best_params
    model = make_pipeline(
        StandardScaler(),
        XGBClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            gamma=best_params["gamma"],
            scale_pos_weight=best_params["scale_pos_weight"],
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    compute_and_save_metrics(
        y_true=y_test,
        y_pred=y_pred,
        beta=beta,
        save_path=f"results/{experiment_dictionary}/metrics.csv",
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        filename=f"results/{experiment_dictionary}/confusion_matrix.txt",
    )

    # Save the best model
    model_save_filepath = f"models/{model_directory}/{experiment_number}.joblib"

    save_model(model, model_save_filepath)


def run_xgboost(filepath):
    df = load_data(filepath)
    beta = 2

    # Create datasets for all the experiments
    X1_train, X1_test, y1_train, y1_test = create_experiment1_dataset(df)
    X2_train, X2_test, y2_train, y2_test = create_experiment2_dataset(df)
    X3_train, X3_test, y3_train, y3_test = create_experiment3_dataset(df)
    X4_train, X4_test, y4_train, y4_test = create_experiment4_dataset(df)

    # Experiment 1
    xgboost_experiment(
        X1_train,
        y1_train,
        X1_test,
        y1_test,
        beta,
        experiment_dictionary="xgboost_experiment1",
        model_directory="xgboost",
        experiment_number="1",
    )

    # Experiment 2
    xgboost_experiment(
        X2_train,
        y2_train,
        X2_test,
        y2_test,
        beta,
        experiment_dictionary="xgboost_experiment2",
        model_directory="xgboost",
        experiment_number="2",
    )

    # Experiment 3
    xgboost_experiment(
        X3_train,
        y3_train,
        X3_test,
        y3_test,
        beta,
        experiment_dictionary="xgboost_experiment3",
        model_directory="xgboost",
        experiment_number="3",
    )

    # Experiment 4
    xgboost_experiment(
        X4_train,
        y4_train,
        X4_test,
        y4_test,
        beta,
        experiment_dictionary="xgboost_experiment4",
        model_directory="xgboost",
        experiment_number="4",
    )


if __name__ == "__main__":
    run_xgboost("data/ai4i2020.csv")
