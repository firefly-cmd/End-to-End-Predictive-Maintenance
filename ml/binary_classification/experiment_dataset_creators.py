import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA


# Cache the data in order to not load over and over again
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv("data/ai4i2020.csv")


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


# Experiment 1 is using only raw data
def create_experiment1_dataset(raw_data):
    # Extract the features and target variable
    X, y = extract_features(raw_data)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test


# Experiment 2 is creating power, strain and temp difference from the raw data
def create_experiment2_dataset(raw_data):
    # Sort the data
    data = raw_data.sort_values("UDI")

    # Create the new features
    power = data["Rotational speed [rpm]"] * data["Torque [Nm]"]
    strain = data["Tool wear [min]"] * data["Torque [Nm]"]
    temp_diff = data["Process temperature [K]"] - data["Air temperature [K]"]
    machine_failure = data["Machine failure"]

    # Combine the new features into a new DataFrame
    X = pd.DataFrame({"power": power, "strain": strain, "temp_diff": temp_diff})

    y = pd.DataFrame({"target": machine_failure})

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test


# Experiment 3 combines the both raw data and the produced data
def create_experiment3_dataset(raw_data):
    # Create a new dataframe
    data_new = raw_data.copy()

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

    # Split the data into train and test
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test


# Experiment 4 uses kernel pca to the raw data
def create_experiment4_dataset(raw_data):
    X, y = extract_features(raw_data)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # Apply Kernel PCA to both train and test set
    X_train_pca, X_test_pca = perform_kernel_pca(
        X_train, X_test, n_components=2, kernel="rbf"
    )

    return X_train_pca, X_test_pca, y_train, y_test
