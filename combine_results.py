import pandas as pd
import os


if __name__ == "__main__":
    # Define the root directory where all experiments are stored
    root_dir = "ml/binary_classification/results"

    # Store each experiment DataFrame in a list
    dfs = []

    # Iterate through each folder inside the root directory
    for experiment_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, experiment_folder)

        # Check if the current item is a folder and contains the metrics.csv file
        if os.path.isdir(folder_path) and "metrics.csv" in os.listdir(folder_path):
            # Define the path to the metrics.csv file
            csv_path = os.path.join(folder_path, "metrics.csv")

            # Read the CSV file into a DataFrame
            experiment_data = pd.read_csv(csv_path)

            # Use the folder name as the index for this DataFrame
            experiment_data["Experiment_Name"] = experiment_folder
            experiment_data.set_index("Experiment_Name", inplace=True)

            # Add the DataFrame to the list
            dfs.append(experiment_data)

    # Combine all DataFrames in the list into a single DataFrame
    all_results = pd.concat(dfs)

    # Display the combined results
    print(all_results)

    # Save the overall results inside a dataframe
    all_results.to_csv("overall_results/binary_classification.csv", header=True)
