from typing import Optional, List, Dict
import pandas as pd
import os

class ClassDistribution:
    def __init__(self, data_directory: str):
        """
        Initialize the ClassDistribution with the directory containing datasets.

        Parameters:
        data_directory (str): The directory containing the dataset files.
        """
        self.data_directory = data_directory

    def load_and_process_dataset(self, dataset_path: str, target_column: str) -> Optional[Dict[str, str]]:
        """
        Load and process a single dataset.

        Parameters:
        dataset_path (str): The full path to the dataset file.
        target_column (str): The name of the target column in the dataset.

        Returns:
        Optional[Dict[str, str]]: A dictionary containing the dataset name, number of instances,
                                  number of features, positive class frequency, positive class count,
                                  and negative class count. Returns None if the target column is not found.
        """
        try:
            # Load the dataset
            data: pd.DataFrame = pd.read_csv(dataset_path)
            
            # Ensure the target column exists
            if target_column not in data.columns:
                print(f"Target column {target_column} not found in {os.path.basename(dataset_path)}. Skipping.")
                return None

            # Remove all columns that start with 'class_' except the target column
            columns_to_remove: List[str] = [col for col in data.columns if col.startswith('class_') and col != target_column]
            data: pd.DataFrame = data.drop(columns=columns_to_remove)

            # Calculate metrics
            num_instances: int = len(data)
            num_features: int = len(data.columns) - 1  # Subtract 1 for the target column
            positive_class_count: int = int(data[target_column].sum())
            negative_class_count: int = num_instances - positive_class_count
            positive_class_frequency: float = (positive_class_count / num_instances) * 100

            return {
                "Dataset": os.path.basename(dataset_path).split("_")[0],
                "Number of Instances": str(num_instances),
                "Number of Features": str(num_features),
                "Positive Class Frequency (%)": f"{positive_class_frequency:.2f}".rstrip('0').rstrip('.'),
                "Positive Class Count": str(positive_class_count),
                "Negative Class Count": str(negative_class_count)
            }

        except Exception as e:
            print(f"Error processing {os.path.basename(dataset_path)}: {e}")
            return None

    def process_datasets(self) -> pd.DataFrame:
        """
        Process all datasets in the specified directory and compile their statistics.

        Returns:
        pd.DataFrame: A DataFrame containing the statistics for each dataset.
        """
        dataset_files: List[str] = os.listdir(self.data_directory)
        results: List[Dict[str, str]] = []

        for dataset_file in dataset_files:
            dataset_path: str = os.path.join(self.data_directory, dataset_file)
            dataset_name: str = dataset_file.split("_")[0]  # Extract the dataset name
            target_column: str = f"class_{dataset_name}_w8"

            result: Optional[Dict[str, str]] = self.load_and_process_dataset(dataset_path, target_column)
            if result:
                results.append(result)

        return pd.DataFrame(results)

    def display_dataset_statistics(self, df_results: pd.DataFrame) -> None:
        """
        Display the dataset statistics in a formatted manner.

        Parameters:
        df_results (pd.DataFrame): The DataFrame containing the dataset statistics.
        """
        print("Dataset Class Distribution\n")
        print("This table shows the statistics for each dataset in the data directory.")
        print("These stats can be used to compare the datasets and understand the class distribution.\n")
        
        # Display the DataFrame as a table without the index
        df_results_styled: pd.io.formats.style.Styler = df_results.style.hide(axis='index')
        print(df_results_styled)

        # Print each column individually, no index, with left-aligned text
        for col in df_results.columns:
            print(df_results[col].to_string(index=False))
            print()

    def run(self) -> None:
        """
        Main function to process all datasets and display their statistics.
        """
        df_results: pd.DataFrame = self.process_datasets()
        
        if not df_results.empty:
            self.display_dataset_statistics(df_results)
        else:
            print("No valid datasets found.")

# Usage Example
if __name__ == "__main__":
    data_directory: str = "./data" # Path to the directory containing the datasets
    processor: ClassDistribution = ClassDistribution(data_directory) # Initialize the ClassDistribution
    processor.run() # Process all datasets and display their statistics
