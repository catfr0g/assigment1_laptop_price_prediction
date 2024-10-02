import pandas as pd
from sklearn.model_selection import train_test_split
import os
from dataloading import load_config
# Load preprocessed dataset
def load_preprocessed_data(filepath):
    """
    Loads the preprocessed dataset from the specified file path.
    :param filepath: Path to the preprocessed dataset file (CSV format).
    :return: pandas DataFrame
    """
    return pd.read_csv(filepath)

# Split the dataset into train and test sets
def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the dataset into train and test sets.
    :param df: Preprocessed pandas DataFrame
    :param target_column: The column name to use as the target (label).
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Seed used by the random number generator.
    :return: X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Save the split datasets
def save_splitted_data(X_train, X_test, y_train, y_test, save_dir):
    """
    Saves the split datasets to CSV files.
    :param X_train: Training features DataFrame
    :param X_test: Testing features DataFrame
    :param y_train: Training target DataFrame
    :param y_test: Testing target DataFrame
    :param save_dir: Directory to save the split datasets
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save train and test datasets
    X_train.to_csv(os.path.join(save_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(save_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(save_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(save_dir, 'y_test.csv'), index=False)
    
    print(f"Train-test data saved to {save_dir}")

# Main function to execute the loading, splitting, and saving process
def prepare_train_test_split(config):
    """
    Main pipeline to load preprocessed data, perform a train-test split, and save the split data.
    :param preprocessed_data_path: Path to the preprocessed dataset CSV file.
    :param target_column: The column name to use as the target (label).
    :param save_dir: Directory to save the split datasets.
    :param test_size: Proportion of the dataset to include in the test split.
    """
    preprocessed_data_path = config['dataset']['download_path']+config['dataset']['file_name']+'_prepared'+'.csv'  # Path to your raw dataset
    save_dir = config['dataset']['download_path']  # Directory to save the processed data
    target_column = config['dataset']['label']
    test_size = config['dataset']['test_size']
    # Step 1: Load preprocessed data
    print("Loading preprocessed dataset...")
    preprocessed_data = load_preprocessed_data(preprocessed_data_path)
    print("Dataset loaded successfully!")

    # Step 2: Perform train-test split
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(preprocessed_data, target_column, test_size)
    print("Data split completed!")

    # Step 3: Save the split data
    print("Saving the split datasets...")
    save_splitted_data(X_train, X_test, y_train, y_test, save_dir)
    print("Split datasets saved successfully!")

# Example usage
if __name__ == "__main__":
    config_file = 'configs\dataset.yaml'  # Path to your YAML config file
    config = load_config(config_file)    # 20% of data will be used for testing
    
    # Run the train-test split pipeline
    prepare_train_test_split(config)
