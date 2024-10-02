import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from dataloading import load_config
# Load dataset
def load_dataset(filepath):
    """
    Loads the raw dataset from the specified file path.
    :param filepath: Path to the raw dataset file (CSV format).
    :return: pandas DataFrame
    """
    return pd.read_csv(filepath)

# Data preprocessing function
def preprocess_data(df:DataFrame,config):
    """
    Preprocess the dataset: handle missing values, encode categorical features, and scale numerical data.
    :param df: Raw pandas DataFrame
    :return: Preprocessed pandas DataFrame
    """
    #df.fillna(df.median(), inplace=True)
    df.drop(config['dataset']['drop_cols'],axis=1,inplace=True)
    
    return df  # Return DataFrame along with encoders and scaler

# Save the prepared dataset
def save_prepared_data(df, save_dir, filename):
    """
    Saves the processed DataFrame to a CSV file.
    :param df: Processed pandas DataFrame
    :param save_dir: Directory to save the prepared dataset
    :param filename: Name of the output file
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    df.to_csv(save_path, index=False)
    print(f"Prepared data saved to {save_path}")

# Main function to execute the data preparation pipeline
def prepare_data_pipeline(config):
    """
    Main pipeline to load, preprocess, and save the dataset.
    :param raw_data_path: Path to the raw dataset CSV file.
    :param save_dir: Directory to save the prepared data.
    :param prepared_filename: Filename for the prepared data.
    """
    raw_data_path = config['dataset']['download_path']+config['dataset']['file_name']+'.csv'  # Path to your raw dataset
    save_dir = config['dataset']['download_path']  # Directory to save the processed data
    prepared_filename = config['dataset']['file_name']+'_prepared'+'.csv'  # Name of the output file
    # Step 1: Load raw data
    print("Loading dataset...")
    raw_data = load_dataset(raw_data_path)
    print("Dataset loaded successfully!")

    # Step 2: Preprocess the data
    print("Preprocessing data...")
    prepared_data = preprocess_data(raw_data,config)
    print("Data preprocessing completed!")

    # Step 3: Save the prepared data
    print("Saving prepared data...")
    save_prepared_data(prepared_data, save_dir, prepared_filename)
    print("Data saved successfully!")

# Example usage
if __name__ == "__main__":
    config_file = 'configs\dataset.yaml'  # Path to your YAML config file
    config = load_config(config_file)    
    # Run the data preparation pipeline
    prepare_data_pipeline(config)
