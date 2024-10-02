import os
import yaml
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Load YAML config
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Initialize Kaggle API and download dataset
def download_dataset(config):
    api = KaggleApi()
    api.authenticate()

    # Extract dataset info from the YAML config
    dataset_author = config['dataset']['author']
    dataset_name = config['dataset']['name']
    download_path = config['dataset']['download_path']

    # Ensure the download directory exists
    os.makedirs(download_path, exist_ok=True)

    # Download the dataset from Kaggle
    print(f"Downloading {dataset_name} dataset...")
    api.dataset_download_files(f'{dataset_author}/{dataset_name}', path=download_path, unzip=False)

    # Unzipping the dataset
    zip_file_path = f"{download_path}{dataset_name}.zip"
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)

    print(f"{dataset_name} dataset downloaded and extracted to {download_path}")

# Main execution
if __name__ == "__main__":
    config_file = 'configs\dataset.yaml'  # Path to your YAML config file
    config = load_config(config_file)  # Load config

    download_dataset(config)  # Download dataset based on config