import mlflow
import mlflow.catboost
import pandas as pd
from catboost import CatBoostRegressor,Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import yaml

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load dataset function (you can customize it for your own dataset)
def load_data(filepath):
    """
    Loads the preprocessed dataset from a CSV file.
    :param filepath: Path to the preprocessed dataset CSV file.
    :return: pandas DataFrame
    """
    return pd.read_csv(filepath)

# Train CatBoost Regressor model
def train_model(train:Pool, val:Pool, X_test,y_test, params):
    """
    Trains the CatBoostRegressor model and returns the trained model.
    :param X_train: Training features
    :param X_test: Testing features
    :param y_train: Training target
    :param y_test: Testing target
    :param params: Hyperparameters for the CatBoost model
    :return: Trained CatBoost model
    """
    # Initialize CatBoost regressor
    model = CatBoostRegressor(**params)
    # Train the model
    model.fit(train, eval_set=val, verbose=False)
    
    # Predict on the test set
    predictions = model.predict(X_test)
    
    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, predictions,squared=False)
    print(f"Root Mean Squared Error: {mse}")
    
    return model, mse

# Save model locally
def save_model_locally(model, save_dir, model_name):
    """
    Saves the trained CatBoost model to a specific directory.
    :param model: Trained CatBoost model
    :param save_dir: Directory where the model will be saved
    :param model_name: The name of the model file
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)
    model.save_model(model_path)
    print(f"Model saved locally at: {model_path}")

# Main function to train model, log with MLflow, and save locally
def run_mlflow_experiment(config):
    """
    Runs the MLflow experiment for training a CatBoostRegressor model and saving the model locally.
    """
    # Load the dataset
    print("Loading data...")
    X_train = load_data(config['data']['data_path']+'X_train.csv')
    X_test = load_data(config['data']['data_path']+'X_test.csv')
    y_train = load_data(config['data']['data_path']+'y_train.csv')
    y_test = load_data(config['data']['data_path']+'y_test.csv')
        
    # Perform train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config['data']['val_size'],
                                                      random_state=config['random_seed'])
    train = Pool(X_train,y_train,cat_features=config['data']['cat_columns'])
    val = Pool(X_val,y_val,cat_features=config['data']['cat_columns'])
    # Set up MLflow tracking
    mlflow.set_experiment("CatBoost Regressor Experiment")
    
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params(config['model'])
        
        # Train the model
        print("Training model...")
        model, mse = train_model(train,val,X_test,y_test, config['model'])
        
        # Log the MSE metric
        mlflow.log_metric("mse", mse)
        
        # Log the model to MLflow
        print("Logging model to MLflow...")
        mlflow.catboost.log_model(model, "catboost_model")
        
        # Save the model locally
        print("Saving model locally...")
        save_model_locally(model, config['data']['model_save'], config['data']['model_file'])
        
        print("Model training, logging, and saving completed!")

# Example usage
if __name__ == "__main__":
    config = load_config('configs\model.yaml')
    # Run the MLflow experiment
    run_mlflow_experiment(config)
