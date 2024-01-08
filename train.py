import logging
import datetime
import pandas as pd
import yaml
from skopt.space import Real, Integer, Categorical
import mlflow
import src.train_utils as tu
import src.log_utils as lu
import argparse


logging.config.fileConfig("config/logs/local.conf")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"config/logs/train_{timestamp}.log"
lu.load_logging_config('config/logs/local.conf', log_filename)
logger = logging.getLogger(__name__)


def main(config_path):
    start_time = datetime.datetime.now() 
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load data
    df = pd.read_csv(config['data_path'])

    try:
        # Preprocess the data
        X_train, X_test, y_train, y_test = tu.preprocess(df, config['target'], config['preprocessing'])
    except Exception as e:
        logger.error("Error in preprocessing data: %s" % e)

    try:
        # Train the model
        best_model = tu.train(X_train, y_train, config['model'], config['tuning'], n_iter=7, experiment_name=config['experiment_name'])
    except Exception as e:
        logger.error("Error in training the model: %s" % e)

    try:
        # Evaluate the model
        _ = tu.evaluate(best_model, X_test, y_test, config['report_save_path'])
    except Exception as e:
        logger.error("Error in calculating metrics: %s" % e)
    
    mlflow.log_artifact(log_filename)

    # End the MLflow run
    mlflow.end_run()

    end_time = datetime.datetime.now()  # Record the end time
    duration = end_time - start_time
    logger.info(f"Total runtime: {duration}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML training pipeline with specified config.")
    parser.add_argument('-c', '--config', dest='config_path', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config_path)