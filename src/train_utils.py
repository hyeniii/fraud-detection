import os
import logging
import numpy as np
import pandas as pd
from typing import Callable, Any, Tuple

from sklearn.model_selection import train_test_split

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, accuracy_score, classification_report, roc_auc_score
import mlflow
import mlflow.sklearn
import importlib
import joblib

SEED = 42 

logger = logging.getLogger(__name__)

def preprocess(df:pd.DataFrame, 
               target:str, 
               config: dict)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocesses the given DataFrame by splitting into train and test sets,
    applying a specified sampling technique to the training set, and then scaling the features.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target (str): The name of the target variable.
    sampler (Callable, optional): The sampling class to be used. Default is RandomOverSampler.
    scaler (Callable, optional): The scaling class to be used. Default is StandardScaler.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Processed train and test data.
    """
    df = pd.DataFrame(df)
    
    # Split data into features and target
    X = df.drop(target, axis=1)
    y = df[target]
    logger.info("Number of features: %s", len(X.columns))
    logger.info("Number of rows: %s", len(X))

    # 80-20 split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    logger.info("Train test shape: Train: %s, Test: %s" % (X_train.shape, X_test.shape))
    print(config['sampler'])
    if config['sampler']['class']:
        # oversample, undersample or use SMOTE
        sampler_module = importlib.import_module(config['sampler']['module'])
        sampler = getattr(sampler_module, config['sampler']['class'])
        sample = sampler(random_state=SEED)
        X_train, y_train = sample.fit_resample(X_train, y_train)
        logger.info('Train target count after sampling: %s' % (y_train.value_counts().to_string()))

    if config['scaler']['class']:
        # scale data
        scaler_module = importlib.import_module(config['scaler']['module'])
        scaler = getattr(scaler_module, config['scaler']['class'])
        scale = scaler()
        X_train = scale.fit_transform(X_train)
        X_test = scale.transform(X_test)
        logger.info("Scaled data using %s" % scaler.__name__)

    logger.info("Completed preprocessing")

    return X_train, X_test, y_train, y_test

def create_hyperparam_space(params_config):
    param_space = {}
    for param, values in params_config.items():
        if isinstance(values[0], int):
            param_space[param] = Integer(values[0], values[1])
        elif isinstance(values[0], float):
            param_space[param] = Real(values[0], values[1])
        elif isinstance(values[0], str):
            param_space[param] = Categorical(values)
        # Add other types as necessary
    return param_space

def instantiate_model(model_config, tune=False):
    module = importlib.import_module(model_config['module'])
    model_class = getattr(module, model_config['class'])
    if tune:
        return model_class(**model_config['params'])
    return model_class()

def save_model(model, model_filename):
    # Extract directory from the model filename
    model_directory = os.path.dirname(model_filename)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        logger.info(f"Created save directory {model_directory}")

    # Save the model
    joblib.dump(model, model_filename)
    logger.info(f"Model saved successfully at {model_filename}")

def train(X_train: pd.DataFrame,
          y_train: pd.Series,
          model_config: dict,
          tune: bool,
          n_iter: int = 10,
          experiment_name: str = "Default Experiment") -> Any:
    """Trains a machine learning model using Bayesian optimization.

    Parameters:
    X_train (pd.DataFrame): Training feature data.
    y_train (pd.Series): Training target data.
    model (Any): The machine learning model to be trained.
    param_space (dict): Hyperparameter space for Bayesian optimization.
    n_iter (int): Number of iterations for Bayesian search.
    experiment_name (str): The name of the MLflow experiment.

    Returns:
    Any: The best model from hyperparameter tuning.
    """
    logger.info("Instantiating MlFlow...")
    mlflow.start_run(run_name=experiment_name)

    if tune:
        try:
            # Define hyperparameter space
            param_space = create_hyperparam_space(model_config['params'])
        except Exception as e:
            logger.error("Error in creating hyperparameter search space: ", e)
        model = instantiate_model(model_config)
        logger.info("Hyperparameter tuning %s..." % model_config['class'])
        best_model = tune_model(X_train, y_train, model, param_space, n_iter, model_config['scoring_metric'])
    else:
        best_model = instantiate_model(model_config, True)
        logger.info("Training %s with given parameters..." % model_config['class'])
        best_model.fit(X_train, y_train)
        mlflow.log_params(model_config['params'])

    logger.info("Successfully trained model")
    save_model(best_model, model_config['save_path'])

    # Return the best model
    return best_model

def tune_model(X_train, y_train, model, params, n_iter, scoring_metric):
    bayes_cv_tuner = BayesSearchCV(estimator=model,
                                   search_spaces=params,
                                   n_iter = n_iter,
                                   cv = 3,
                                   n_jobs=-1,
                                   verbose=1,
                                   scoring=scoring_metric,
                                   random_state=SEED,)
    logger.info("Hyperparameter tuning %s with scoring metric %s..." % (model, scoring_metric))
    np.int = int
    bayes_cv_tuner.fit(X_train, y_train)

    best_model = bayes_cv_tuner.best_estimator_
    logger.info("Best model: %s" % best_model)
    best_params = bayes_cv_tuner.best_params_
    logger.info("Best parameters: %s" % best_params)

    # Log parameters, and model to MLflow
    mlflow.log_params(best_params)
    mlflow.sklearn.log_model(best_model, "model")
    return best_model


def evaluate(model: Any,
             X_test: pd.DataFrame,
             y_test: pd.Series,
             save_path: str) -> dict:
    """Evaluates a trained machine learning model.

    Parameters:
    model (Any): The trained machine learning model.
    X_test (pd.DataFrame): Test feature data.
    y_test (pd.Series): Test target data.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    predictions = model.predict(X_test)

    probas = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probas)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, probas)
    f1_score = f1_score()

    accuracy = accuracy_score(y_test, predictions)
    logger.info("Accuracy of the model: %s" % accuracy)

    report = classification_report(y_test, predictions, output_dict=True)
    logger.info("Classification report: %s" % report)

    report_dir = os.path.dirname(save_path)
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(save_path)
    return {
        'roc_curve': (fpr, tpr),
        'roc_auc': roc_auc,
        'precision_recall_curve': (precision, recall),
        'confusion_matrix': confusion_matrix(y_test, model.predict(X_test)),
        'classification_report': classification_report(y_test, model.predict(X_test), output_dict=True)
        # Add other metrics as needed
    }