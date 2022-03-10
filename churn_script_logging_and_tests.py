import os
import logging
from typing import Callable

from math import isclose
from sklearn.metrics import f1_score

from core.settings import settings, config
from core.ml.utils import load_model

import churn_library as cl

logging.basicConfig(
    filename=settings.LOGS_PATH / 'churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data: Callable) -> None:
    """
    test data import - this example is completed for you to assist with
    the other test functions
    """
    try:
        bank_data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert bank_data.shape[0] > 0
        assert bank_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda: Callable) -> None:
    """
    test perform eda function
    """
    bank_data = cl.import_data(settings.DATA_PATH / 'bank_data.csv')
    OUT_FILES = set(getattr(config, 'test_parameters').get('eda_files', {}))

    try:
        perform_eda(bank_data)
        logging.info("Testing import_data: SUCCESS")
    except KeyError as err:
        logging.error(
            f"Testing perform_eda: Column {err.args[0]} not"
            f" found in input schema.")
        raise err

    files_created = set(os.listdir(settings.EDA_PATH))
    missing_files = OUT_FILES - files_created

    if len(missing_files) > 0:
        logging.error(
            f"Testing perform_eda: Files: {missing_files} not"
            f" found at {settings.EDA_PATH}.")
        raise FileNotFoundError

    else:
        logging.info("Testing perform_eda files: SUCCESS")


def test_encoder_helper(encoder_helper: Callable) -> None:
    """
    test encoder helper
    """
    category_lst = getattr(config, 'test_parameters').get('category_vars', [])

    bank_data = cl.import_data(settings.DATA_PATH / 'bank_data.csv')

    bank_data = encoder_helper(
        bank_data=bank_data,
        category_lst=category_lst,
        response='Churn'
    )

    for col in category_lst:

        bank_data_test = bank_data. \
            groupby(by=col). \
            agg(
            test_col=('Churn', 'mean'),
            ref_val=(col + '_' + 'Churn', 'mean')
        )

        try:
            assert all(bank_data_test['test_col'] == bank_data_test['ref_val'])
            logging.info(f"Testing test_encoder_helper: SUCCESS "
                         f"for '{col}' column.")

        except AssertionError as e:
            logging.error(
                f"Testing test_encoder_helper: values for column:"
                f" {col} do not match with Churn mean by category.")


def test_perform_feature_engineering(
        perform_feature_engineering: Callable
) -> None:
    """
    test perform_feature_engineering
    """
    category_lst = getattr(config, 'test_parameters').get('category_vars', [])

    bank_data = cl.import_data(settings.DATA_PATH / 'bank_data.csv')

    bank_data = cl.encoder_helper(
        bank_data=bank_data,
        category_lst=category_lst,
        response='Churn'
    )

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        bank_data=bank_data,
        response='Churn'
    )

    validation_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    for key, val in validation_dict.items():

        if 'train' in key:
            percentage = 0.7
        else:
            percentage = 0.3

        try:
            assert isclose(val.shape[0], bank_data.shape[0] * percentage, abs_tol=1)
            logging.info(f"Testing test_perform_feature_engineering: SUCCESS "
                         f"{key} has {percentage * 100}% of records.")

        except AssertionError as e:
            logging.error(
                f"Testing test_perform_feature_engineering: {key} does not "
                f"have {percentage * 100}% of records.")


def test_train_models(train_models: Callable) -> None:
    """
    test train_models
    """
    category_lst = getattr(config, 'test_parameters').get('category_vars', [])
    model_files = set(getattr(config, 'test_parameters').get('models_files', {}))
    results_files = set(getattr(config, 'test_parameters').get('results_files', {}))

    bank_data = cl.import_data(settings.DATA_PATH / 'bank_data.csv')

    bank_data = cl.encoder_helper(
        bank_data=bank_data,
        category_lst=category_lst,
        response='Churn'
    )

    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
        bank_data=bank_data,
        response='Churn'
    )

    train_models(X_train, X_test, y_train, y_test)

    model_files_created = set(os.listdir(settings.MODELS_PATH))
    missing_files = model_files - model_files_created

    if len(missing_files) > 0:
        logging.error(
            f"Testing train_models: Model(s) {missing_files} not"
            f" found at {settings.MODELS_PATH}.")
        raise FileNotFoundError

    else:
        for file in model_files:
            logging.info(f"Testing train_models: SUCCESS {file} created")

    results_files_created = set(os.listdir(settings.RESULTS_PATH))
    missing_files = results_files - results_files_created

    if len(missing_files) > 0:
        logging.error(
            f"Testing train_models: Results files(s) {missing_files} not"
            f" found at {settings.RESULTS_PATH}.")
        raise FileNotFoundError

    else:
        for file in results_files:
            logging.info(f"Testing train_models: SUCCESS {file} created")

    for model_file in model_files:

        model = load_model(
            model_name=model_file.split('.')[0]
        )

        score = f1_score(
            y_true=y_test,
            y_pred=model.predict(X_test)
        )

        try:
            assert score > 0.5
            logging.info(f"Testing train_models: SUCCESS {model_file} has "
                         f"F1-score greater than 0.5")
        except AssertionError:
            logging.error(
                f"Testing train_models: {model_file} has F1-score lower than 0.5")



if __name__ == "__main__":
    test_import(cl.import_data)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)
