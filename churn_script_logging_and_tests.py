import os
import logging
from typing import Callable

from core.settings import settings
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
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda: Callable) -> None:
    """
    test perform eda function
    """
    df = cl.import_data(settings.DATA_PATH / 'bank_data.csv')
    OUT_FILES = {'churn_distribution.png', 'customer_age_distribution.png',
                 'heatmap.png', 'marital_status_distribution.png',
                 'total_transaction_distribution.png'}

    try:
        perform_eda(df)
        logging.info("Testing import_data: SUCCESS")
    except KeyError as err:
        logging.error(
            f"Testing perform_eda: Column {err.args[0]} not"
            f" found in input schema.")
        raise err

    files_created = set(os.listdir(settings.EDA_PATH))
    missing_files = OUT_FILES - files_created

    if len(missing_files)>0:
        logging.error(
            f"Testing perform_eda files: Files: {missing_files} not"
            f" found at output path.")
        raise FileNotFoundError

    else:
        logging.info("Testing perform_eda files: SUCCESS")


def test_encoder_helper(encoder_helper: Callable) -> None:
    """
    test encoder helper
    """
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
    ]

    df = cl.import_data(settings.DATA_PATH / 'bank_data.csv')

    df = encoder_helper(
        df=df,
        category_lst=category_lst,
        response='Churn'
    )

    for col in category_lst:

        df_test = df.\
            groupby(by=col).\
            agg(
                test_col=('Churn', 'mean'),
                ref_val=(col + '_' + 'Churn', 'mean')
        )

        try:
            assert all(df_test['test_col'] == df_test['ref_val'])
            logging.info(f"Testing test_encoder_helper: SUCCESS "
                         f"for '{col}' column.")

        except AssertionError as e:
            logging.error(
                f"Testing test_encoder_helper: values for column:"
                f" {col} do not match with Churn mean by category.")


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """


def test_train_models(train_models):
    """
    test train_models
    """


if __name__ == "__main__":

    test_import(cl.import_data)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
