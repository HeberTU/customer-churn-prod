# library doc string


# import libraries
from typing import List, Optional, Union, Tuple

import numpy as np

import pandas as pd
import pandera as pa
import seaborn as sns
from pandera.typing import DataFrame

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report


from core.schemas.bank import \
    BankInputSchema, \
    BankOutputSchema, \
    BankMLSchema, \
    BankMLSchemaInPlace, \
    get_ml_schema

from core.settings import settings, DirectoryPath
from core.ml.factory import create_model, ModelType, \
    RandomForestClassifier, LogisticRegression
from core.ml.utils import save_model


def import_data(pth: str) -> DataFrame[BankOutputSchema]:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)

    @pa.check_types
    def check_inputs(df: DataFrame[BankInputSchema]
                     ) -> DataFrame[BankOutputSchema]:
        return df.assign(
            Churn=lambda x: x.Attrition_Flag.isin(["Attrited Customer"]))

    df = check_inputs(df)

    return df


def perform_eda(df: DataFrame[BankOutputSchema]) -> None:
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    HIST_COLUMNS = ["Churn", "Customer_Age"]

    # Histograms
    for col in HIST_COLUMNS:
        plt.figure(figsize=(20, 10))
        df[col].hist()
        plt.savefig(fname=settings.EDA_PATH / f"{col}_distribution.png")

    # Marital status distribution
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig(fname=settings.EDA_PATH / "marital_status_distribution.png")

    # total transaction distribution
    plt.figure(figsize=(20, 10))
    sns.distplot(df.Total_Trans_Ct)
    plt.savefig(fname=settings.EDA_PATH / "total_transaction_distribution.png")

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(fname=settings.EDA_PATH / "heatmap.png")


def encoder_helper(
        df: DataFrame[BankOutputSchema],
        category_lst: List[str],
        response: Optional[str]
) -> Union[DataFrame[BankMLSchema], DataFrame[BankMLSchemaInPlace]]:
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the
    notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
             used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for.
    """
    for col in category_lst:
        df = pd.merge(
            how='left',
            left=df,
            right=df.
                groupby(by=col).
                agg(temp=('Churn', 'mean')).
                reset_index(),
            on=col
        )

        if response:
            df = df. \
                rename(columns={'temp': col + '_' + response})

        else:
            df = df. \
                drop(columns=col). \
                rename(columns={'temp': col})

    MLSchema = get_ml_schema(response=response)

    @pa.check_types
    def check_ml_schema(
            df: DataFrame[MLSchema]
    ) -> DataFrame[MLSchema]:
        return df

    df = check_ml_schema(df)

    return df


def perform_feature_engineering(
        df: Union[DataFrame[BankMLSchema], DataFrame[BankMLSchemaInPlace]],
        response: Optional[str]
) -> Tuple[
    Union[DataFrame[BankMLSchema], DataFrame[BankMLSchemaInPlace]],
    Union[DataFrame[BankMLSchema], DataFrame[BankMLSchemaInPlace]],
    pd.Series, pd.Series
]:
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    MLSchema = get_ml_schema(response=response)

    keep_cols = list(MLSchema.to_schema().columns.keys())

    X = df[keep_cols].copy()
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def classification_report_image(
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        y_train_preds_lr: np.ndarray,
        y_train_preds_rf: np.ndarray,
        y_test_preds_lr: np.ndarray,
        y_test_preds_rf: np.ndarray,
) -> None:
    """
    produces classification report for training and testing results and stores
    report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    results_dict = {
        'Random Forest': [y_train_preds_rf, y_test_preds_rf],
        'Logistic Regression': [y_train_preds_lr, y_test_preds_lr]
    }

    for model, preds in results_dict.items():
        plt.clf()
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str(f'{model} Train'), {'fontsize': 10},
                 fontproperties='monospace')
        plt.text(0.01, 0.05,
                 str(classification_report(y_train, preds[0])),
                 {'fontsize': 10},
                 fontproperties='monospace')
        plt.text(0.01, 0.6, str(f'{model} Test'), {'fontsize': 10},
                 fontproperties='monospace')
        plt.text(0.01, 0.7,
                 str(classification_report(y_test, preds[1])),
                 {'fontsize': 10},
                 fontproperties='monospace')
        plt.axis('off')
        plt.savefig(
            settings.RESULTS_PATH / f"{model}_class_report.png"
        )


def feature_importance_plot(
        model: RandomForestClassifier,
        X_data: Union[DataFrame[BankMLSchema], DataFrame[BankMLSchemaInPlace]],
        output_pth: DirectoryPath):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(
        output_pth / 'feature_importance.png')


def train_models(
        X_train: Union[DataFrame[BankMLSchema], DataFrame[BankMLSchemaInPlace]],
        X_test: Union[DataFrame[BankMLSchema], DataFrame[BankMLSchemaInPlace]],
        y_train: pd.Series,
        y_test: pd.Series
) -> None:
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    cv_rfc = create_model(model_type=ModelType.random_forest)
    lrc = create_model(model_type=ModelType.logistic_regression)

    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # ROC Curve
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test,
                              ax=ax, alpha=0.8)
    lrc_plot = plot_roc_curve(lrc, X_test, y_test,
                              ax=ax, alpha=0.8)
    plt.savefig(
        settings.RESULTS_PATH / "roc_curves.png"
    )

    save_model(
        model=cv_rfc.best_estimator_,
        model_name='rfc_model'
    )

    save_model(
        model=lrc,
        model_name='logistic_model'
    )

    classification_report_image(
        y_train=y_train,
        y_test=y_test,
        y_train_preds_lr=y_train_preds_lr,
        y_train_preds_rf=y_train_preds_rf,
        y_test_preds_lr=y_test_preds_lr,
        y_test_preds_rf=y_test_preds_rf
    )

    feature_importance_plot(
        model=cv_rfc.best_estimator_,
        X_data=X_test,
        output_pth=settings.RESULTS_PATH
    )
