# -*- coding: utf-8 -*-
"""Machine learning model factory module.

Created on: 08/03/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Union
from dataclasses import dataclass
from enum import Enum

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from core.settings import config
from core.exceptions import ModelNotImplementedError


class ModelType(str, Enum):
    """Available Learning Models."""

    random_forest = 'random_forest'
    logistic_regression = 'logistic_regression'


@dataclass
class ModelFactory:
    """Machine Learning Models Factory."""

    random_forest: RandomForestClassifier = RandomForestClassifier(
        **config.hyperparameters.get('random_forest').get('init')
    )
    logistic_regression: LogisticRegression = LogisticRegression()


def get_model(
        model_type: ModelType
) -> Union[RandomForestClassifier, LogisticRegression]:
    """Return Machine learning model instance.

    Args:
        model_type: Model Type.

    Returns:
        model: machine learning model instance.
    """
    model_factory = ModelFactory()

    try:
        model = model_factory.__getattribute__(model_type)
        return model

    except AttributeError:
        raise ModelNotImplementedError(f'Model type {model_type} '
                                       f'not implemented in ModelFactory')


def apply_grid_search(
        model: Union[RandomForestClassifier, LogisticRegression],
        model_type: ModelType
) -> Union[GridSearchCV, LogisticRegression]:
    """Apply grid search over a machine learning model instance.

        Args:
            model: machine learning model instance.
            model_type: Model Type.

        Returns:
            model: machine learning model instance.
    """

    grid = config.hyperparameters.get(model_type).get('grid', None)

    if grid:
        param_grid = grid. \
            get('params')

        cv = grid. \
            get('cv', 1)

        cv_model = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv
        )

        return cv_model

    else:

        return model


def create_model(
    model_type: ModelType
) -> Union[GridSearchCV, LogisticRegression]:
    """Machine learning wrapper factory."""

    model = get_model(model_type=model_type)

    return apply_grid_search(model=model, model_type=model_type)

