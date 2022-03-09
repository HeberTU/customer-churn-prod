# -*- coding: utf-8 -*-
"""Machine learning model utils.

Created on: 09/03/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Union
import joblib

from core.settings import settings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def save_model(
        model: Union[RandomForestClassifier, LogisticRegression],
        model_name: str
) -> None:
    """Write the pickled representation of the model.

    This function writes the pickled representation of the model to the
    setting's model path.


    Args:
        model: Machine learning model.
        model_name: model name.

    Returns:
        None
    """
    model_name += '.pkl'

    joblib.dump(
        value=model,
        filename=settings.MODELS_PATH / model_name
    )




