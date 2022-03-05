# -*- coding: utf-8 -*-
"""settings module.

Created on: 04/03/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import os
from pathlib import Path

from pydantic import BaseSettings
from pydantic.types import DirectoryPath


class Settings(BaseSettings):
    """This class manages all project settings."""

    PACKAGE_PATH: DirectoryPath = Path(__file__).parents[2]

    CONFIG_PATH: DirectoryPath = Path(__file__).parents[1] / "config"
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(CONFIG_PATH)

    DATA_PATH: DirectoryPath = PACKAGE_PATH / "data"
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    IMAGES_PATH: DirectoryPath = PACKAGE_PATH / "images"
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)

    EDA_PATH: DirectoryPath = IMAGES_PATH / "eda"
    if not os.path.exists(EDA_PATH):
        os.makedirs(EDA_PATH)

    RESULTS_PATH: DirectoryPath = IMAGES_PATH / "results"
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    MODELS_PATH: DirectoryPath = PACKAGE_PATH / "models"
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    LOGS_PATH: DirectoryPath = PACKAGE_PATH / "logs"
    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)

settings = Settings()
