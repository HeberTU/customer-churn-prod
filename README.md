# customer-churn-prod
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

- [Origin](https://github.com/HeberTU/customer-churn-prod)
- Author: Heber Trujillo <heber.trj.urt@gmail.com>
- Date of last README.md update: 10.03.2022

## Project Overview

### Motivation

A manager at the bank is disturbed by more and more customers leaving their credit card services. They 
would appreciate it if one could predict who is going to get churned, so they can proactively go to the customer 
to provide them better services and turn customers' decisions in the opposite direction.

This project implements two machine learning approaches to identify credit card customers most likely to 
churn. Although models' performance it's essential, the focus will be developing a Python package that 
follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).

## How to Run Scripts 

### Dependencies Installation 

1. Create and activate a virtual environment for the project. For example:
    ```bash
    python3 -m venv ./.venv
    ./.venv/Scripts/activate
    ```
   
2. Install Poetry, the tool used for dependency management. To install it, run from a terminal:
    ```bash
    pip install poetry
    ```

3. From the virtual environment, install the required dependencies with:
    ```bash
    poetry install --no-root
    ```

### Scripts

The project has the following scrips:

1. **churn_library.py**: Main Script that performs churn customer analysis.
   1. Usage:
   ```bash
    python .\churn_library.py 
    ```
   2. Input Files:
      * ./data:
        * bank_data.csv
   3. Output Files:
      * ./images:
        * eda:
          * churn_distribution.png
          * customer_age_distribution.png
          * marital_status_distribution.png
          * total_transaction_distribution.png
          * heatmap.png
        * results:
          * roc_curves.png
          * feature_importance.png
          * Logistic Regression_class_report.png
          * Random Forest_class_report.png
      * ./models
        * logistic_model.pkl
        * rfc_model.pkl

2. **churn_script_logging_and_tests.py**: Test Script for churn customer analysis.
   1. Usage:
      ```bash
       python .\churn_script_logging_and_tests.py 
       ```
   2. Input Files:
      * ./data:
        * bank_data.csv
   3. Output Files:
      * ./logs
        * churn_library.log

### Package

Since the project focuses on showing coding and engineering best practices, I've created a core 
python package that encapsulates the machinery needed to perform the churn customer analysis. This 
python package has the following structure:

* ./core
  * config: 
    * hyperparameters.yaml: Model parameters configuration.
    * test_parameters.yaml: Constant variables for testing purposes.
  * ml:
    * factory.py: this module implements the [factory](https://refactoring.guru/design-patterns/factory-method) pattern
    to create the machine learning models 
    * utils.py: save and load functions
  * schemas:
    * bank.py: This module implements [pandera](https://pandera.readthedocs.io/en/latest/schema_models.html) schemas to 
    performing data validation
  * settings: This module implements all the configurations and settings needed to run the main scripts.
  * exceptions.py: This module has the ad-hoc exceptions needed to run the main scripts.

## Contribute 

### Conventions

#### Linting

All valid python files (*.py) must pass a linting process; It takes care of code format, style, documentation, 
and imports via flake8, black, isort, pydocstyle. The lining process can be run using the following command:
   
   ```bash
    .\scripts\linting\bach-lint.bat ./Path_to/PYTHON_FILE.py
   ```
