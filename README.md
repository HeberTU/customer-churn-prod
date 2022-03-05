# customer-churn-prod
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

- [Origin](https://github.com/HeberTU/customer-churn-prod)
- Author: Heber Trujillo <heber.trj.urt@gmail.com>
- Date of last README.md update: 05.03.2022

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


## Contribute 

### Conventions

#### Linting

All valid python files (*.py) must pass a linting process; It takes care of code format, style, documentation, 
and imports via flake8, black, isort, pydocstyle. The lining process can be run using the following command:
   
   ```bash
    .\scripts\linting\bach-lint.bat ./Path_to/PYTHON_FILE.py
   ```
