# HDB resale project

## Overview

Simple project to perform analysis on HDB resale flat data on that downloads the latest HDB resale data from [https://data.gov.sg](https://data.gov.sg)

This will perform the following:

1. Download data from https://data.gov.sg

2. Combine dataframes into master table

3. Augment resale data with latitude/longitude/postal code information from OneMap API, with rate polling and responses persisted in an SQLite DB

4. Add additional engineered features

5. Modeling (TBD)

## Installation

If you have Miniconda/Anaconda installed and a virtual environment set up, you can go to step 4.

1. Download the Miniconda installation script [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh).

2. Run `bash Miniconda3-latest-Linux-x86_64.sh` where you downloaded the script.

3. Run `conda create --name <venv_name> python=3.6` where <venv_name> is the name of your virtual environment. This code was tested with Python 3.6.

4. Run `conda activate <venv_name>`.

### Install dependencies

1. Change to the project folder and run `pip install -r requirements.txt`

### Demo

Change to `src` subfolder and run `python generate_data.py parameters.yml`
