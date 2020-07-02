# [Car Price Predictor - Linear Regression](#car-price-predictor---linear-regression)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edesz/car-price-predictor-linear-regression/master?urlpath=lab) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/car-price-predictor-linear-regression/master/5_experiments_in_regression.ipynb) [![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/car-price-predictor-linear-regression) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) ![CI](https://github.com/edesz/car-price-predictor-linear-regression/workflows/CI/badge.svg) [![Build Status](https://dev.azure.com/elsdes3/elsdes3/_apis/build/status/edesz.car-price-predictor?branchName=master)](https://dev.azure.com/elsdes3/elsdes3/_build/latest?definitionId=25&branchName=master) [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/mit)

## [Table of Contents](#table-of-contents)
1. [Project Idea](#project-idea)
   * [Project Overview](#project-overview)
   * [Motivation](#motivation)
2. [Data acquisition](#data-acquisition)
   * [Primary data source](#primary-data-source)
   * [Supplementary data sources](#supplementary-data-sources)
   * [Data file creation](#data-file-creation)
3. [Analysis](#analysis)
4. [Usage](#usage)
5. [Project Organization](#project-organization)

## [Project Idea](#project-idea)
### [Project Overview](#project-overview)
This project aims to use features/attributes of brand new cars to predict their sale price in Austin, TX and Seattle, WA within 20 miles of a (single) local zipcode within each city. The predicted price should be within $5,000 of the true price. The cars are listed on the e-marketplace cars.com.

## [Motivation](#motivation)
The model/tool would be useful to
- a prospective cars.com user (buyer) by providing an estimate of the price they would have to pay based on specified car features, dealership ratings and their (buyer's) desired financing
- cars.com to compliment their internal pricing tool across these cities
- prospective seller (car manufacturer) looking to advertise new cars with a pricing service (cars.com) in order to approximate the sale price that can be expected in these two cities
- non-digital car dealerships, looking to expand from WA/TX into TX/WA and wanting to get a better understanding of the main factors involved in setting a price for their own (local physical site) inventory of new cars but using a larger pool (online) of listings across the two cities

## [Data acquisition](#data-acquisition)
### [Primary data source](#primary-data-source)
Scraping data from cars.com
1. Use `1_cars_controller.ipynb`
   - with `selenium` and `bs4` to apply the zipcode, radius (20 miles) and city filters
   - scrape listing URLs from the resulting search results and save the results into a Listing_IDs_*.txt file
2. Use `2_cars_listings_controller.ipynb`
   - scrape the individual listings themselves
     - this will give car features and dealership ratings and reviews

### [Supplementary data sources](#supplementary-data-sources)
Demographics data for US, based on zipcode, using Python library [`uszipcode`](https://pypi.org/project/uszipcode/). In the current version of this analysis, this supplementary data was not used.

### [Data file creation](#data-file-creation)
1. `processed_data__AUS29_SEA22_2SEAzipcodes_20191010_201827.csv`
   - includes
     - 11 pages of IDs for SEA from `Listings_IDs_20191009_SEA_2nd_zip_only.txt` (which contains 30 pages for SEA)
     - 14 pages of IDs for AUS and 11 pages of IDs for SEA from `Listings_IDs_20191005_AUS_1_to_15_SEA_11.txt` (which contains 15 pages for AUS 11 pages for SEA)
     - 15 pages of IDs for SEA from `Listings_IDs_20191008_AUS_16_to_29.txt` (which contains 30 pages for AUS)

## [Analysis](#analysis)
Analysis will be performed using linear models in Python. Details are included in the various notebooks in the root directory.

## [Usage](#usage)
1. Clone this repository
   ```bash
   $ git clone
   ```
2. Create Python virtual environment, install packages and launch interactive Python platform
   ```bash
   $ make build
   ```
3. Run notebooks in the following order
   - `1_cars_controller.ipynb`
     - programmatically searches for car listings and scrapes listing IDs from each page of search results
     - saves scraped ID to `Listings_IDs_*.txt`, one line per page of search results
   - `2_cars_listing.ipynb`
     - scrapes car listing details using listing IDs scraped in `1_cars_controller.ipynb`
     - saves scraped details in `data/(AUS or SEA)/p#_0_99.csv` where `#` is the zero-indexed page number of search results found in `1_cars_controller.ipynb`
     - uses `src/listing_scraper.py`
   - `2_get_cars_listings_details.ipynb`
     - scrapes the details of a single listing
       - this was the notebook used to develop the code in `src/listing_scraper.py`
     - this requires that the webpage for a single listing is saved to a `*.html` file in `data/*.html`
   - `3_cleaning.ipynb`
     - extracts features (for linear regression analysis) from raw data
   - `4_demographics_prices_joined.ipynb`
     - combines car listings with demographics data for both cities
     - not used in current analysis
   - `5_experiments_in_regression.ipynb` ([view online](https://nbviewer.jupyter.org/github/edesz/car-price-predictor-linear-regression/executed-notebooks/blob/master/5_experiments_in_regression-20200702-113336.ipynb))
     - transforms features and filters outliers based on exploratory data analysis
     - performs linear regression analysis on retrieved data
     - evaluates performance of regression analysis

## [Project Organization](#project-organization)

    ├── .pre-commit-config.yaml       <- configuration file for pre-commit hooks
    ├── .github
    │   ├── workflows
    │       └── integrate.yml         <- configuration file for Github Actions
    ├── LICENSE
    ├── environment.yml               <- configuration file to create environment to run project on Binder
    ├── Makefile                      <- Makefile with commands like `make lint` or `make build`
    ├── README.md                     <- The top-level README for developers using this project.
    ├── data
    │   ├── AUS                       <- scraped listing details for Austin, TX
    |   └── SEA                       <- scraped listing details for Seattle, WA
    ├── executed-notebooks            <- Notebooks with output.
    │   ├── 5_experiments_in_regression-20200702-113336.ipynb
    │
    ├── .old_scripts                  <- Python scripts not used in analysis.
    │   ├── pd_pipe_helpers.py
    |
    ├── *.ipynb                       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                    and a short `-` delimited description
    │
    ├── requirements.txt              <- packages required to execute all Jupyter notebooks interactively (not from CI)
    ├── setup.py                      <- makes project pip installable (pip install -e .) so `src` can be imported
    ├── src                           <- Source code for use in this project.
    │   ├── __init__.py               <- Makes {{cookiecutter.module_name}} a Python module
    │   └── *.py                      <- Scripts to use in analysis for pre-processing, training, etc.
    ├── papermill_runner.py           <- Python functions that execute system shell commands.
    └── tox.ini                       <- tox file with settings for running tox; see tox.testrun.org

## [Notes](#notes)
1. Failing CI build is at step where data must be retrieved from cloud blob storage per [1](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python), [2](https://medium.com/@syed.sohaib/working-with-azure-blob-storage-2fbc8cfd3f7), [3](https://stackoverflow.com/questions/61935564/issues-reading-azure-blob-csv-into-python-pandas-df/61994538#61994538). Error is due to failed attempt at reading from/connecting to service and could be due to bug [here](https://github.com/microsoft/azure-pipelines-tasks/issues/13124).

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #<a target="_blank" href="https://asciinema.org/a/244658">cookiecutterdatascience</a></small></p>
