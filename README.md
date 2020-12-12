# [Car Price Predictor - Linear Regression](#car-price-predictor---linear-regression)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edesz/car-price-predictor-linear-regression/master?urlpath=lab) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/car-price-predictor-linear-regression/master/5_experiments_in_regression.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/edesz/car-price-predictor/tree/master/) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) ![CI](https://github.com/edesz/car-price-predictor-linear-regression/workflows/CI/badge.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/mit)

[**See the accompanying blog post for this project here**](https://edesz.github.io/explanatory-pages/end-to-end-ml-development-project/web-scraping/exploratory-data-analysis/project/machine-learning/2020/12/11/cpp.html)

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
This project aims to use features/attributes of brand new cars to predict their sale price in Austin, TX and Seattle, WA within 20 miles of a one or two local zipcodes within each city. The predicted price should be within $5,000 of the true price. The cars are listed on the e-marketplace cars.com.

## [Motivation](#motivation)
The ML model is developed here in order to be embedded in a car pricing service using listings on cars.com. Listing prices will be predicted as part of this service. The intended client is a website that offers financing for vehicle purchases made on cars.com. This financing option offered to their customers is different to what is offered by cars.com and so the customer can then choose between the two when purchasing a vehicle from the e-marketplace.

Although the listing price is available on cars.com, the client wants an independent estimate of the cost of the listing in order to ensure they can rely on this number when determining the best financing package to offer their customers.

Our client expects that customers will use their website to specify vehicle requirements, maximum price, etc. The website will then call our pricing service to get an approximate cost of matching listings found on cars.com. Based on our service's predicted vehicle price, our client's existing fiancing service (also hosted on the website) will offer financing options to the customer to support purchase of the vehicle. This financing is not intended to cover the full price of vehicle as listed on cars.com.

When this service is deployed, using knowledge of the field and factors identified as being important predictors of the listing price, the client's own financing service weighs several factors when determining the best financing option to offer. Among these are factors that are used by the client's internal financing algorithm (such as credit history) and also the factors found to be important predictors of the predicted listing price that comes from our service. The customer may wish to know why they were not offered certain financing options. The client will refer to these important predictors of the estimated price as being part of the reason for the choice of financing plan they offered their customer.

Also when this service is deployed, financing will be offered based on the estimated price. The client's hope is that this estimated price will be within the range where they haven't observed unhappy customers. Depending on this estimation, and on our determination of the factors important to the listing price, the client may also choose not to offer financing. This represents a lost customer to them, but if the predicted and true prices are significantly different then the client may not be able to make a reliable financing offer to their customers. Factors important to a listing price are not shown on cars.com. The client will be relying on our service's estimation of these when building a custom financing package. So, we need to develop a quantitative predictive technique that is efficient and also interpretable.

Since the client wants to minimize inaccurate pricing details on their website, our service will be subjected to greater penalties for predictions outside (above or below) than within a $5,000 range surrounding the true price of the car. For predictions that fall within this range, the client will pay us a fee corresponding to an agreed-upon fraction of the true cost (from cars.com) of the listing. Thus, in order to maximize our return on developing and offering this prediction service, we will need to generate car price predictions that must be within $5,000 of the true listing price.

Note that our service is also expected to retrieve the matching listings from cars.com based on the customer's input preferences. So, we will also need to [scrape](https://en.wikipedia.org/wiki/Web_scraping) cars.com to retrieve the listings data that we use to train the quantitative prediction model used by our price prediction service.

### [Objective](#objective)
Here, we will focus on the data acquisition (web scraping) and quantitative analysis (machine learning) parts of this use-case. The objective of this project is to develop a minimum viable ML technique for estimating listing price. We'll only earn a return on this service if it is performant. So, we will also approximate the cost of errors in the predicted price and weigh this against sufficiently accurate predictions, in order to determine whether the achieved level of performance of our service will make us a net return or end up costing us.

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
   - `5_experiments_in_regression.ipynb`
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
    ├── .old_scripts                  <- Python scripts not used in analysis.
    │   ├── pd_pipe_helpers.py
    |
    ├── *.ipynb                       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                    and a short `-` delimited description
    │
    ├── requirements.txt              <- packages required to execute all Jupyter notebooks interactively (not from CI)
    ├── setup.py                      <- makes project pip installable (pip install -e .) so `src` can be imported
    ├── src                           <- Source code for use in this project.
    │   ├── __init__.py               <- Makes src a Python module
    │   └── *.py                      <- Scripts to use in analysis for pre-processing, training, etc.
    ├── papermill_runner.py           <- Python functions that execute system shell commands.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    └── tox.ini                       <- tox file with settings for running tox; see tox.testrun.org

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #<a target="_blank" href="https://asciinema.org/a/244658">cookiecutterdatascience</a></small></p>
