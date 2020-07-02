#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Run notebooks."""


import os
from datetime import datetime
from typing import Dict, List

import papermill as pm

PROJ_ROOT_DIR = os.path.abspath(os.getcwd())
output_notebook_dir = os.path.join(PROJ_ROOT_DIR, "executed_notebooks")

five_dict_nb_name = "5_experiments_in_regression.ipynb"

five_dict = {
    "nums": ["Mileage", "MPG", "consumer_reviews", "seller_reviews"],
    "cats": [
        "Fuel Type",
        "Drivetrain",
        "State",
        "Comfort",
        "Exterior Styling",
        "Interior Design",
        "year",
        "trans_speed",
    ],
    "NUM_NESTED_CV_TRIALS": 5,
}


def papermill_run_notebook(
    nb_dict: Dict, output_notebook_dir: str = "executed_notebooks"
) -> None:
    """Execute notebook with papermill.
    Parameters
    ----------
    nb_dict : Dict
        nested dictionary of parameters needed to run a single notebook with
        key as notebook name and value as dictionary of parameters and values
    Usage
    -----
    > import os
    > papermill_run_notebook(
          nb_dict={
              os.path.join(os.getcwd(), '0_demo.ipynb'): {'mylist': [1,2,3]}
          }
      )
    """
    for notebook, nb_params in nb_dict.items():
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_nb = os.path.basename(notebook).replace(
            ".ipynb", f"-{now}.ipynb"
        )
        print(
            f"\nInput notebook path: {notebook}",
            f"Output notebook path: {output_notebook_dir}/{output_nb} ",
            sep="\n",
        )
        for key, val in nb_params.items():
            print(key, val, sep=": ")
        pm.execute_notebook(
            input_path=notebook,
            output_path=f"{output_notebook_dir}/{output_nb}",
            parameters=nb_params,
        )


def run_notebooks(
    notebook_list: List, output_notebook_dir: str = "executed_notebooks"
) -> None:
    """Execute notebooks from CLI.
    Parameters
    ----------
    nb_dict : List
        list of notebooks to be executed
    Usage
    -----
    > import os
    > run_notebook(
          notebook_list=[
              os.path.join(os.getcwd(), five_dict_nb_name)
          ]
      )
    """
    for nb in notebook_list:
        papermill_run_notebook(
            nb_dict=nb, output_notebook_dir=output_notebook_dir
        )


if __name__ == "__main__":
    PROJ_ROOT_DIR = os.getcwd()
    notebook_list = [
        {os.path.join(PROJ_ROOT_DIR, five_dict_nb_name): five_dict}
    ]
    run_notebooks(
        notebook_list=notebook_list, output_notebook_dir=output_notebook_dir
    )
