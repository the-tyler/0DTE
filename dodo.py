"""
Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based
"""

import sys
sys.path.insert(1, './src/')

import config
from pathlib import Path
import os

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

CSV_FILE = config.CSV_FILE

def task_create_pulled_dir():
    pulled_dir = DATA_DIR / 'pulled'
    return {
        "actions":[(os.makedirs, [pulled_dir], {"exist_ok":True})]
                   }

def task_create_output_dir():
    return {
        "actions":[(os.makedirs, [OUTPUT_DIR], {"exist_ok": True})]
    }

def task_pull_data():

    file_dep = [
        "./src/config.py",
        "./src/pull_0dte.py"
        ]

    targets = [
        Path(DATA_DIR) / "pulled" / file for file in 
        [
            CSV_FILE
        ]
    ]

    return {
        "actions" : [
            "ipython src/config.py",
            "ipython src/pull_0dte.py"
        ],
        "targets" : targets,
        "file_dep" : file_dep,
        "clean": True
    }