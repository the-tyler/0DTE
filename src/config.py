"""
Provides easy access to paths and credentials used in the project.
Meant to be used as an imported module.

If `config.py` is run on its own, it will create the appropriate
directories.

Note that decouple mentions that it will help to ensure that
the project has "only one configuration module to rule all your instances."
This is achieved by putting all the configuration into the `.env` file.

"""

from decouple import config
from pathlib import Path
import datetime

# DIRECTORY MANAGEMENT
BASE_DIR = Path(__file__).parent.parent

DATA_DIR = config('DATA_DIR', default=BASE_DIR / 'data/', cast=Path)
OUTPUT_DIR = config('OUTPUT_DIR', default=BASE_DIR / 'output/', cast=Path)

# DATA PULL PARAMS
URL = "https://api.orats.io/datav2/hist/one-minute/strikes/chain"
TOKEN = "REPLACE WITH YOUR TOKEN"
TICKER = "SPX"

# Set time: (YYYY, M, D, H, M[M]) in 24h format. Set in EST - Do not touch the minute and hour unless necessary
START_TIME = datetime.datetime(2024, 4, 30, 9, 31)
END_TIME = datetime.datetime(2024, 4, 30, 16, 0)

# Name the csv file as "month_day_data.csv"
CSV_FILE = f"{START_TIME.month}_{START_TIME.day}_{START_TIME.year}_data.csv"
CSV_FILE_0DTE = f"{START_TIME.month}_{START_TIME.day}_{START_TIME.year}_0DTE.csv"

if __name__ == "__main__":
    
    ## If they don't exist, create the data and output directories
    (DATA_DIR / 'pulled').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'derived').mkdir(parents=True, exist_ok=True)

    # Sometimes, I'll create other folders to organize the data
    (DATA_DIR / 'manual').mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'plots').mkdir(parents=True, exist_ok=True)
