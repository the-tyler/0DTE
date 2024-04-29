import pandas as pd
import numpy as np

import config
import os

data_dir = config.DATA_DIR
csv_file = config.CSV_FILE

def load_clean(data_dir_ = data_dir, csv_file_ = csv_file, dte_ = 1):

    """
    Loads 0 dte options by default.
    """
    
    ## Read and Prepare the Data
    path = data_dir_ / "pulled" / csv_file_

    # If the pulled data directory is empty, then use the manually added data
    if not os.path.isfile(path):
        path = data_dir_ / "manual" / csv_file_

    try:
        df_raw = pd.read_csv(path, on_bad_lines = 'skip')
        df_clean = df_raw[df_raw["dte"] <= dte_]
        return df_clean.reset_index(drop=True)

    except Exception as e1:
        print(f"Failed to read csv with error: {e1}")

        try:
            df_raw = pd.read_csv(path, on_bad_lines='skip', quoting = 3)
            df_clean = df_raw[df_raw["dte"] <= dte_]
            return df_clean.reset_index(drop=True)
        
        except Exception as e2:
            print(f"Failed again with error: {e2}")