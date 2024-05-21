import os
import load_0dte
import config

data_dir = config.DATA_DIR
csv_file = config.CSV_FILE
csv_file_0dte = config.CSV_FILE_0DTE

def save_data_and_clean(data_dir_ = data_dir, csv_file_ = csv_file, csv_file_0dte_ = csv_file_0dte):

    try:
        clean_df = load_0dte.load_clean(data_dir_, csv_file_)
        clean_df.to_csv(data_dir_ / "derived" / csv_file_0dte_)

        print(f"{csv_file_0dte_} is saved into derived")

        try:
            os.remove(data_dir_ / "pulled" / csv_file_)
            print(f"{csv_file_} is removed from pulled")
        except FileNotFoundError:
            os.remove(data_dir_ / "manual" / csv_file_)
            print(f"{csv_file_} is removed from manual")
    
    except Exception as e:
        print(f"Failed to save with error: {e}")

if __name__ == "__main__":
    save_data_and_clean(data_dir_ = data_dir, csv_file_ = csv_file, csv_file_0dte_ = csv_file_0dte)