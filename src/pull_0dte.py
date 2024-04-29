import requests
import config
import datetime

# Set the parameters in config
url = config.URL
token  = config.TOKEN
ticker = config.TICKER
csv_file = config.CSV_FILE

start_time = config.START_TIME
end_time = config.END_TIME

data_dir = config.DATA_DIR

def convert_to_api_format(dt):
    return dt.strftime('%Y%m%d%H%M')

def ensure_newline_at_eof(filename):
    with open(filename, 'r+', newline='') as f:
        f.seek(0, 2)  
        if f.tell() == 0:
            return
        f.seek(f.tell() - 3, 0) 
        if f.read(1) != '\n':
            f.write('\n')

def pull_data(start_time_ = start_time, end_time_ = end_time, url_ = url, token_ = token, ticker_ = ticker, csv_file_ = csv_file, data_dir_ = data_dir):
    
    csv_file_ = data_dir_ / "pulled" / csv_file_

    current_time = start_time_
    with open(csv_file_, mode='a', newline='') as file:
        while current_time <= end_time_:
            ensure_newline_at_eof(csv_file_)
            
            trade_date = convert_to_api_format(current_time)

            querystring = {"token": token_, "ticker": ticker_, "tradeDate": trade_date}
            response = requests.get(url_, params=querystring)
            if response.status_code == 200:
                if current_time == start_time_:
                    file.write(response.text)
                else:
                    file.write('\n' + '\n'.join(response.text.split('\n')[1:]))
            else:
                print(f"Failed to fetch data for {trade_date}: {response.status_code}")

            current_time += datetime.timedelta(minutes=1)

    print(f"Data successfully written to {csv_file_}")


if __name__ == "__main__":
    
    pull_data(start_time_ = start_time, end_time_ = end_time, \
              url_ = url, token_ = token, ticker_ = ticker, csv_file_ = csv_file, data_dir_ = data_dir)