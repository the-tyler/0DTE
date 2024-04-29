"""
Tools from Nick

"""

# ==============================================

# Imports
import warnings
warnings.filterwarnings('ignore')

# tools
import pandas as pd
import numpy as np
#import dask.dataframe as dd
#import sympy
#import scipy
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

from pandas.tseries.offsets import BDay

# system
import os
from datetime import datetime
import json

# ML
#import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.callbacks import EarlyStopping

import statsmodels.api as sm

# plotting
import matplotlib.pyplot as plt


# data
#import quandl
#import wrds
#import yfinance as yf


# ==============================================


def grab_quandl_table(
    table_path,
    quandl_api_key,
    avoid_download=False,
    replace_existing=False,
    date_override=None,
    allow_old_file=False,
    **kwargs,
):
    root_data_dir = os.path.join(os.getcwd(), "quandl_data_table_downloads")
    data_symlink = os.path.join(root_data_dir, f"{table_path}_latest.zip")
    if avoid_download and os.path.exists(data_symlink):
        print(f"Skipping any possible download of {table_path}")
        return data_symlink
    
    table_dir = os.path.dirname(data_symlink)
    if not os.path.isdir(table_dir):
        print(f'Creating new data dir {table_dir}')
        os.makedirs(table_dir)

    if date_override is None:
        my_date = datetime.datetime.now().strftime("%Y%m%d")
    else:
        my_date = date_override
    data_file = os.path.join(root_data_dir, f"{table_path}_{my_date}.zip")

    if os.path.exists(data_file):
        file_size = os.stat(data_file).st_size
        if replace_existing or not file_size > 0:
            print(f"Removing old file {data_file} size {file_size}")
        else:
            print(
                f"Data file {data_file} size {file_size} exists already, no need to download"
            )
            return data_file

    dl = quandl.export_table(
        table_path, filename=data_file, api_key=quandl_api_key, **kwargs
    )
    file_size = os.stat(data_file).st_size
    if os.path.exists(data_file) and file_size > 0:
        print(f"Download finished: {file_size} bytes")
        if not date_override:
            if os.path.exists(data_symlink):
                print(f"Removing old symlink")
                os.unlink(data_symlink)
            print(f"Creating symlink: {data_file} -> {data_symlink}")
            os.symlink(
                data_file, data_symlink,
            )
    else:
        print(f"Data file {data_file} failed download")
        return
    return data_symlink if (date_override is None or allow_old_file) else "NoFileAvailable"

def fetch_quandl_table(table_path, api_key, avoid_download=True, **kwargs):
    return pd.read_csv(
        grab_quandl_table(table_path, api_key, avoid_download=avoid_download, **kwargs)
    )


def fix_concatenated_lines(file_path, fixed_file_path):
    with open(file_path, 'r') as original_file:
        content = original_file.read()

    fixed_content = content.replace('SPX,', '\nSPX,')[1:]

    with open(fixed_file_path, 'w') as fixed_file:
        fixed_file.write(fixed_content)

# ==============================================

def plot_volatility_surface(df):
    df['midIv'] = (df['callMidIv'] + df['putMidIv']) / 2

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    x = df['strike']
    y = df['dte']
    z = df['midIv']

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Days to Expiration (DTE)')
    ax.set_zlabel('Mid Implied Volatility')

    plt.title('Volatility Surface')
    plt.show()


def plot_interpolated_volatility_surface(filtered_df):
    if not filtered_df.empty:
        points = filtered_df[['strike', 'dte']].values
        values = filtered_df['midIv'].values

        grid_x, grid_y = np.mgrid[min(filtered_df['strike']):max(filtered_df['strike']):100j, 
                                  min(filtered_df['dte']):max(filtered_df['dte']):100j]

        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')

        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Days to Expiration (DTE)')
        ax.set_zlabel('Mid Implied Volatility')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title('Interpolated Volatility Surface')
        plt.show()
    else:
        print("The filtered DataFrame is empty. Unable to plot the volatility surface.")


def plot_interactive_volatility_surface(filtered_df):
    grouped = filtered_df.groupby('quoteDate')

    fig = go.Figure()

    for date, group in grouped:
        points = group[['strike', 'dte']].values
        values = group['midIv'].values
        grid_x, grid_y = np.mgrid[min(group['strike']):max(group['strike']):100j, 
                                  min(group['dte']):max(group['dte']):100j]

        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

        if not np.isnan(grid_z).all():
            fig.add_trace(
                go.Surface(z=grid_z, x=grid_x[:,0], y=grid_y[0,:], name=str(date), visible=False)
            )

    fig.data[0].visible = True

    steps = []
    for i, date in enumerate(grouped.groups.keys()):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)},
                  {'title': f'Volatility Surface for {date}'}],
            label=str(date)
        )
        step['args'][0]['visible'][i] = True 
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Timestamp: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title='Volatility Surface',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiration (DTE)',
            zaxis_title='Mid Implied Volatility'
        )
    )

    fig.show()


    
# ==============================================
def plot_dte_volatility_by_strike(df):
    df_filtered = df[df['dte'].isin([1, 2])]

    fig = go.Figure()

    dates = sorted(df_filtered['quoteDate'].unique())
    
    for date in dates:
        for dte in [1, 2]:
            temp_df = df_filtered[(df_filtered['quoteDate'] == date) & (df_filtered['dte'] == dte)]
            fig.add_trace(
                go.Scatter(
                    x=temp_df['strike'],
                    y=temp_df['midIv'],
                    mode='lines+markers',
                    name=f'DTE={dte-1}',
                    visible=False
                )
            )

    steps = []
    for i, date in enumerate(dates):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)}],
            label=date
        )
        step['args'][0]['visible'][i*2] = True  # Toggle 0 DTE trace
        step['args'][0]['visible'][i*2 + 1] = True  # Toggle 1 DTE trace
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Date: "},
        steps=steps
    )]

    # Update layout with slider
    fig.update_layout(
        sliders=sliders,
        xaxis_title='Strike Price',
        yaxis_title='Mid Implied Volatility',
        title='Volatility by Strike Over Time'
    )

    if fig.data:
        fig.data[0].visible = True
        fig.data[1].visible = True

    fig.show()



def plot_metrics_over_time_with_controls(df):
    metrics = ['gamma', 'rho', 'callVolume', 'putVolume', 'callSmvVol', 'extSmvVol']
    
    df_filtered = df[df['dte'].isin([1, 2])]

    fig = go.Figure()

    dates = sorted(df_filtered['quoteDate'].unique())
    
    for date in dates:
        for dte in [1, 2]:
            for metric in metrics:
                temp_df = df_filtered[(df_filtered['quoteDate'] == date) & (df_filtered['dte'] == dte)]
                fig.add_trace(
                    go.Scatter(
                        x=temp_df['strike'],
                        y=temp_df[metric],
                        mode='lines+markers',
                        name=f'{metric} DTE={dte-1} {date}',
                        visible=False  # Make all traces invisible by default
                    )
                )

    steps = []
    for i, date in enumerate(dates):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)},
                  {'title': f'Date: {date}'}],
            label=date
        )
        offsets = [i * len(metrics) * len([1, 2]) + dte_offset * len(metrics) for dte_offset in range(len([1, 2]))]
        for offset in offsets:
            for j in range(len(metrics)):
                step['args'][0]['visible'][offset + j] = True
        steps.append(step)

    buttons = []
    for i, metric in enumerate(metrics):
        button = dict(
            label=metric,
            method='update',
            args=[{'visible': [False] * len(fig.data)},
                  {'title': f'Metric: {metric}'}]
        )
        for j in range(len(dates) * len([1, 2])):
            button['args'][0]['visible'][i + j * len(metrics)] = True
        buttons.append(button)

    fig.update_layout(
        updatemenus=[dict(buttons=buttons,
                          direction="down",
                          pad={"r": 10, "t": 10},
                          showactive=True,
                          x=0.1,
                          xanchor="left",
                          y=1.1,
                          yanchor="top")],
        sliders=[dict(steps=steps,
                      currentvalue={"prefix": "Date: "},
                      pad={"t": 50})],
        xaxis_title='Strike Price',
        title='Metrics by Strike Over Time'
    )

    for i in range(len(metrics)):
        fig.data[i].visible = True

    fig.show()



def plot_0dte_call_volume_over_time(df):
    df_0dte = df[df['dte'] == 1]

    summed_call_volumes = []
    dates = []

    for date in sorted(df_0dte['quoteDate'].unique()):
        df_date = df_0dte[df_0dte['quoteDate'] == date]

        stock_price = df_date['stockPrice'].iloc[0]  # Assuming stockPrice is constant for each quoteDate
        lower_bound = stock_price - 100
        upper_bound = stock_price + 100

        df_about_money = df_date[(df_date['strike'] >= lower_bound) & (df_date['strike'] <= upper_bound)]

        sum_call_volume = df_about_money['callVolume'].sum()

        summed_call_volumes.append(sum_call_volume)
        dates.append(date)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates, y=summed_call_volumes, mode='lines+markers', name='0 DTE Call Volume'))

    fig.update_layout(
        title='0 DTE Call Volume Over Time (About the Money ±100)',
        xaxis_title='Date',
        yaxis_title='Sum of Call Volume',
        xaxis=dict(rangeslider=dict(visible=True), type='date')
    )

    fig.show()




def plot_0dte_call_volume_daily(df):
    df_0dte = df[df['dte'] == 1]
    daily_call_volumes = df_0dte.groupby('quoteDate')['callVolume'].sum().reset_index()

    fig = go.Figure(data=go.Scatter(x=daily_call_volumes['quoteDate'], y=daily_call_volumes['callVolume'],
                                     mode='lines+markers'))

    fig.update_layout(title='0 DTE Call Volume Daily',
                      xaxis_title='Date',
                      yaxis_title='Call Volume',
                      xaxis=dict(rangeslider=dict(visible=True), type='date'))

    fig.show()


import plotly.graph_objects as go

def plot_0dte_call_volume_by_strike_with_slider(df):
    df_0dte = df[df['dte'] == 1]

    fig = go.Figure()

    dates = sorted(df_0dte['quoteDate'].unique())

    for date in dates:
        df_date = df_0dte[df_0dte['quoteDate'] == date]
        for strike in sorted(df_date['strike'].unique()):
            df_strike = df_date[df_date['strike'] == strike]
            fig.add_trace(
                go.Scatter(
                    x=[date],
                    y=df_strike['callVolume'],
                    mode='markers',
                    name=f'Strike {strike}',
                    marker=dict(size=10),
                    visible=False
                )
            )

    steps = []
    for i, date in enumerate(dates):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)}],
            label=date
        )
        step_range = range(i * len(fig.data) // len(dates), (i + 1) * len(fig.data) // len(dates))
        for j in step_range:
            step['args'][0]['visible'][j] = True  # Toggle trace visibility
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Date: "},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title='0 DTE Call Volume by Strike Over Time',
        xaxis_title='Date',
        yaxis_title='Call Volume'
    )

    for i in range(len(fig.data) // len(dates)):
        fig.data[i].visible = True

    fig.show()



# ==============================================


def plot_0dte_call_volume_about_money_by_timestamp(df):
    df_0dte = df[df['dte'] == 1]

    plot_data = []

    for snapshot_time in sorted(df_0dte['snapShotEstTime'].unique()):
        df_snapshot = df_0dte[df_0dte['snapShotEstTime'] == snapshot_time]

        total_call_volume = 0
        for _, row in df_snapshot.iterrows():
            if row['stockPrice'] - 100 <= row['strike'] <= row['stockPrice'] + 100:
                total_call_volume += row['callVolume']
        
        plot_data.append((snapshot_time, total_call_volume))

    df_plot = pd.DataFrame(plot_data, columns=['Snapshot Time', 'Call Volume'])

    fig = go.Figure(data=go.Scatter(x=df_plot['Snapshot Time'], y=df_plot['Call Volume'], mode='lines+markers'))

    fig.update_layout(title='0 DTE Call Volume About the Money (±100) by Timestamp',
                      xaxis_title='Snapshot Time',
                      yaxis_title='Call Volume',
                      xaxis=dict(type='category'))  

    fig.show()


def plot_0dte_call_volume_by_strike_dynamic_timestamp1(df):
    df_0dte = df[df['dte'] == 1]

    fig = go.Figure()

    timestamps = sorted(df_0dte['snapShotEstTime'].unique())

    for timestamp in timestamps:
        df_timestamp = df_0dte[df_0dte['snapShotEstTime'] == timestamp]
        for strike in sorted(df_timestamp['strike'].unique()):
            df_strike = df_timestamp[df_timestamp['strike'] == strike]
            fig.add_trace(
                go.Scatter(
                    x=[strike],
                    y=df_strike['callVolume'],
                    mode='markers',
                    name=f'Strike {strike}',
                    visible=False  # Make all traces invisible by default
                )
            )

    steps = []
    for i, timestamp in enumerate(timestamps):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)}],
            label=str(timestamp)  # Convert timestamp to string
        )
        # Calculate index range for the current timestamp
        step_range = range(i * len(fig.data) // len(timestamps), (i + 1) * len(fig.data) // len(timestamps))
        for j in step_range:
            step['args'][0]['visible'][j] = True  # Toggle trace visibility
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Timestamp: "},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title='0 DTE Call Volume by Strike Over Time',
        xaxis_title='Strike Price',
        yaxis_title='Call Volume'
    )

    for i in range(len(fig.data) // len(timestamps)):
        fig.data[i].visible = True

    fig.show()


def plot_0dte_call_volume_by_strike_dynamic_timestamp2(df):
    df_0dte = df[df['dte'] == 1]

    fig = go.Figure()

    timestamps = sorted(df_0dte['snapShotEstTime'].unique())

    formatted_times = [datetime.strptime(str(ts), '%H%M').strftime('%H:%M') for ts in timestamps]

    for timestamp, formatted_time in zip(timestamps, formatted_times):
        df_timestamp = df_0dte[df_0dte['snapShotEstTime'] == timestamp]
        for strike in sorted(df_timestamp['strike'].unique()):
            df_strike = df_timestamp[df_timestamp['strike'] == strike]
            fig.add_trace(
                go.Scatter(
                    x=[strike],
                    y=df_strike['callVolume'],
                    mode='markers',
                    name=f'Strike {strike} at {formatted_time}',
                    visible=False  # Make all traces invisible by default
                )
            )

    steps = []
    for i, formatted_time in enumerate(formatted_times):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)}],
            label=formatted_time  # Use the formatted time
        )
        step_range = range(i * len(fig.data) // len(timestamps), (i + 1) * len(fig.data) // len(timestamps))
        for j in step_range:
            step['args'][0]['visible'][j] = True  # Toggle trace visibility
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Time: "},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title='0 DTE Call Volume by Strike Over Time',
        xaxis_title='Strike Price',
        yaxis_title='Call Volume'
    )

    for i in range(len(fig.data) // len(timestamps)):
        fig.data[i].visible = True

    fig.show()




def plot_0dte_call_volume_by_strike_dynamic_timestamp(df):
    df_0dte = df[df['dte'] == 1]

    fig = go.Figure()

    timestamps = sorted(df_0dte['snapShotEstTime'].unique())

    formatted_times = [datetime.strptime(str(ts), '%H%M').strftime('%H:%M') for ts in timestamps]

    trace_count_per_timestamp = []  # To keep track of the number of traces per timestamp

    for timestamp, formatted_time in zip(timestamps, formatted_times):
        df_timestamp = df_0dte[df_0dte['snapShotEstTime'] == timestamp]
        trace_count = 0  # Reset trace count for the current timestamp

        for strike in sorted(df_timestamp['strike'].unique()):
            df_strike = df_timestamp[df_timestamp['strike'] == strike]
            call_volume = df_strike['callVolume'].iloc[0]  # Assuming one entry per strike per timestamp

            fig.add_trace(
                go.Scatter(
                    x=[strike],
                    y=[call_volume],
                    mode='markers',
                    name=f'Strike {strike} at {formatted_time}',
                    visible=False  # Make all traces invisible by default
                )
            )

            trace_count += 1  # Increment trace count for the current timestamp

        trace_count_per_timestamp.append(trace_count)

    steps = []
    start_index = 0

    for i, formatted_time in enumerate(formatted_times):
        visible_traces = [False] * len(fig.data)
        end_index = start_index + trace_count_per_timestamp[i]

        for j in range(start_index, end_index):
            visible_traces[j] = True  # Make traces for the current timestamp visible

        step = dict(
            method='update',
            args=[{'visible': visible_traces}],
            label=formatted_time
        )

        steps.append(step)
        start_index = end_index  # Update start index for the next timestamp

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Time: "},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title='0 DTE Call Volume by Strike Over Time',
        xaxis_title='Strike Price',
        yaxis_title='Call Volume',
        xaxis=dict(type='category')  # Treating strikes as discrete categories
    )

    if fig.data:
        for i in range(trace_count_per_timestamp[0]):
            fig.data[i].visible = True  # Make the first set of traces visible

    fig.show()

def plot_0dte_call_volume_by_timestamp(df):
    df_0dte = df[df['dte'] == 1]
    df_0dte['formatted_time'] = df_0dte['snapShotEstTime'].apply(lambda x: datetime.strptime(str(x), '%H%M').strftime('%H:%M'))
    fig = go.Figure()

    strikes = sorted(df_0dte['strike'].unique())

    for strike in strikes:
        df_strike = df_0dte[df_0dte['strike'] == strike]

        fig.add_trace(
            go.Scatter(
                x=df_strike['formatted_time'],
                y=df_strike['callVolume'],
                mode='lines+markers',
                name=f'Strike {strike}',
                visible=False  # Make all traces invisible by default
            )
        )

    steps = []
    for i, strike in enumerate(strikes):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)},
                  {'title': f'Call Volume for Strike {strike}'}],
            label=f'Strike {strike}'
        )
        step['args'][0]['visible'][i] = True  # Toggle trace visibility for the current strike
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Strike: "},
        steps=steps
    )]

    if len(fig.data) > 0:
        fig.data[0].visible = True

    fig.update_layout(
        sliders=sliders,
        title='0 DTE Call Volume by Timestamp for Each Strike',
        xaxis_title='Timestamp',
        yaxis_title='Call Volume',
        xaxis=dict(type='category')  # Treat timestamps as discrete categories
    )

    fig.show()



def plot_0dte_call_volume_diff_by_timestamp(df):
    df_0dte = df[df['dte'] == 1].copy()
    df_0dte['formatted_time'] = df_0dte['snapShotEstTime'].apply(lambda x: datetime.strptime(str(x), '%H%M').strftime('%H:%M'))
    df_0dte.sort_values(by=['strike', 'formatted_time'], inplace=True)

    df_0dte['callVolume_diff'] = df_0dte.groupby('strike')['callVolume'].diff().fillna(0)

    fig = go.Figure()

    strikes = sorted(df_0dte['strike'].unique())

    for strike in strikes:
        df_strike = df_0dte[df_0dte['strike'] == strike]

        fig.add_trace(
            go.Scatter(
                x=df_strike['formatted_time'],
                y=df_strike['callVolume_diff'],
                mode='lines+markers',
                name=f'Strike {strike}',
                visible=False  # Make all traces invisible by default
            )
        )

    steps = []
    for i, strike in enumerate(strikes):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)},
                  {'title': f'Call Volume Difference for Strike {strike}'}],
            label=f'Strike {strike}'
        )
        step['args'][0]['visible'][i] = True  # Toggle trace visibility for the current strike
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Strike: "},
        steps=steps
    )]

    if len(fig.data) > 0:
        fig.data[0].visible = True

    fig.update_layout(
        sliders=sliders,
        title='0 DTE Call Volume Difference by Timestamp for Each Strike',
        xaxis_title='Timestamp',
        yaxis_title='Call Volume Difference',
        xaxis=dict(type='category')  # Treat timestamps as discrete categories
    )

    fig.show()



def plot_stock_price_over_time(df):
    df['formatted_time'] = df['snapShotEstTime'].apply(lambda x: datetime.strptime(str(x), '%H%M').strftime('%H:%M'))

    df.sort_values(by='formatted_time', inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['formatted_time'], y=df['stockPrice'], mode='lines+markers', name='Stock Price'))
    fig.update_layout(title='Stock Price Over Time',
                      xaxis_title='Timestamp',
                      yaxis_title='Stock Price',
                      xaxis=dict(type='category'))  # Treat timestamps as discrete categories
    fig.show()
