import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import sys, re, warnings

warnings.filterwarnings("ignore")

# =============================================================================
print ("Prophet Prediction")
print ("="*70)
print ("Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.")
print ("="*70)
print ("Parameters to include: 'datafile.csv u 0:4', 'split_percentage=0.8', 'show_fft=False', 'show_decompose=False', 'daily_seasonality=True, 'frequency'=D")
print ("="*70)

# =============================================================================
# Take user input otherwise use default values except for datafile
data_stx = input("Data Syntax: ")

# Parse the data syntax into datafile, col1, col2
datafile = data_stx.split(" ")[0]
col1 = int(data_stx.split(" ")[2].split(":")[0])-1
col2 = int(data_stx.split(" ")[2].split(":")[1])-1

try:
    split_percentage = float(input("Split Percentage: "))
except ValueError:
    split_percentage = 0.8
    
show_fft = True if input("Show FFT: ") == "True" else False
show_decompose = True if input("Show Decompose: ") == "True" else False
daily_seasonality = True if input("Daily Seasonality: ") == "True" else False
frequency = input("Frequency: ") or "D"

print ("Given Parameters: ", datafile, split_percentage, show_fft, show_decompose, daily_seasonality, frequency)

# =============================================================================
# Function to check if the first column contains dates
def is_first_column_date(df):
    first_row_first_col = df.iloc[1, 0]

    # Regex to check if the first row of the first column is a date
    date_regex = re.compile(r'\d{4}-\d{2}-\d{2}')
    # Regex to check if the first row of the first column is a datetime
    datetime_regex = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')

    is_date = bool(date_regex.match(str(first_row_first_col)))
    is_datetime = bool(datetime_regex.match(str(first_row_first_col)))

    if is_date or is_datetime:
        print ("> Date column detected.")
        return True
    else:
        print ("> Date column not detected.")
        return False
    
# If datafile extension is csv
if datafile.endswith('.csv'):
    df_data = pd.read_csv(datafile)

elif datafile.endswith('.dat') or datafile.endswith('.txt'):
    df_data = pd.read_csv(datafile, sep="\s+", header=None)

else:
    print (">> Invalid datafile extension. Please provide a csv, dat or txt file.")
    sys.exit()

print ("> Datafile loaded successfully")

# Check if the first column contains dates
if not is_first_column_date(df_data):
    # Define the starting date
    start_date = pd.to_datetime('1970-01-01')

    # Create a new date column
    df_data['Date'] = start_date + pd.to_timedelta(df_data.index, unit='D')

    # Remove the original index column
    df_data = df_data.drop(df_data.columns[0], axis=1)

    # Reorder the columns
    df_data = df_data[['Date'] + [col for col in df_data.columns if col != 'Date']]

# =============================================================================
#Select the columns to be used for the analysis
df_data = df_data.iloc[:, [col1, col2]]

# Rename second column to y_data and first column to Date
df_data = df_data.rename(columns={df_data.columns[0]: "Date", df_data.columns[1]: "y_data"})

# Convert the Date column to datetime
df_data["Date"] = pd.to_datetime(df_data["Date"])

# Split the data by percentage 
split_index = int(len(df_data) * split_percentage)

# Split the data into training and test sets
df_data_train = df_data.iloc[:split_index]
df_data_test = df_data.iloc[split_index:]

# =============================================================================
# Detect Seasonality Period
y = df_data_train['y_data'].dropna().values

# Perform Fourier Transform
fft_result = fft(y)
frequencies = np.fft.fftfreq(len(y))

if show_fft:
    # Plot the magnitude of the FFT result
    plt.figure(figsize=(12,6))
    plt.plot(np.abs(frequencies), np.abs(fft_result))
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Magnitude', fontsize=18)
    plt.title('Fourier Transform', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.show()

# Filter out zero and negative frequencies
positive_frequencies = frequencies[frequencies > 0]
positive_magnitudes = np.abs(fft_result)[frequencies > 0]

# Find peaks in the FFT magnitude spectrum
peaks, _ = find_peaks(positive_magnitudes)

# If no peaks are found, use the maximum frequency
if len(peaks) == 0:
    peak_frequency = positive_frequencies[np.argmax(positive_magnitudes)]

else:
    peak_frequency = positive_frequencies[peaks[np.argmax(positive_magnitudes[peaks])]]

# Calculate seasonality period
seasonality_period = int(1 / peak_frequency) if peak_frequency != 0 else None
print("Detected seasonality period:", seasonality_period)

# =============================================================================
if show_decompose:
    try:
        decompose = seasonal_decompose(df_data_train.y_data)
    except:
        print ("Decompose failed due to missing values. Interpolating missing values")
        df_data_train.loc[df_data_train['y_data'].isnull(), 'y_data'] = df_data_train['y_data'].interpolate()

    decompose = seasonal_decompose(df_data_train.y_data, model='additive', extrapolate_trend='freq', period=seasonality_period)

    # Plot the decomposed time series
    plt.figure(figsize=(12,6))
    plt.subplot(411)
    plt.plot(df_data_train.y_data, label='Original', color="green")
    plt.legend(loc='upper left', frameon=False)
    plt.subplot(412)
    plt.plot(decompose.trend, label='Trend', color="blue")
    plt.legend(loc='upper left', frameon=False)
    plt.subplot(413)
    plt.plot(decompose.seasonal,label='Seasonality', color="magenta")
    plt.legend(loc='upper left', frameon=False)
    plt.subplot(414)
    plt.plot(decompose.resid, label='Residuals', color="orange")
    plt.axhline(0, linestyle='--', color='gray')
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()

# =============================================================================
df_train_prophet = df_data_train.copy()

# Date variable needs to be named "ds" for prophet
df_train_prophet = df_train_prophet.rename(columns={"Date": "ds"})

# Target variable needs to be named "y" for prophet
df_train_prophet = df_train_prophet.rename(columns={"y_data": "y"})

model_prophet  = Prophet(daily_seasonality=daily_seasonality)
model_prophet.fit(df_train_prophet)

period_in_future = len(df_data_test)
df_future = model_prophet.make_future_dataframe(periods=period_in_future, freq=frequency)

forecast_prophet = model_prophet.predict(df_future)
print ("Forecasted values:")
print ('='*70)
print(forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round().tail())
print ('='*70)

# =============================================================================
# Plot the time series 
forecast_plot = model_prophet.plot(forecast_prophet, figsize=(12, 6))

# Add a vertical line at the end of the training period
axes = forecast_plot.gca()
last_training_date = forecast_prophet['ds'].iloc[-period_in_future]
axes.axvline(x=last_training_date, color='red', linestyle='--', label='Training End')

plt.plot(df_data_test['Date'], df_data_test['y_data'], 'ro', markersize=3, label='True Test Data')
plt.xlabel('Date', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(frameon=False, fontsize=14)

plt.tight_layout()
plt.show()

# =============================================================================
# Save the forecasted values
save_forecast = input("Save Forecasted Values (True/False):")
if save_forecast == "True" or save_forecast == "true" or save_forecast == "T" or save_forecast == "t":
    forecast_prophet.to_csv("forecast_prophet.csv", index=False)
    print ("Forecasted values saved as forecast_prophet.csv")
    plt.savefig("forecast_prophet.png")