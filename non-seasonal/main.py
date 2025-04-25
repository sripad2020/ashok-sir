import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Step 1: Data Visualization and Preprocessing
# Load the dataset
df = pd.read_csv('time-series-19-covid-combined.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Let's focus on Argentina's confirmed cases
argentina = df[df['Country/Region'] == 'Argentina'][['Confirmed']].dropna()
argentina = argentina.asfreq('D')  # Ensure daily frequency

# Plot the data
plt.figure(figsize=(12, 6))
argentina['Confirmed'].plot(title='Daily Confirmed COVID-19 Cases in Argentina')
plt.ylabel('Confirmed Cases')
plt.xlabel('Date')
plt.grid(True)
plt.show()


# Step 2: Make the dataset stationary if not stationary
# Test for stationarity
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

    # Plot rolling statistics
    rolmean = timeseries.rolling(window=7).mean()
    rolstd = timeseries.rolling(window=7).std()

    plt.figure(figsize=(12, 6))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.grid(True)
    plt.show()


test_stationarity(argentina['Confirmed'])

# The data is clearly not stationary (p-value > 0.05). Let's difference it.
argentina['First Difference'] = argentina['Confirmed'] - argentina['Confirmed'].shift(1)
argentina['First Difference'].dropna().plot(figsize=(12, 6))
plt.title('First Difference')
plt.grid(True)
plt.show()

test_stationarity(argentina['First Difference'].dropna())

# Still not stationary enough, let's try second difference
argentina['Second Difference'] = argentina['First Difference'] - argentina['First Difference'].shift(1)
argentina['Second Difference'].dropna().plot(figsize=(12, 6))
plt.title('Second Difference')
plt.grid(True)
plt.show()

test_stationarity(argentina['Second Difference'].dropna())

# Step 3: Finding models (ARIMA)
# Plot ACF and PACF to determine AR and MA terms
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(argentina['Second Difference'].dropna(), lags=20, ax=ax1)
plot_pacf(argentina['Second Difference'].dropna(), lags=20, ax=ax2)
plt.show()

# Based on ACF and PACF, let's try ARIMA(1,2,1)
# Step 4: Parameter Redundancy and Step 5: Parameter Estimation
model = ARIMA(argentina['Confirmed'], order=(1, 2, 1))
model_fit = model.fit()
print(model_fit.summary())

# Step 6: Residual Analysis
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(figsize=(12, 6))
plt.title('Residuals')
plt.grid(True)
plt.show()

residuals.plot(kind='kde', figsize=(12, 6))
plt.title('Residual Density')
plt.grid(True)
plt.show()

print(residuals.describe())

# Ljung-Box test for residual autocorrelation
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("Ljung-Box test for residual autocorrelation:")
print(lb_test)

# Step 7: Forecasting
# Split into train and test sets
train = argentina['Confirmed'][:int(0.8 * (len(argentina)))]
test = argentina['Confirmed'][int(0.8 * (len(argentina))):]

# Fit model on training data
model = ARIMA(train, order=(1, 2, 1))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast, index=test.index)

# Plot forecasts against actual values
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training')
plt.plot(test, label='Actual')
plt.plot(forecast, label='Forecast')
plt.title('ARIMA Forecast vs Actuals')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'Test RMSE: {rmse:.3f}')

# Forecast future values (next 30 days)
future_forecast = model_fit.forecast(steps=30)
future_dates = pd.date_range(start=argentina.index[-1], periods=31, freq='D')[1:]

plt.figure(figsize=(12, 6))
plt.plot(argentina['Confirmed'], label='Historical')
plt.plot(future_dates, future_forecast, label='30-Day Forecast')
plt.title('30-Day COVID-19 Cases Forecast for Argentina')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.legend()
plt.grid(True)
plt.show()

print("30-Day Forecast:")
print(pd.DataFrame({'Date': future_dates, 'Forecasted Cases': future_forecast}).set_index('Date'))