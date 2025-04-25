import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from arch import arch_model

warnings.filterwarnings('ignore')
# Load the data
df = pd.read_csv('Country Wise Airport.csv')

# Data Cleaning
# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Fix data entry errors (like '57.l' which should be '57.1')
df = df.replace('57.l', '57.1', regex=True)
df = df.replace('I.I', '1.1', regex=True)
df = df.replace('II.I', '11.1', regex=True)
df = df.replace('IO.I', '10.1', regex=True)

# Convert all numeric columns to float
numeric_cols = df.columns[1:]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Reshape the data for better analysis
years = ['2014', '2015', '2016', '2017', '2018', '2019', '2020']
airports = ['Delhi', 'Mumbai', 'Chennai', 'Calicut', 'Benguluru', 'Kolkata', 'Hyderabad', 'Cochin']

# Create a multi-index dataframe
data = []
for year in years:
    for airport in airports:
        col_name = f"{year} {airport} (Airport)"
        if col_name in df.columns:
            temp_df = df[['Country of Nationality', col_name]].copy()
            temp_df['Year'] = year
            temp_df['Airport'] = airport
            temp_df = temp_df.rename(columns={col_name: 'Percentage'})
            data.append(temp_df)

long_df = pd.concat(data, ignore_index=True)

# Remove rows with missing values
long_df = long_df.dropna()

# Create a summary dataframe for country totals
country_totals = long_df.groupby(['Country of Nationality', 'Year'])['Percentage'].sum().reset_index()
# Get top 10 countries by average percentage
top_countries = country_totals.groupby('Country of Nationality')['Percentage'].mean().nlargest(10).index.tolist()

fig1 = px.line(country_totals[country_totals['Country of Nationality'].isin(top_countries)],
              x='Year', y='Percentage', color='Country of Nationality',
              title='Top 10 Countries by Total Passenger Percentage (2014-2020)',
              labels={'Percentage': 'Total Passenger Percentage (%)'},
              template='plotly_white')
fig1.update_layout(hovermode='x unified')
fig1.show()
airport_year = long_df.groupby(['Airport', 'Year'])['Percentage'].mean().reset_index()

fig2 = px.density_heatmap(airport_year, x='Year', y='Airport', z='Percentage',
                         title='Average Passenger Distribution by Airport and Year',
                         color_continuous_scale='Viridis',
                         labels={'Percentage': 'Avg. Percentage (%)'},
                         template='plotly_white')
fig2.show()

# Get data for a specific year (2019 as example)
year_data = long_df[long_df['Year'] == '2019']

fig3 = px.sunburst(year_data, path=['Country of Nationality', 'Airport'], values='Percentage',
                  title='Passenger Distribution by Country and Airport (2019)',
                  color_continuous_scale='RdBu',
                  template='plotly_white')
fig3.update_traces(textinfo="label+percent parent")
fig3.show()

# Select some representative countries
sample_countries = ['United States Of America', 'United Kingdom', 'China', 'Germany', 'Japan']

fig4 = px.bar(long_df[long_df['Country of Nationality'].isin(sample_countries) & (long_df['Year'] == '2019')],
             x='Country of Nationality', y='Percentage', color='Airport',
             title='Airport Preferences by Country (2019)',
             barmode='group',
             labels={'Percentage': 'Passenger Percentage (%)'},
             template='plotly_white')
fig4.update_layout(xaxis_title='Country', yaxis_title='Percentage (%)')
fig4.show()

# Select top airports
top_airports = long_df.groupby('Airport')['Percentage'].mean().nlargest(4).index.tolist()

fig5 = px.line(long_df[long_df['Airport'].isin(top_airports)],
              x='Year', y='Percentage', color='Country of Nationality',
              facet_col='Airport', facet_col_wrap=2,
              title='Passenger Percentage Trends for Top Airports by Country',
              labels={'Percentage': 'Passenger Percentage (%)'},
              template='plotly_white')
fig5.update_layout(showlegend=False)
fig5.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig5.show()

# We'll use country totals for 2019 for the map
map_data = country_totals[country_totals['Year'] == '2019']

fig6 = px.choropleth(map_data,
                    locations='Country of Nationality',
                    locationmode='country names',
                    color='Percentage',
                    hover_name='Country of Nationality',
                    color_continuous_scale='Plasma',
                    title='Global Passenger Distribution to Indian Airports (2019)',
                    labels={'Percentage': 'Total Percentage (%)'},
                    template='plotly_white')
fig6.show()
# Calculate average percentage by airport and year
airport_avg = long_df.groupby(['Airport', 'Year'])['Percentage'].mean().reset_index()

fig7 = px.scatter(airport_avg, x='Year', y='Percentage', color='Airport',
                 size='Percentage', animation_frame='Year',
                 title='Evolution of Airport Preferences Over Time',
                 labels={'Percentage': 'Average Passenger Percentage (%)'},
                 range_y=[0, airport_avg['Percentage'].max() * 1.1],
                 template='plotly_white')
fig7.update_layout(showlegend=True)
fig7.show()

country = 'United States Of America'
airport = 'Delhi'
ts_data = df[df['Country of Nationality'] == country][[col for col in df.columns if airport in col]].T
ts_data.columns = ['Percentage']
ts_data.index = pd.to_datetime(ts_data.index.str.extract('(\d{4})')[0] + '-01-01')
ts_data = ts_data.asfreq('AS-JAN')  # Annual start frequency


## Step 2: Check Stationarity and Make Stationary if Needed
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

    # Plot rolling statistics
    rolmean = timeseries.rolling(window=3).mean()
    rolstd = timeseries.rolling(window=3).std()

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries, name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolmean.index, y=rolmean, name='Rolling Mean'), row=1, col=1)
    fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries, name='Original'), row=2, col=1)
    fig.add_trace(go.Scatter(x=rolstd.index, y=rolstd, name='Rolling Std'), row=2, col=1)
    fig.update_layout(title='Rolling Mean & Standard Deviation', height=600)
    fig.show()

    return dftest[1] > 0.05  # Return True if non-stationary


print(f"Testing stationarity for {country} passengers at {airport} airport:")
is_non_stationary = test_stationarity(ts_data['Percentage'])

# Make data stationary if needed
if is_non_stationary:
    print("\nData is non-stationary. Applying differencing...")
    ts_data['Percentage_diff'] = ts_data['Percentage'].diff().dropna()
    is_still_non_stationary = test_stationarity(ts_data['Percentage_diff'])

    if is_still_non_stationary:
        print("\nData is still non-stationary after first difference. Applying log transformation...")
        ts_data['Percentage_log'] = np.log(ts_data['Percentage'])
        ts_data['Percentage_log_diff'] = ts_data['Percentage_log'].diff().dropna()
        test_stationarity(ts_data['Percentage_log_diff'])
        stationary_series = ts_data['Percentage_log_diff']
    else:
        stationary_series = ts_data['Percentage_diff']
else:
    print("\nData is stationary. Proceeding to model identification.")
    stationary_series = ts_data['Percentage']


def plot_acf_pacf(timeseries, lags=20):
    # ACF and PACF Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # ACF plot
    plot_acf(timeseries, lags=lags, ax=ax1)

    # PACF plot
    plot_pacf(timeseries, lags=lags, ax=ax2)

    plt.tight_layout()
    plt.show()


# Assuming `stationary_series` is your stationary time series data
# Here, I'm creating a dummy stationary time series for illustration purposes.
# Replace this with your actual data (e.g., after differencing if necessary).
stationary_series = pd.Series(np.random.randn(100))  # Replace with your actual data

# ACF and PACF plots for stationary series
plot_acf_pacf(stationary_series)

# ARIMA Model - We will determine p, d, q based on the ACF/PACF plots
# For demonstration, using p=1, d=1, q=1 as an example.

p, d, q = 1, 1, 1  # Adjust these values based on your analysis of ACF/PACF plots

# Fit the ARIMA model
model = ARIMA(stationary_series, order=(p, d, q))
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Step 4: Residual Diagnostics - Check if the residuals from the model are white noise
residuals = model_fit.resid

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals from ARIMA Model')
plt.show()

# Perform Ljung-Box test on the residuals to check if the residuals are white noise
ljung_box_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("Ljung-Box Test Results:")
print(ljung_box_test)

# Step 5: Forecasting - Using the model to forecast future values
forecast_steps = 5  # Forecast for the next 5 steps (adjust based on your requirements)
forecast = model_fit.forecast(steps=forecast_steps)

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(stationary_series, label='Historical Data')
plt.plot(np.arange(len(stationary_series), len(stationary_series) + forecast_steps), forecast, label='Forecast',
         color='red')
plt.title(f'ARIMA({p},{d},{q}) Forecast')
plt.legend()
plt.show()

# Step 4 (Continued): Residual Diagnostics - Check if the residuals from the model are white noise
residuals = model_fit.resid

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals from ARIMA Model')
plt.show()

# Perform Ljung-Box test on the residuals to check if the residuals are white noise
ljung_box_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("Ljung-Box Test Results:")
print(ljung_box_test)

# Step 5: Forecasting - Using the model to forecast future values
forecast_steps = 5  # Forecast for the next 5 steps (adjust based on your requirements)
forecast = model_fit.forecast(steps=forecast_steps)

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(stationary_series, label='Historical Data')
plt.plot(np.arange(len(stationary_series), len(stationary_series) + forecast_steps), forecast, label='Forecast', color='red')
plt.title(f'ARIMA({p},{d},{q}) Forecast')
plt.legend()
plt.show()


# Step 5: Forecasting - Using the model to forecast future values
forecast_steps = 5  # Forecast for the next 5 time steps (adjust as needed)

# Forecasting the next steps using the ARIMA model
forecast = model_fit.forecast(steps=forecast_steps)

# Plot the historical series along with the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(stationary_series, label='Historical Data', color='blue')  # Historical data (stationary series)
plt.plot(np.arange(len(stationary_series), len(stationary_series) + forecast_steps), forecast, label='Forecast', color='red')
plt.title(f'ARIMA({p},{d},{q}) Forecast for the next {forecast_steps} steps')
plt.legend()
plt.show()

p, d, q = 1, 1, 1  # Example parameters, adjust according to your ACF/PACF analysis
arima_model = ARIMA(stationary_series, order=(p, d, q))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=5)

# Forecasting with SARIMA model (Seasonal ARIMA)
sarima_model = SARIMAX(stationary_series, order=(1,1,1), seasonal_order=(1,1,1,12))  # Adjust seasonal order
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.forecast(steps=5)

# Forecasting with Exponential Smoothing
es_model = ExponentialSmoothing(stationary_series, trend='add', seasonal='add', seasonal_periods=12)
es_fit = es_model.fit()
es_forecast = es_fit.forecast(steps=5)

# Forecasting with Prophet model (make sure the series is in a DataFrame with columns ['ds', 'y'])
df = stationary_series.reset_index()
df.columns = ['ds', 'y']

# Forecasting with Linear Regression model (using time as a feature)
X = np.arange(len(stationary_series)).reshape(-1, 1)
y = stationary_series.values
lr_model = LinearRegression()
lr_model.fit(X, y)
lr_forecast = lr_model.predict(np.arange(len(stationary_series), len(stationary_series) + 5).reshape(-1, 1))

# Plotting all forecasts together
plt.figure(figsize=(12, 8))
plt.plot(stationary_series, label='Historical Data', color='blue')  # Historical data (stationary series)

# Plotting each model's forecast
plt.plot(np.arange(len(stationary_series), len(stationary_series) + 5), arima_forecast, label='ARIMA Forecast', color='red')
plt.plot(np.arange(len(stationary_series), len(stationary_series) + 5), sarima_forecast, label='SARIMA Forecast', color='green')
plt.plot(np.arange(len(stationary_series), len(stationary_series) + 5), es_forecast, label='Exponential Smoothing Forecast', color='purple')
plt.plot(np.arange(len(stationary_series), len(stationary_series) + 5), lr_forecast, label='Linear Regression Forecast', color='brown')

plt.title(f"Comparison of Forecasts (5 steps ahead)")
plt.legend()
plt.show()

# Printing forecasted values for all models
print("ARIMA Forecast:", arima_forecast)
print("SARIMA Forecast:", sarima_forecast)
print("Exponential Smoothing Forecast:", es_forecast)
print("Linear Regression Forecast:", lr_forecast)

# Step 6: Residual Analysis (Diagnostic Checks)

# Fit ARIMA model (you can adjust (p,d,q) parameters based on previous steps)
p, d, q = 1, 1, 1  # Example parameters, adjust according to your ACF/PACF analysis
arima_model = ARIMA(stationary_series, order=(p, d, q))
arima_fit = arima_model.fit()

# Get residuals from the ARIMA model
residuals = arima_fit.resid

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals from ARIMA Model')
plt.show()

# Perform Ljung-Box test on the residuals to check if the residuals are white noise
ljung_box_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("Ljung-Box Test Results:\n", ljung_box_test)

# Plot ACF and PACF of residuals to check for autocorrelation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plot_acf(residuals, lags=20, ax=ax1)
plot_pacf(residuals, lags=20, ax=ax2)
plt.show()

# Step 7: Forecasting and Further Analysis (ARIMA-GARCH)
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ARIMA Model - Forecasting
forecast_steps = 5  # Forecast for the next 5 time steps
arima_forecast = arima_fit.forecast(steps=forecast_steps)

# ARIMA-GARCH Model - Volatility Forecasting
garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit()
garch_forecast = garch_fit.forecast(horizon=forecast_steps)
volatility_forecast = garch_forecast.variance.values[-1, :]

# Create Plotly figure
fig = go.Figure()

# Handle both datetime and integer indices
if isinstance(stationary_series.index, pd.DatetimeIndex):
    # For datetime indices
    historical_x = stationary_series.index
    forecast_x = pd.date_range(
        start=historical_x[-1] + pd.DateOffset(years=1),
        periods=forecast_steps,
        freq='AS'
    )
else:
    # For integer/numeric indices
    historical_x = stationary_series.index
    forecast_x = np.arange(historical_x[-1] + 1, historical_x[-1] + 1 + forecast_steps)

# Add historical data
fig.add_trace(go.Scatter(
    x=historical_x,
    y=stationary_series,
    name='Historical Data',
    line=dict(color='royalblue', width=3),
    mode='lines'
))

# Add forecasted values
fig.add_trace(go.Scatter(
    x=forecast_x,
    y=arima_forecast,
    name='ARIMA Forecast',
    line=dict(color='firebrick', width=3, dash='dot'),
    mode='lines'
))

# Add volatility bands
fig.add_trace(go.Scatter(
    x=forecast_x,
    y=arima_forecast + np.sqrt(volatility_forecast),
    name='Upper Volatility Band',
    line=dict(color='gray', width=1),
    mode='lines',
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=forecast_x,
    y=arima_forecast - np.sqrt(volatility_forecast),
    name='Lower Volatility Band',
    line=dict(color='gray', width=1),
    fill='tonexty',
    fillcolor='rgba(128,128,128,0.2)',
    mode='lines'
))

# Update layout
fig.update_layout(
    title='<b>ARIMA-GARCH Forecast with Volatility Bands (5 steps ahead)</b>',
    xaxis_title='Time Period',
    yaxis_title='Value',
    hovermode='x unified',
    template='plotly_white',
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Add annotation for forecast period
fig.add_vrect(
    x0=forecast_x[0], x1=forecast_x[-1],
    fillcolor="lightgray", opacity=0.2,
    layer="below", line_width=0,
    annotation_text="Forecast Period",
    annotation_position="top left"
)

fig.show()

# Print forecast details
forecast_df = pd.DataFrame({
    'Period': forecast_x,
    'ARIMA Forecast': arima_forecast,
    'Volatility (Std Dev)': np.sqrt(volatility_forecast),
    'Upper Bound': arima_forecast + np.sqrt(volatility_forecast),
    'Lower Bound': arima_forecast - np.sqrt(volatility_forecast)
})

print("\nForecast Details:")
print(forecast_df.to_string(float_format="%.2f"))