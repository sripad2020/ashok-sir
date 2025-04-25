import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('time-series-19-covid-combined.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
argentina = df[df['Country/Region'] == 'Argentina'][['Confirmed']].dropna()
argentina = argentina.asfreq('D')

# 1. Original Data Visualization
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=argentina.index, y=argentina['Confirmed'],
                          mode='lines+markers', name='Confirmed Cases',
                          line=dict(color='royalblue', width=2),
                          marker=dict(size=4, color='royalblue')))
fig1.update_layout(
    title='Daily Confirmed COVID-19 Cases in Argentina',
    xaxis_title='Date',
    yaxis_title='Confirmed Cases',
    template='plotly_dark',
    hovermode='x unified',
    height=600
)
fig1.show()


# 2. Stationarity Test and Visualization
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    # Rolling statistics plot
    rolmean = timeseries.rolling(window=7).mean()
    rolstd = timeseries.rolling(window=7).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries,
                             mode='lines', name='Original',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=rolmean.index, y=rolmean,
                             mode='lines', name='Rolling Mean',
                             line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=rolstd.index, y=rolstd,
                             mode='lines', name='Rolling Std',
                             line=dict(color='green', dash='dash')))
    fig.update_layout(
        title='Rolling Mean & Standard Deviation',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_dark',
        hovermode='x unified',
        height=600
    )
    fig.show()

    return dfoutput


print("Original Data Stationarity Test:")
stationarity_result = test_stationarity(argentina['Confirmed'])
print(stationarity_result)

# Differencing
argentina['First Difference'] = argentina['Confirmed'] - argentina['Confirmed'].shift(1)
argentina['Second Difference'] = argentina['First Difference'] - argentina['First Difference'].shift(1)

# Plot differences
fig_diff = make_subplots(rows=2, cols=1, subplot_titles=('First Difference', 'Second Difference'))
fig_diff.add_trace(go.Scatter(x=argentina.index, y=argentina['First Difference'],
                              mode='lines', name='First Difference',
                              line=dict(color='orange')), row=1, col=1)
fig_diff.add_trace(go.Scatter(x=argentina.index, y=argentina['Second Difference'],
                              mode='lines', name='Second Difference',
                              line=dict(color='purple')), row=2, col=1)
fig_diff.update_layout(
    title='First and Second Differences',
    template='plotly_dark',
    height=800,
    showlegend=True
)
fig_diff.show()

# Stationarity tests after differencing
print("\nFirst Difference Stationarity Test:")
print(test_stationarity(argentina['First Difference'].dropna()))

print("\nSecond Difference Stationarity Test:")
print(test_stationarity(argentina['Second Difference'].dropna()))


# Corrected ACF and PACF plots
def create_corr_plot(data, title, type='acf'):
    if type == 'acf':
        corr = acf(data, nlags=20)
    else:
        corr = pacf(data, nlags=20)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=np.arange(len(corr)),
        y=corr,
        name=title,
        marker_color='lightseagreen'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Lag',
        yaxis_title='Correlation',
        template='plotly_dark',
        height=400
    )
    fig.add_shape(type='line',
                  x0=0, x1=20,
                  y0=0, y1=0,
                  line=dict(color='white', width=1))
    return fig


# Create ACF plot
acf_fig = create_corr_plot(argentina['Second Difference'].dropna(), 'Autocorrelation Function (ACF)', 'acf')
acf_fig.show()

# Create PACF plot
pacf_fig = create_corr_plot(argentina['Second Difference'].dropna(), 'Partial Autocorrelation Function (PACF)', 'pacf')
pacf_fig.show()

# ARIMA Model
train = argentina['Confirmed'][:int(0.8 * (len(argentina)))]
test = argentina['Confirmed'][int(0.8 * (len(argentina))):]

model = ARIMA(train, order=(1, 2, 1))
model_fit = model.fit()
print(model_fit.summary())

# Residual Analysis
residuals = pd.DataFrame(model_fit.resid)

fig_residuals = make_subplots(rows=2, cols=1, subplot_titles=('Residuals Over Time', 'Residual Distribution'))
fig_residuals.add_trace(go.Scatter(x=residuals.index, y=residuals[0],
                                   mode='lines', name='Residuals',
                                   line=dict(color='cyan')), row=1, col=1)
fig_residuals.add_trace(go.Histogram(x=residuals[0],
                                     name='Residual Distribution',
                                     marker_color='teal',
                                     opacity=0.7), row=2, col=1)
fig_residuals.update_layout(
    title='Residual Analysis',
    template='plotly_dark',
    height=800
)
fig_residuals.show()

# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("Ljung-Box test for residual autocorrelation:")
print(lb_test)

# Forecasting
forecast = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast, index=test.index)

# Forecast vs Actual
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=train.index, y=train,
                                  mode='lines', name='Training Data',
                                  line=dict(color='blue')))
fig_forecast.add_trace(go.Scatter(x=test.index, y=test,
                                  mode='lines', name='Actual Test Data',
                                  line=dict(color='green')))
fig_forecast.add_trace(go.Scatter(x=forecast.index, y=forecast,
                                  mode='lines+markers', name='Forecast',
                                  line=dict(color='red', dash='dot')))
fig_forecast.update_layout(
    title='ARIMA Forecast vs Actuals',
    xaxis_title='Date',
    yaxis_title='Confirmed Cases',
    template='plotly_dark',
    hovermode='x unified',
    height=600
)
fig_forecast.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'Test RMSE: {rmse:.3f}')

# 30-Day Forecast
future_forecast = model_fit.forecast(steps=30)
future_dates = pd.date_range(start=argentina.index[-1], periods=31, freq='D')[1:]

# Enhanced 30-Day Forecast Visualization - CORRECTED VERSION
fig_future = go.Figure()

# Historical data
fig_future.add_trace(go.Scatter(
    x=argentina.index,
    y=argentina['Confirmed'],
    mode='lines',
    name='Historical Data',
    line=dict(color='#636EFA', width=2),
    hovertemplate='Date: %{x|%Y-%m-%d}<br>Cases: %{y:,}'
))

# Forecasted data
fig_future.add_trace(go.Scatter(
    x=future_dates,
    y=future_forecast,
    mode='lines+markers',
    name='30-Day Forecast',
    line=dict(color='#FFA15A', width=3, dash='dot'),
    marker=dict(size=8, symbol='diamond'),
    hovertemplate='Date: %{x|%Y-%m-%d}<br>Forecast: %{y:,.0f}'
))

# Corrected confidence interval using numpy arrays
future_dates_np = future_dates.to_numpy()
fig_future.add_trace(go.Scatter(
    x=np.concatenate([future_dates_np, future_dates_np[::-1]]),
    y=np.concatenate([future_forecast * 0.9, (future_forecast * 1.1)[::-1]]),
    fill='toself',
    fillcolor='rgba(255,161,90,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo='skip',
    name='90% Confidence Interval'
))

fig_future.update_layout(
    title='30-Day COVID-19 Cases Forecast for Argentina',
    xaxis_title='Date',
    yaxis_title='Confirmed Cases',
    template='plotly_dark',
    hovermode='x unified',
    height=700,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=3, label='3m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible=True),
        type='date'
    )
)
fig_future.add_annotation(
    xref='paper', yref='paper',
    x=0.05, y=0.9,
    text=f'Model RMSE: {rmse:.2f}',
    showarrow=False,
    font=dict(size=12, color='white'),
    bgcolor='rgba(0,0,0,0.5)',
    bordercolor='white',
    borderwidth=1
)
fig_future.show()

# Create interactive forecast table
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Cases': future_forecast,
    'Lower Bound': future_forecast * 0.9,
    'Upper Bound': future_forecast * 1.1
})

fig_table = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>Date</b>', '<b>Forecast</b>', '<b>Lower Bound</b>', '<b>Upper Bound</b>'],
        fill_color='#2a3f5f',
        align='center',
        font=dict(color='white', size=12)
    ),
    cells=dict(
        values=[
            forecast_df['Date'].dt.strftime('%Y-%m-%d'),
            forecast_df['Forecasted Cases'].round(0).astype(int).apply(lambda x: f"{x:,}"),
            forecast_df['Lower Bound'].round(0).astype(int).apply(lambda x: f"{x:,}"),
            forecast_df['Upper Bound'].round(0).astype(int).apply(lambda x: f"{x:,}")
        ],
        fill_color='#4a6da7',
        align='center',
        font=dict(color='white', size=11)
    )
)])
fig_table.update_layout(
    title='30-Day COVID-19 Forecast Details',
    template='plotly_dark',
    height=400 + len(forecast_df) * 20,
    margin=dict(l=10, r=10, t=60, b=10)
)
fig_table.show()