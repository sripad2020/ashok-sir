# ==============================================
# Step 1: Data Visualization and Preprocessing
# ==============================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('General Data 2014-2020.csv')
df['year'] = pd.to_datetime(df['year'], format='%Y')
df.set_index('year', inplace=True)

# Visualize all time series
fig = make_subplots(rows=7, cols=4, subplot_titles=df.columns)
for i, col in enumerate(df.columns):
    row = i // 4 + 1
    col_num = i % 4 + 1
    fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col), row=row, col=col_num)
fig.update_layout(height=1600, width=1200, title_text="All Time Series Visualization", showlegend=False)
fig.show()

# Select main metric and visualize
main_metric = 'noftaii'
fig = px.line(df, x=df.index, y=main_metric, title=f'{main_metric} Over Time')
fig.show()

# Handle missing values
print("Missing values:\n", df.isnull().sum())
df.fillna(method='ffill', inplace=True)

# ==============================================
# Step 2: Make the Dataset Stationary
# ==============================================

def test_stationarity(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    return dftest[1]

p_value = test_stationarity(df[main_metric])
if p_value > 0.05:
    print("Series is not stationary")
    # First difference
    df[f'{main_metric}_diff'] = df[main_metric].diff().dropna()
    p_value_diff = test_stationarity(df[f'{main_metric}_diff'].dropna())
    if p_value_diff > 0.05:
        print("First difference not sufficient, trying second difference")
        df[f'{main_metric}_diff2'] = df[f'{main_metric}_diff'].diff().dropna()
        test_stationarity(df[f'{main_metric}_diff2'].dropna())
else:
    print("Series is stationary")

# ==============================================
# Step 3: Finding Models (ARIMA)
# ==============================================

# ACF/PACF analysis
max_allowed_lags = int(len(df[main_metric].dropna()) * 0.5)
nlags = min(20, max_allowed_lags - 1)
print(f"Using {nlags} lags for ACF/PACF calculation")

acf_values = acf(df[main_metric].dropna(), nlags=nlags)
pacf_values = pacf(df[main_metric].dropna(), nlags=nlags)

fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF', 'PACF'))
fig.add_trace(go.Bar(x=np.arange(nlags+1), y=acf_values, name='ACF'), row=1, col=1)
fig.add_trace(go.Bar(x=np.arange(nlags+1), y=pacf_values, name='PACF'), row=1, col=2)
conf_int = 1.96 / np.sqrt(len(df[main_metric].dropna()))
for col in [1, 2]:
    fig.add_shape(type='line', x0=-0.5, x1=nlags+0.5, y0=conf_int, y1=conf_int,
                 line=dict(color='gray', dash='dash'), row=1, col=col)
    fig.add_shape(type='line', x0=-0.5, x1=nlags+0.5, y0=-conf_int, y1=-conf_int,
                 line=dict(color='gray', dash='dash'), row=1, col=col)
fig.update_layout(height=400, width=900, showlegend=False,
                 title_text=f'ACF/PACF Plots for {main_metric}')
fig.show()

# ==============================================
# Step 4: Parameter Redundancy
# Step 5: Parameter Estimation
# ==============================================

def evaluate_arima_model(X, arima_order):
    model = ARIMA(X, order=arima_order)
    model_fit = model.fit()
    return model_fit.aic

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    aic = evaluate_arima_model(dataset, order)
                    if aic < best_score:
                        best_score, best_cfg = aic, order
                    print('ARIMA%s AIC=%.3f' % (order,aic))
                except:
                    continue
    print('Best ARIMA%s AIC=%.3f' % (best_cfg, best_score))
    return best_cfg

p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)
best_order = evaluate_models(df[main_metric], p_values, d_values, q_values)

best_model = ARIMA(df[main_metric], order=best_order)
best_model_fit = best_model.fit()
print(best_model_fit.summary())

# ==============================================
# Step 6: Residual Analysis
# ==============================================

residuals = pd.DataFrame(best_model_fit.resid).dropna()

# Ljung-Box Test
max_possible_lags = min(10, len(residuals)-1)
if max_possible_lags > 0:
    lb_test = acorr_ljungbox(residuals, lags=[max_possible_lags])
    print("\nLjung-Box Test Results:")
    print(f"Test Statistic: {lb_test.lb_stat.values[0]}")
    print(f"p-value: {lb_test.lb_pvalue.values[0]}")
    if lb_test.lb_pvalue.values[0] < 0.05:
        print("Significant autocorrelation in residuals")
    else:
        print("Residuals appear white noise")

# Residual plots
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=residuals.index, y=residuals[0], mode='lines', name='Residuals'), row=1, col=1)
fig.add_trace(go.Histogram(x=residuals[0], name='Distribution'), row=2, col=1)
fig.update_layout(title_text='Residual Analysis')
fig.show()

# ==============================================
# Step 7: Forecasting and Further Analysis
# ==============================================

# Forecasting
forecast_steps = 3
forecast = best_model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='YS')[1:]
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[main_metric], name='Actual'))
fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, name='Forecast'))
fig.add_trace(go.Scatter(x=forecast_index, y=forecast_ci.iloc[:,0], fill=None, mode='lines', line_color='gray'))
fig.add_trace(go.Scatter(x=forecast_index, y=forecast_ci.iloc[:,1], fill='tonexty', mode='lines', line_color='gray'))
fig.update_layout(title=f'{forecast_steps}-Year Forecast for {main_metric}')
fig.show()

# ARIMA-GARCH model
garch_model = arch_model(best_model_fit.resid.dropna()*100, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit()
print(garch_fit.summary())

fig = px.line(x=residuals.index, y=garch_fit.conditional_volatility,
              title='Conditional Volatility from GARCH Model')
fig.show()