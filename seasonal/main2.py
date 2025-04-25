import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import jarque_bera
import seaborn as sns

# Load data
df = pd.read_csv('Country Wise Gender.csv')
df.fillna(0, inplace=True)

# Convert all Male/Female columns to int
year_columns = [col for col in df.columns if 'Male' in col or 'Female' in col]
df[year_columns] = df[year_columns].astype(int)

# ------------------- STEP 2: Check Stationarity -------------------
# Step 2.1: Create time series (sum across all countries per year)
years = sorted(list(set([col.split()[0] for col in year_columns])))
total_series = []
for year in years:
    total = df[f"{year} Male"].sum() + df[f"{year} Female"].sum()
    total_series.append(total)

time_series = pd.Series(total_series, index=pd.to_datetime(years))

# Step 2.2: Apply ADF test
print("\n📈 Augmented Dickey-Fuller Test:")
result = adfuller(time_series)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
if result[1] > 0.05:
    print("Result: Time series is not stationary. Applying first differencing.")
    time_series_diff = time_series.diff().dropna()
else:
    print("Result: ✅ Time series is stationary.")
    time_series_diff = time_series

# Optional: Plot differenced time series
fig_stationary = px.line(x=time_series_diff.index, y=time_series_diff.values,
                         title='Stationarized Time Series (Total Population)',
                         labels={'x': 'Year', 'y': 'Differenced Count'})
fig_stationary.update_layout(template='plotly_dark')
fig_stationary.show()

# -------------------- 1. Plotly Stacked Bar Plot (2019) --------------------
fig1 = go.Figure(data=[
    go.Bar(name='2019 Male', x=df['Country of Nationality'], y=df['2019 Male']),
    go.Bar(name='2019 Female', x=df['Country of Nationality'], y=df['2019 Female'])
])
fig1.update_layout(
    barmode='stack',
    title='Stacked Bar Plot - 2019 Male vs Female',
    xaxis_title='Country',
    yaxis_title='Population',
    xaxis_tickangle=-45,
    template='plotly_dark'
)
fig1.show()

# -------------------- 2. Plotly Histogram (2020 Male) --------------------
fig2 = px.histogram(df, x='2020 Male', nbins=10, title='Histogram of 2020 Male Population',
                    labels={'2020 Male': 'Male Population'},
                    color_discrete_sequence=['#636EFA'])
fig2.update_layout(template='plotly_white')
fig2.show()

# -------------------- 3. Plotly Pie Chart (Top 5 by 2020 Female) --------------------
top5 = df.nlargest(5, '2020 Female')
fig3 = px.pie(top5, values='2020 Female', names='Country of Nationality',
              title='Top 5 Countries by 2020 Female Count',
              hole=0.4,
              color_discrete_sequence=px.colors.sequential.RdPu)
fig3.update_traces(textinfo='percent+label')
fig3.show()

# -------------------- 4. Plotly Scatter Matrix (Pairplot style) --------------------
pair_df = df[['2018 Male', '2018 Female', '2019 Male', '2019 Female', '2020 Male', '2020 Female']]
fig4 = px.scatter_matrix(pair_df,
                         dimensions=pair_df.columns,
                         title="Scatter Matrix (Male vs Female 2018-2020)",
                         height=800,
                         color_discrete_sequence=['#00CC96'])
fig4.update_layout(template='plotly_dark')
fig4.show()

# -------------------- 5. Plotly Grouped Bar Chart (2020 Male & Female) --------------------
melted = pd.melt(df[['Country of Nationality', '2020 Male', '2020 Female']],
                 id_vars='Country of Nationality',
                 var_name='Gender', value_name='Count')

fig5 = px.bar(melted, x='Country of Nationality', y='Count', color='Gender',
              barmode='group',
              title='Grouped Bar Chart - 2020 Gender Population by Country',
              height=600)
fig5.update_layout(xaxis_tickangle=-45, template='plotly_white')
fig5.show()


print('--------------------------------------------------------------------------------------')
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib import pyplot as plt

# Reuse this from Step 2
# time_series = pd.Series(total_series, index=pd.to_datetime(years))
# time_series_diff = time_series.diff().dropna()

# -------------------- MA Model --------------------
from statsmodels.tsa.arima.model import ARIMA
ma_model = ARIMA(time_series, order=(0, 1, 1))  # MA(q=1) with differencing
ma_fit = ma_model.fit()
print("✔️ MA Summary:\n", ma_fit.summary())

# -------------------- ARMA Model --------------------
arma_model = ARIMA(time_series_diff, order=(2, 0, 2))  # ARMA(p=2, q=2)
arma_fit = arma_model.fit()
print("✔️ ARMA Summary:\n", arma_fit.summary())

# -------------------- ARIMA Model --------------------
arima_model = ARIMA(time_series, order=(1, 1, 1))  # ARIMA(p,d,q)
arima_fit = arima_model.fit()
print("✔️ ARIMA Summary:\n", arima_fit.summary())

# -------------------- VARMA Model --------------------
# Build a multivariate time series (Male & Female yearly totals)
male_series = pd.Series([df[f'{y} Male'].sum() for y in years], index=pd.to_datetime(years))
female_series = pd.Series([df[f'{y} Female'].sum() for y in years], index=pd.to_datetime(years))
varma_df = pd.concat([male_series, female_series], axis=1)
varma_df.columns = ['Male', 'Female']
varma_df_diff = varma_df.diff().dropna()

varma_model = VARMAX(varma_df_diff, order=(1, 1))
varma_fit = varma_model.fit(disp=False)
print("✔️ VARMA Summary:\n", varma_fit.summary())


# -------------------- Plot Forecast --------------------
forecast_steps = 5

plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Original', marker='o')
plt.plot(arima_fit.predict(start=0, end=len(time_series)-1), label='ARIMA Fitted', linestyle='--')
plt.plot(arima_fit.forecast(steps=forecast_steps), label='ARIMA Forecast', marker='x')
plt.legend()
plt.title("ARIMA Model Forecasting")
plt.grid(True)
plt.tight_layout()
plt.show()
print('----------------------------------------------------------------------------------------')
# -------------------- STEP 4: Parameter Redundancy Check (Plotly Version) --------------------
import plotly.express as px
import pandas as pd

# Collect p-values from each model
redundancy_data = {
    "Model": [],
    "Parameter": [],
    "P-Value": [],
    "Significance": []
}

def collect_pvalues_plot(model_result, model_name):
    for param, pval in model_result.pvalues.items():
        redundancy_data["Model"].append(model_name)
        redundancy_data["Parameter"].append(param)
        redundancy_data["P-Value"].append(pval)
        significance = "Significant ✅" if pval < 0.05 else "Not Significant ⚠️"
        redundancy_data["Significance"].append(significance)

# Apply to each model
collect_pvalues_plot(ma_fit, "MA(1)")
collect_pvalues_plot(arma_fit, "ARMA(2,2)")
collect_pvalues_plot(arima_fit, "ARIMA(1,1,1)")
collect_pvalues_plot(varma_fit, "VARMA(1,1)")

# Convert to DataFrame
pval_df = pd.DataFrame(redundancy_data)

# Create interactive bar chart with Plotly
fig = px.bar(pval_df,
             x="Parameter",
             y="P-Value",
             color="Significance",
             facet_col="Model",
             color_discrete_map={
                 "Significant ✅": "green",
                 "Not Significant ⚠️": "red"
             },
             title="🔍 Parameter Redundancy Check (P-Values)",
             labels={"P-Value": "P-Value", "Parameter": "Model Parameter"},
             height=600)

# Add horizontal line at p=0.05 in each subplot
fig.update_layout(template='plotly_white', showlegend=True)
fig.update_yaxes(showgrid=True, range=[0, 1])

# Add p=0.05 threshold line manually for each facet
for i in range(len(fig.data)):
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(pval_df["Parameter"].unique()) - 0.5,
        y0=0.05,
        y1=0.05,
        line=dict(color="black", dash="dash"),
        xref="x",
        yref="y"
    )

# Show plot
fig.show()
print('---------------------------------------------------------')
# Collect coefficients and standard errors for each model
estimation_data = {
    "Model": [],
    "Parameter": [],
    "Coefficient": [],
    "Standard Error": [],
    "Significance": []
}

def collect_estimates(model_result, model_name):
    for param in model_result.params.index:
        coefficient = model_result.params[param]
        std_err = model_result.bse[param]
        significance = "Significant ✅" if model_result.pvalues[param] < 0.05 else "Not Significant ⚠️"
        estimation_data["Model"].append(model_name)
        estimation_data["Parameter"].append(param)
        estimation_data["Coefficient"].append(coefficient)
        estimation_data["Standard Error"].append(std_err)
        estimation_data["Significance"].append(significance)

# Apply to each model
collect_estimates(ma_fit, "MA(1)")
collect_estimates(arma_fit, "ARMA(2,2)")
collect_estimates(arima_fit, "ARIMA(1,1,1)")
collect_estimates(varma_fit, "VARMA(1,1)")

# Convert to DataFrame
estimation_df = pd.DataFrame(estimation_data)

# Create interactive bar chart with Plotly for parameter coefficients
fig_params = px.bar(estimation_df,
                    x="Parameter",
                    y="Coefficient",
                    color="Significance",
                    facet_col="Model",
                    color_discrete_map={
                        "Significant ✅": "green",
                        "Not Significant ⚠️": "red"
                    },
                    title="🔍 Parameter Estimation (Coefficients)",
                    labels={"Coefficient": "Coefficient", "Parameter": "Model Parameter"},
                    height=600)

# Add error bars for standard errors
for trace in fig_params.data:
    trace.error_y = dict(type="data", array=estimation_df["Standard Error"])

# Show plot
fig_params.update_layout(template='plotly_white')
fig_params.show()

# -------------------- Forecasting --------------------
forecast_steps = 5

plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Original', marker='o')
plt.plot(arima_fit.predict(start=0, end=len(time_series)-1), label='ARIMA Fitted', linestyle='--')
plt.plot(arima_fit.forecast(steps=forecast_steps), label='ARIMA Forecast', marker='x')
plt.legend()
plt.title("ARIMA Model Forecasting")
plt.grid(True)
plt.tight_layout()
plt.show()
print('----------------------------------------------------------------')
# Function to plot residuals and perform tests
# Function to plot residuals and perform tests
# Function to plot residuals and perform tests
def residual_analysis(model, time_series, model_name):
    # Calculate residuals
    residuals = model.resid
    print(f"\n📊 Residual Analysis for {model_name}:")

    # Plot residuals
    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.title(f"Residuals of {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Histogram of residuals
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=20)
    plt.title(f"Histogram of Residuals: {model_name}")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # ACF and PACF plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Calculate the maximum number of lags for both ACF and PACF
    max_lags_acf = min(len(residuals) - 1, 40)
    max_lags_pacf = min(len(residuals) // 2, 40)

    # Plot ACF
    plot_acf(residuals, lags=max_lags_acf, ax=axes[0])
    axes[0].set_title(f"ACF of Residuals: {model_name}")

    # Plot PACF
    plot_pacf(residuals, lags=max_lags_pacf, ax=axes[1])
    axes[1].set_title(f"PACF of Residuals: {model_name}")

    plt.tight_layout()
    plt.show()

    # Jarque-Bera test for normality
    jb_result = jarque_bera(residuals)
    jb_stat = jb_result[0]
    jb_p_value = jb_result[1]
    print(f"Jarque-Bera Test for Normality (p-value): {jb_p_value}")
    if jb_p_value > 0.05:
        print("Residuals are normally distributed ✅")
    else:
        print("Residuals are not normally distributed ⚠️")

    # Plotting Q-Q plot for normality check
    from scipy import stats
    plt.figure(figsize=(12, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of Residuals: {model_name}")
    plt.tight_layout()
    plt.show()

# Apply residual analysis to each model

# Residual analysis for MA model
residual_analysis(ma_fit, time_series, "MA(1)")

# Residual analysis for ARMA model
residual_analysis(arma_fit, time_series, "ARMA(2,2)")

# Residual analysis for ARIMA model
residual_analysis(arima_fit, time_series, "ARIMA(1,1,1)")
print('----------------------------------------------------------------------------')
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# Forecasting Steps
forecast_steps = 12  # Forecast the next 12 time points

# Fit ARIMA model
arima_model = ARIMA(time_series, order=(1, 1, 1))
arima_fit = arima_model.fit()

# Get residuals from ARIMA model
residuals = arima_fit.resid

# Fit GARCH model on residuals
garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit()

# Forecast future values and volatility with ARIMA
arima_forecast = arima_fit.get_forecast(steps=forecast_steps)  # Use get_forecast to get mean forecast

# The forecasted values are returned as a Pandas Series in `arima_forecast`
forecasted_values = arima_forecast.predicted_mean  # ARIMA forecasted mean

# Forecasted volatility using GARCH model
garch_volatility_forecast = garch_fit.forecast(horizon=forecast_steps)
forecasted_volatility = garch_volatility_forecast.variance.values[-1, :]  # GARCH forecasted variance

# Ensure that the forecasted volatility array matches the forecast length
# Repeat volatility forecast values to match forecast_steps (if necessary (not much mandatory))
forecasted_volatility = np.tile(forecasted_volatility, forecast_steps // len(forecasted_volatility) + 1)[:forecast_steps]

# Plot ARIMA-GARCH forecasts
plt.figure(figsize=(12, 6))
plt.plot(time_series, label="Actual")
plt.plot(range(len(time_series), len(time_series) + forecast_steps), forecasted_values, label="ARIMA Forecast", color="orange")
plt.fill_between(range(len(time_series), len(time_series) + forecast_steps),
                 forecasted_values - 1.96 * np.sqrt(forecasted_volatility),
                 forecasted_values + 1.96 * np.sqrt(forecasted_volatility),
                 color='orange', alpha=0.3, label="Forecast Interval")
plt.title("ARIMA-GARCH Forecasting")
plt.legend()
plt.show()