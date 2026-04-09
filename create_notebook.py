import nbformat as nbf

nb = nbf.v4.new_notebook()

imports_code = """import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import shap
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)"""

download_code = """tickers = ['AAPL', 'TSLA', 'NFLX']
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(years=2)

df_list = []
for ticker in tickers:
    # Explicitly fetching from yfinance
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Depending on yfinance version, columns could be MultiIndex. Ensure single level
    if isinstance(data.columns, pd.MultiIndex):
        # We only care about the first level if the second is ticker
        data.columns = [c[0] for c in data.columns]
    
    data['Ticker'] = ticker
    df_list.append(data)

df = pd.concat(df_list)
df.reset_index(inplace=True)
df.head()"""

fe_code = """def engineer_features(group):
    group = group.sort_values('Date')
    
    # 1. daily_return
    group['daily_return'] = group['Close'].pct_change()
    
    # 2. volatility_7d
    group['volatility_7d'] = group['daily_return'].rolling(window=7).std()
    
    # 3. volume_spike_ratio
    group['volume_7d_avg'] = group['Volume'].rolling(window=7).mean()
    group['volume_spike_ratio'] = group['Volume'] / group['volume_7d_avg']
    
    # 4. price_gap
    group['prev_close'] = group['Close'].shift(1)
    group['price_gap'] = group['Open'] / group['prev_close'] - 1
    
    group = group.dropna()
    
    # 5. 7-day rolling z-scores
    cols_to_zscore = ['daily_return', 'volatility_7d', 'volume_spike_ratio', 'price_gap']
    for col in cols_to_zscore:
        roll_mean = group[col].rolling(window=7).mean()
        roll_std = group[col].rolling(window=7).std()
        group[f'{col}_zscore'] = (group[col] - roll_mean) / (roll_std + 1e-8)
        
    group = group.dropna()
    return group

df_features = df.groupby('Ticker').apply(engineer_features).reset_index(drop=True)
df_features.head()"""

eda_code = """# Price and Volume Time Series
for ticker in tickers:
    ticker_data = df_features[df_features['Ticker'] == ticker]
    fig, ax = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    ax[0].plot(ticker_data['Date'], ticker_data['Close'], label='Close Price', color='blue')
    ax[0].set_title(f'{ticker} Price Time Series')
    ax[0].legend()
    ax[1].bar(ticker_data['Date'], ticker_data['Volume'], label='Volume', color='orange')
    ax[1].set_title(f'{ticker} Volume')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

# Return distribution
plt.figure()
sns.histplot(data=df_features, x='daily_return', hue='Ticker', kde=True, bins=50)
plt.title('Daily Return Distribution')
plt.show()

# Correlation Heatmap
features = ['daily_return', 'volatility_7d', 'volume_spike_ratio', 'price_gap']
for ticker in tickers:
    plt.figure(figsize=(6, 5))
    ticker_data = df_features[df_features['Ticker'] == ticker]
    sns.heatmap(ticker_data[features].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'{ticker} Feature Correlation')
    plt.show()

# Rolling Volatility
plt.figure()
sns.lineplot(data=df_features, x='Date', y='volatility_7d', hue='Ticker')
plt.title('Rolling 7-Day Volatility')
plt.show()"""

model_code = """features_for_modeling = [
    'daily_return_zscore',
    'volatility_7d_zscore',
    'volume_spike_ratio_zscore',
    'price_gap_zscore'
]

df_anomalies = []

for ticker in tickers:
    ticker_data = df_features[df_features['Ticker'] == ticker].copy()
    X = ticker_data[features_for_modeling].values
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    ticker_data['iso_anomaly'] = iso_forest.fit_predict(X)
    
    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
    ticker_data['lof_anomaly'] = lof.fit_predict(X)
    
    # Ensemble Anomaly: Only flag if both agree (-1 denotes anomaly)
    ticker_data['is_anomaly'] = ((ticker_data['iso_anomaly'] == -1) & (ticker_data['lof_anomaly'] == -1)).astype(int)
    
    df_anomalies.append(ticker_data)

df_final = pd.concat(df_anomalies)

for ticker in tickers:
    ticker_data = df_final[df_final['Ticker'] == ticker]
    anomalies = ticker_data[ticker_data['is_anomaly'] == 1]
    
    plt.figure(figsize=(14, 6))
    plt.plot(ticker_data['Date'], ticker_data['Close'], label='Close Price', alpha=0.7)
    plt.scatter(anomalies['Date'], anomalies['Close'], color='red', label='Anomaly', zorder=5, s=50)
    plt.title(f'{ticker} Price with Flagged Anomalies')
    plt.legend()
    plt.show()

print(df_final.groupby('Ticker')['is_anomaly'].sum())"""

shap_code = """# Explaining the top 5 anomalies for AAPL
ticker_for_shap = 'AAPL'
ticker_data = df_final[df_final['Ticker'] == ticker_for_shap].copy()
X = ticker_data[features_for_modeling]

iso_forest = IsolationForest(contamination=0.02, random_state=42)
iso_forest.fit(X.values)

def score_func(X_in):
    return iso_forest.decision_function(X_in)

background_summary = shap.kmeans(X, 10)
explainer = shap.KernelExplainer(score_func, background_summary)

ticker_data['anomaly_score'] = score_func(X.values)
top_5_anomalies = ticker_data[ticker_data['is_anomaly'] == 1].sort_values('anomaly_score').head(5)

# If no anomalies found, just explain the top 5 lowest scores
if len(top_5_anomalies) == 0:
    top_5_anomalies = ticker_data.sort_values('anomaly_score').head(5)

X_top_5 = top_5_anomalies[features_for_modeling]
shap_values = explainer.shap_values(X_top_5)

for i in range(len(X_top_5)):
    plt.figure()
    
    # Generate the waterfall plot using shap.plots.waterfall
    exp = shap.Explanation(values=shap_values[i], 
                           base_values=explainer.expected_value, 
                           data=X_top_5.iloc[i].values, 
                           feature_names=features_for_modeling)
    shap.plots.waterfall(exp, show=False)
    
    plt.title(f"Anomaly Explanation for {ticker_for_shap} on {top_5_anomalies.iloc[i]['Date'].date()}")
    plt.show()"""

nb.cells = [
    nbf.v4.new_markdown_cell("# Stock Anomaly Detector\\nBuild an anomaly detector using yfinance, Isolation Forest, and SHAP."),
    nbf.v4.new_code_cell(imports_code),
    nbf.v4.new_markdown_cell("## Data Downloading"),
    nbf.v4.new_code_cell(download_code),
    nbf.v4.new_markdown_cell("## Feature Engineering"),
    nbf.v4.new_code_cell(fe_code),
    nbf.v4.new_markdown_cell("## Exploratory Data Analysis"),
    nbf.v4.new_code_cell(eda_code),
    nbf.v4.new_markdown_cell("## Modeling"),
    nbf.v4.new_code_cell(model_code),
    nbf.v4.new_markdown_cell("## Explainability with SHAP"),
    nbf.v4.new_code_cell(shap_code)
]

with open('anomaly_detector.ipynb', 'w') as f:
    nbf.write(nb, f)
