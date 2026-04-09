import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Anomaly Detector", layout="wide")
st.title("📈 Stock Anomaly Detector")
st.write("Detect stock price and volume anomalies for AAPL, TSLA, and NFLX using Isolation Forest and LOF.")

@st.cache_data(ttl=3600)
def load_data_and_detect_anomalies():
    tickers = ['AAPL', 'TSLA', 'NFLX']
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=2)
    
    df_list = []
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        data['Ticker'] = ticker
        df_list.append(data)
        
    df = pd.concat(df_list)
    df.reset_index(inplace=True)
    
    def engineer_features(group):
        group = group.sort_values('Date')
        group['daily_return'] = group['Close'].pct_change()
        group['volatility_7d'] = group['daily_return'].rolling(window=7).std()
        group['volume_7d_avg'] = group['Volume'].rolling(window=7).mean()
        group['volume_spike_ratio'] = group['Volume'] / group['volume_7d_avg']
        group['prev_close'] = group['Close'].shift(1)
        group['price_gap'] = group['Open'] / group['prev_close'] - 1
        group = group.dropna()
        
        cols_to_zscore = ['daily_return', 'volatility_7d', 'volume_spike_ratio', 'price_gap']
        for col in cols_to_zscore:
            roll_mean = group[col].rolling(window=7).mean()
            roll_std = group[col].rolling(window=7).std()
            group[f'{col}_zscore'] = (group[col] - roll_mean) / (roll_std + 1e-8)
        return group.dropna()

    df_features = df.groupby('Ticker').apply(engineer_features).reset_index(drop=True)
    
    features_for_modeling = ['daily_return_zscore', 'volatility_7d_zscore', 'volume_spike_ratio_zscore', 'price_gap_zscore']
    
    df_anomalies = []
    for ticker in tickers:
        ticker_data = df_features[df_features['Ticker'] == ticker].copy()
        X = ticker_data[features_for_modeling].values
        
        iso = IsolationForest(contamination=0.02, random_state=42)
        ticker_data['iso_anomaly'] = iso.fit_predict(X)
        ticker_data['anomaly_score'] = iso.decision_function(X)
        
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
        ticker_data['lof_anomaly'] = lof.fit_predict(X)
        
        ticker_data['is_anomaly'] = ((ticker_data['iso_anomaly'] == -1) & (ticker_data['lof_anomaly'] == -1)).astype(int)
        df_anomalies.append(ticker_data)
        
    return pd.concat(df_anomalies), features_for_modeling

with st.spinner("Loading data and computing models..."):
    df_final, features_for_modeling = load_data_and_detect_anomalies()

st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Select Ticker", ['AAPL', 'TSLA', 'NFLX'])

min_date = df_final['Date'].min().date()
max_date = df_final['Date'].max().date()
start_d, end_d = st.sidebar.slider("Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

ticker_data = df_final[(df_final['Ticker'] == selected_ticker) &
                       (df_final['Date'].dt.date >= start_d) &
                       (df_final['Date'].dt.date <= end_d)].copy()

st.subheader(f"{selected_ticker} Price Chart with Anomalies")

fig = go.Figure()

fig.add_trace(go.Candlestick(x=ticker_data['Date'],
                open=ticker_data['Open'],
                high=ticker_data['High'],
                low=ticker_data['Low'],
                close=ticker_data['Close'],
                name='Candlestick'))

anomalies = ticker_data[ticker_data['is_anomaly'] == 1]
fig.add_trace(go.Scatter(x=anomalies['Date'],
                         y=anomalies['Close'],
                         mode='markers',
                         name='Anomaly',
                         marker=dict(color='red', size=12, symbol='x', line=dict(width=2, color='DarkSlateGrey'))))

fig.update_layout(xaxis_rangeslider_visible=False, height=550, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("SHAP Values: Understand the Anomalies")
st.write("Select an anomaly date from the dropdown to see why the model flagged it.")

if len(anomalies) > 0:
    anomaly_dates = anomalies['Date'].dt.date.tolist()
    selected_date = st.selectbox("Select Anomaly Date to Explain:", anomaly_dates)
    
    if st.button("Generate SHAP Explanation"):
        with st.spinner("Generating Explainer..."):
            X_all = df_final[df_final['Ticker'] == selected_ticker][features_for_modeling]
            iso = IsolationForest(contamination=0.02, random_state=42)
            iso.fit(X_all.values)
            
            def score_func(X_in):
                return iso.decision_function(X_in)
                
            background_summary = shap.kmeans(X_all, 10)
            explainer = shap.KernelExplainer(score_func, background_summary)
            
            target_row = anomalies[anomalies['Date'].dt.date == selected_date]
            X_target = target_row[features_for_modeling]
            
            shap_values = explainer.shap_values(X_target)
            
            fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
            exp = shap.Explanation(values=shap_values[0], 
                                   base_values=explainer.expected_value, 
                                   data=X_target.iloc[0].values, 
                                   feature_names=features_for_modeling)
            shap.plots.waterfall(exp, show=False)
            st.pyplot(fig_shap)
else:
    st.info("No anomalies detected in this date range.")
