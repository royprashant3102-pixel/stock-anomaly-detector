<div align="center">
  <h1>📈 Stock Anomaly Detector</h1>
  <p>Detect unusual market behavior using Machine Learning and Explainable AI.</p>

  [![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-Native-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

## 📖 Description
The **Stock Anomaly Detector** is an interactive web application designed to automatically identify anomalous behavior in financial time-series data. By combining statistical feature engineering (volatility, price gaps, volume spikes) with unsupervised Machine Learning (Isolation Forest and Local Outlier Factor), the app flags high-risk trading days and explains the reasoning behind every flag using **SHAP** values.

## 🚀 Live Demo
Experience the live application here:  
**[👉 Stock Anomaly Detector App](https://stock-anomaly-detector-prashant.streamlit.app/)**

---

## ✨ Features
- **Real-Time Data Ingestion:** Instantly fetches the last 2 years of OHLCV daily data for major tech tickers (AAPL, TSLA, NFLX).
- **Interactive UI:** Control the evaluated date range using intuitive sliders and dropdown controls.
- **Dynamic Visualizations:** Beautiful, zoomable Plotly candlestick charts clearly highlighting anomaly dates with red markers.
- **Explainable AI (XAI):** Dive into any individual anomaly using SHAP XAI Waterfall diagrams to understand exactly what market conditions triggered the model.
- **Ensemble Detection:** Leverages agreement between Isolation Forest and LOF algorithms for high-confidence outlier detection.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** Streamlit
- **Data Manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn (Isolation Forest, LOF)
- **Explainability:** SHAP (KernelExplainer)
- **Data Source:** yfinance
- **Visualization:** Plotly, Matplotlib, Seaborn

---

## 💻 Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/royprashant3102-pixel/stock-anomaly-detector.git
   cd stock-anomaly-detector
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Usage Instructions

To run the application locally, execute the following command in your terminal:
```bash
python -m streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 🧠 How It Works
1. **Feature Engineering:** We calculate a moving 7-day rolling Z-score across four distinct metrics: Daily Return, Volatility, Volume Spike Ratio, and Price Gaps.
2. **Modeling:** The data is fed into an `IsolationForest` to isolate outliers based on decision tree path lengths, alongside a `LocalOutlierFactor` (LOF) evaluating local dataset density.
3. **Thresholding:** Anomalies are flagged exclusively when both unsupervised models agree, utilizing a rigorous 2% `contamination` threshold.
4. **SHAP Interpretation:** A `shap.KernelExplainer` unboxes the model to output a deterministic waterfall visualization assigning feature impact values per anomaly.

---

## 📁 Project Structure

```text
📦 stock-anomaly-detector
 ┣ 📜 app.py               # Main Streamlit application configuration
 ┣ 📜 create_notebook.py   # Script to auto-generate the research pipeline 
 ┣ 📜 anomaly_detector.ipynb # Full EDA and model playground notebook
 ┣ 📜 requirements.txt     # Python package dependencies
 ┗ 📜 README.md            # Project documentation
```

---

## 📸 Screenshots

*App Dashboard - Candlestick chart overlaid with Isolation Forest anomalies.*
> `[Placeholder: Add Application Screenshot here]`

*SHAP Dashboard - Understanding why a day was flagged.*
> `[Placeholder: Add SHAP Plot Screenshot here]`

---

## 🔮 Future Improvements
- [ ] Incorporate a broader set of ticker symbols encompassing the S&P 500.
- [ ] Add an interactive sidebar slider to tweak the model `contamination` rate without touching code.
- [ ] Implement short-term (intra-day) anomaly detection utilizing trailing 15-minute intervals.
- [ ] Export flagged anomalies to a `.csv` direct download.

---

## 🤝 Contributing Guidelines
Contributions are welcome! If you have a feature idea or find a bug, please feel free to:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author
**Prashant Roy**
* GitHub: [@royprashant3102-pixel](https://github.com/royprashant3102-pixel)