# 🌍 Forex Currency Prediction System (End-to-End Machine Learning Project)

## 📌 Project Overview

This project is an end-to-end machine learning system for forecasting foreign exchange (Forex) currency rates against the US Dollar.

It includes:
- Data preprocessing and feature engineering
- Multiple ML/DL models (XGBoost, LSTM, Prophet, ARIMA)
- Model comparison and best-model selection per currency
- Interactive Streamlit web application
- Dockerized deployment for production use

The goal is to simulate a real-world ML production pipeline.

---

## 🎯 Objectives

- Analyse historical Forex exchange rate data
- Build time-series features for forecasting
- Train multiple predictive models
- Compare model performance across currencies
- Select the best model per currency
- Deploy a Streamlit forecasting application
- Containerize the system using Docker

---

## 📊 Dataset

- File: `Foreign_Exchange_Rates.xls` (converted to CSV for processing)
- Time period: 2000 – 2019
- Frequency: Daily exchange rates
- Target: Multiple currency exchange rates vs USD

Currencies include:
EUR, GBP, JPY, INR, CAD, CHF, AUD, and more.

---

## 🧹 Data Preprocessing

The dataset was cleaned and transformed as follows:

- Converted `Time Serie` to datetime format (`dayfirst=True`)
- Set datetime as index
- Converted all currency columns to numeric values
- Handled missing values using forward fill
- Removed unnecessary columns (`Unnamed: 0`, etc.)

---

## 🧠 Feature Engineering

The following features were created for time-series modeling:

### 📉 Lag Features
- lag_1
- lag_7
- lag_30

### 📊 Rolling Statistics
- rolling_mean_7
- rolling_std_7
- rolling_mean_30
- rolling_std_30

### 📅 Time Features
- day_of_week
- month
- year

---

## 🤖 Machine Learning Models Used

Multiple models were evaluated per currency:

### 📈 Machine Learning Models
- XGBoost
- LightGBM

### 🧠 Deep Learning Models
- LSTM (Long Short-Term Memory)

### 📊 Statistical Models
- ARIMA / SARIMA
- Facebook Prophet

---

## 🏆 Model Selection Strategy

- Each currency is modelled independently
- Data split: train/test (last 60 days as test set)
- Evaluation metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)

The best-performing model for each currency was selected and saved.

---

## 💾 Model Storage

Each trained model is saved as a pickle file:


models/
│── EURO_xgb.pkl
│── INR_lstm.pkl
│── JPY_xgb.pkl
│── GBP_xgb.pkl
│── ...


Each file corresponds to the best model for that currency.

---

## 🌐 Streamlit Web Application

The Streamlit app provides an interactive forecasting interface.

### Features:
- Currency selection dropdown
- Forecast horizon input (e.g. 30 days)
- Dynamic model loading per currency
- Forecast visualization chart
- Prediction output table

### Run locally:
```bash
streamlit run app.py
🐳 Docker Deployment

The application is fully containerized using Docker.

Build Docker image:
docker build -t forex-app .
Run container:
docker run -p 8501:8501 forex-app
Access application:
http://localhost:8501


📁 Project Structure
Forex Currency Prediction/
│
├── app.py
├── analysis.ipynb
├── Foreign_Exchange_Rates.csv
├── requirements.txt
├── Dockerfile
│
├── models/
│   ├── AUSTRALIA_xgb.pkl
│   ├── EURO_xgb.pkl
│   ├── INR_lstm.pkl
│   └── ...
│
└── README.md



📈 Results Summary
XGBoost performed best for most currencies
LSTM performed better for highly volatile currencies
Prophet and ARIMA performed well on trend-based currencies
Per-currency model selection significantly improved accuracy
🔍 Key Insights
No single model performs best for all currencies
Feature engineering has major impact on accuracy
LSTM improves performance for nonlinear time-series patterns
Ensemble or hybrid approach is beneficial in financial forecasting
🚀 Future Improvements
Add ensemble learning (stacking models)
Implement attention-based LSTM/Transformer models
Add prediction confidence intervals
Automate retraining pipeline
Deploy on cloud (AWS / Azure / Streamlit Cloud)

👨‍💻 Author: Farhaan

Focus: Time-Series Forecasting, Machine Learning, and Deployment

📌 License

This project is for academic and educational purposes only.