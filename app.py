import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Forex Forecast System", layout="wide")

st.title("📊 Forex Currency Forecasting System")
st.write("Best model is automatically selected per currency.")

# -----------------------------
# LOAD BEST MODELS TABLE
# -----------------------------
@st.cache_data
def load_results():
    return pd.read_csv("best_models.csv")

results_df = load_results()

# -----------------------------
# USER INPUT
# -----------------------------
currency = st.selectbox("Select Currency", results_df["Currency"].unique())

currency_row = results_df[results_df["Currency"] == currency].iloc[0]
best_model = currency_row["Model"]

st.subheader(f"Currency: {currency}")
st.write(f"🏆 Best Model: **{best_model}**")

# -----------------------------
# LOAD MODEL
# -----------------------------
model_path = f"models/{currency}_xgb.pkl"

try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except:
    st.error(f"Model not found: {model_path}")
    st.stop()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Foreign_Exchange_Rates.xls")  # CSV disguised as XLS
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.drop(columns=['Unnamed: 24'], errors='ignore')

    df['Time Serie'] = pd.to_datetime(df['Time Serie'], dayfirst=True)
    df = df.set_index('Time Serie')

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.ffill()

    return df

data = load_data()

# -----------------------------
# COLUMN MAPPING
# -----------------------------
col_map = {
    'AUSTRALIAN DOLLAR': 'AUSTRALIA - AUSTRALIAN DOLLAR/US$',
    'EURO': 'EURO AREA - EURO/US$',
    'NEW ZEALAND DOLLAR': 'NEW ZEALAND - NEW ZEALAND DOLLAR/US$',
    'GREAT BRITAIN POUNDS': 'UNITED KINGDOM - UNITED KINGDOM POUND/US$',
    'BRAZILIAN REAL': 'BRAZIL - REAL/US$',
    'CANADIAN DOLLAR': 'CANADA - CANADIAN DOLLAR/US$',
    'CHINESE YUAN$': 'CHINA - YUAN/US$',
    'HONG KONG DOLLAR': 'HONG KONG - HONG KONG DOLLAR/US$',
    'INDIAN RUPEE': 'INDIA - INDIAN RUPEE/US$',
    'KOREAN WON$': 'KOREA - WON/US$',
    'MEXICAN PESO': 'MEXICO - MEXICAN PESO/US$',
    'SOUTH AFRICAN RAND$': 'SOUTH AFRICA - RAND/US$',
    'SINGAPORE DOLLAR': 'SINGAPORE - SINGAPORE DOLLAR/US$',
    'DANISH KRONE': 'DENMARK - DANISH KRONE/US$',
    'JAPANESE YEN$': 'JAPAN - YEN/US$',
    'MALAYSIAN RINGGIT': 'MALAYSIA - RINGGIT/US$',
    'NORWEGIAN KRONE': 'NORWAY - NORWEGIAN KRONE/US$',
    'SWEDEN KRONA': 'SWEDEN - KRONA/US$',
    'SRILANKAN RUPEE': 'SRI LANKA - SRI LANKAN RUPEE/US$',
    'SWISS FRANC': 'SWITZERLAND - FRANC/US$',
    'NEW TAIWAN DOLLAR': 'TAIWAN - NEW TAIWAN DOLLAR/US$',
    'THAI BAHT': 'THAILAND - BAHT/US$'
}

series_col = col_map[currency]
series = data[series_col].dropna()

# -----------------------------
# FORECAST SETTINGS
# -----------------------------
st.subheader("Forecast Settings")
days = st.slider("Forecast Horizon (days)", 1, 60, 30)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def create_features(df):
    df = df.copy()
    df['lag_1'] = df['y'].shift(1)
    df['lag_7'] = df['y'].shift(7)
    df['lag_30'] = df['y'].shift(30)

    df['rolling_mean_7'] = df['y'].rolling(7).mean()
    df['rolling_std_7'] = df['y'].rolling(7).std()

    df['rolling_mean_30'] = df['y'].rolling(30).mean()
    df['rolling_std_30'] = df['y'].rolling(30).std()

    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year

    return df.dropna()

# -----------------------------
# FORECAST BUTTON
# -----------------------------
if st.button("📈 Predict Future"):

    df_temp = pd.DataFrame(series)
    df_temp.columns = ['y']

    df_feat = create_features(df_temp)
    X = df_feat.drop(columns=['y'])

    # Last known input
    current_input = X.iloc[-1].values.reshape(1, -1)

    preds = []

    for _ in range(days):
        pred = model.predict(current_input)[0]
        preds.append(pred)

        # Update input (simple rolling logic)
        new_row = current_input.flatten()
        new_row = np.roll(new_row, -1)
        new_row[-4] = pred  # update lag_1 approx

        current_input = new_row.reshape(1, -1)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.subheader("📊 Forecast Results")

    forecast_df = pd.DataFrame({
        "Day": np.arange(1, days + 1),
        "Forecast": preds
    })

    # -------- Clean Streamlit Chart --------
    st.subheader("📈 Forecast Chart")
    st.line_chart(forecast_df.set_index("Day"))

    # -------- Matplotlib (Smaller Plot) --------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(forecast_df["Day"], forecast_df["Forecast"], linewidth=2)
    ax.set_title(f"{currency} Forecast", fontsize=8)
    ax.set_xlabel("Days")
    ax.set_ylabel("Exchange Rate")

    #st.pyplot(fig)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.pyplot(fig)

    # -------- Table --------
    st.subheader("📄 Forecast Table")
    st.dataframe(forecast_df)