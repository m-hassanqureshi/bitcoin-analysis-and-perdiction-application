import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objects as go
from PIL import Image 
from io import StringIO
import joblib
from xgboost import XGBRegressor
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# ---- Page Setup ----
icon= Image.open("icon.png")  
st.set_page_config(
    page_title="Bitcoin EDA Dashboard",
    layout="wide",
    page_icon=icon
)
# --- Custom CSS Styling ---
# --- Custom Modern Flutter-Inspired Styling ---
st.markdown("""
<style>
body, .stApp {
    background-color: #ffffff !important;
    color: #333333 !important;
    font-family: 'Poppins', sans-serif;
}

.stCard {
    background: rgba(255,105,180,0.05);
    box-shadow: 0 8px 32px 0 rgba(255,105,180,0.15);
    backdrop-filter: blur(6px);
    border-radius: 18px;
    border: 1px solid rgba(255,105,180,0.25);
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 2rem;
    transition: box-shadow 0.3s;
}
.stCard:hover {
    box-shadow: 0 16px 40px 0 rgba(255,105,180,0.35);
}
.stCard h3 {
    margin-top: 0;
    color: #d272ff;
    font-weight: 700;
    letter-spacing: 0.02em;
}

h1, h2, h3, h4, h5, h6 {
    color: #d272ff !important;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 1.2rem;
}

[data-testid="stSidebar"] {
    background: rgba(255,105,180,0.05);
    color: #333333 !important;
    padding: 20px;
    border-right: 1px solid rgba(255,105,180,0.2);
}

.stDataFrame, .stTable {
    background-color: rgba(255,105,180,0.08) !important;
    color: #333333 !important;
    border-radius: 12px;
    font-size: 0.95rem;
    box-shadow: 0 4px 12px rgba(255,105,180,0.2);
    padding: 10px;
}

[data-testid="stMetric"] {
    background: rgba(255,105,180,0.08);
    border-radius: 12px;
    padding: 15px 20px;
    color: #d272ff !important;
    box-shadow: 0 4px 10px rgba(255,105,180,0.2);
    text-align: center;
}

.stButton>button {
    background: linear-gradient(to right, #ff92c2, #d272ff);
    color: white;
    border-radius: 10px;
    font-weight: bold;
    border: none;
    padding: 0.6em 1.8em;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 2px 8px rgba(255,105,180,0.3);
}
.stButton>button:hover {
    background: linear-gradient(to right, #d272ff, #ff4785);
    color: white;
    transform: scale(1.05);
}

[data-testid="stDownloadButton"] {
    background: #fff0f7;
    color: #d272ff;
    border-radius: 8px;
    border: 1px solid #d272ff;
    font-weight: 600;
    padding: 0.5em 1.2em;
}
[data-testid="stDownloadButton"]:hover {
    background: #d272ff;
    color: white;
}

input, select, textarea {
    background: rgba(255,105,180,0.08) !important;
    color: #333333 !important;
    border-radius: 8px !important;
    border: 1px solid #d272ff !important;
    padding: 0.5em;
}

.stAlert {
    background: rgba(255,105,180,0.07) !important;
    color: #d272ff !important;
    border-left: 5px solid #d272ff !important;
    border-radius: 8px;
    padding: 10px 15px;
    box-shadow: 0 2px 10px rgba(255,105,180,0.15);
}

.js-plotly-plot .plotly {
    background-color: rgba(255,105,180,0.07) !important;
    border-radius: 8px;
    padding: 10px;
}

hr {
    border: none;
    height: 1px;
    background: #d272ff;
    margin: 20px 0;
}

.fab {
    position: fixed;
    bottom: 30px;
    right: 30px;
    background: #d272ff;
    color: white;
    border: none;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    font-size: 28px;
    box-shadow: 0 4px 10px rgba(255,105,180,0.3);
    cursor: pointer;
    transition: 0.3s ease;
    z-index: 9999;
}
.fab:hover {
    background: #ff4785; 
    color: white;
    transform: scale(1.1);
}
#MainMenu {visibility: hidden;} 
</style> 
<a href="#bitcoin-analysis-and-prediction-app">  
    <button class="fab" title="Back to Top">↑</button>
</a>
""", unsafe_allow_html=True) 

# ---- Sidebar Branding ---- 
logo=Image.open("logo.png")
st.sidebar.image(logo, use_container_width=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("bitcoin.csv")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.sort_values('Date', inplace=True)
    df['Daily Return'] = df['Close'].pct_change()
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    df['Daily Range'] = df['High'] - df['Low']
    df['Price Change %'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA30'] = df['Close'].rolling(30).mean()
    df['MA90'] = df['Close'].rolling(90).mean()
    return df

df = load_data()

# Sidebar - Filter date range
st.sidebar.header("Filter by Date")
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
df = df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]

# --- Basic Understanding and Cleaning ---
st.header("🥮 Bitcoin Analysis and Prediction App")
st.write("**Date Range:**", f"{df['Date'].min().date()} to {df['Date'].max().date()}") 
st.write("**Original Data Set Overview :**")
df_1=pd.read_csv("bitcoin.csv")
st.write(df_1.head(5))
st.write("**Total Records:**", len(df))
st.write("**Extended Data Set Overview:**")
st.write(df.head(10))
st.write("**Missing Values:**")
st.dataframe(df.isnull().sum())
st.write("**Data Types:**")
st.dataframe(df.dtypes) 
st.write("**Decriptive Stats:**")
st.write(df.describe())
st.write("**Data Set Info:**")
buffer = StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.code(info_str, language='text')

# --- Trend Analysis ---
st.header("📈 Trend Analysis Over Time") 

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'))
fig1.update_layout(title="Bitcoin Close Price Over Time", xaxis_title="Date", yaxis_title="Close Price")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Moving Averages")
st.line_chart(df.set_index('Date')[['MA7', 'MA30', 'MA90']])

monthly_avg = df.resample('ME', on='Date')['Close'].mean()
st.subheader("Monthly Average Close Price")
st.line_chart(monthly_avg)

yearly_avg = df.resample('YE', on='Date')['Close'].mean()
st.subheader("Yearly Average Close Price")
st.bar_chart(yearly_avg)

# --- Volatility ---
st.header("📊 Volatility and Price Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df['Daily Range'], bins=50, kde=True, ax=ax2)
ax2.set_title("Distribution of Daily Price Range (High - Low)")
st.pyplot(fig2)

st.subheader("Top Volatility Days")
st.dataframe(df.nlargest(5, 'Daily Range')[['Date', 'High', 'Low', 'Daily Range']])

fig3, ax3 = plt.subplots()
sns.boxplot(x=df['Close'], ax=ax3)
ax3.set_title("Boxplot for Close Price")
st.pyplot(fig3)

# --- Comparative Insights ---
st.header("🔁 Comparative and Relative Insights")
st.subheader("Days Closed Higher Than Open")
st.write((df['Close'] > df['Open']).sum(), "days")

corr = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap']].corr()
fig4, ax4 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
ax4.set_title("Correlation Matrix")
st.pyplot(fig4)

# --- Technical Metrics ---
st.header("🧠 Technical Metrics")
st.subheader("Top 5 Positive and Negative Daily Returns")
st.write("**Top Gains:**")
st.dataframe(df.nlargest(5, 'Daily Return')[['Date', 'Daily Return']])
st.write("**Top Losses:**")
st.dataframe(df.nsmallest(5, 'Daily Return')[['Date', 'Daily Return']])

# --- Data Overview ---
# --- Time-Based Grouping ---
st.header("🗓️ Time-Based Grouping")
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
monthly_return = df.groupby(['Year', 'Month'])['Daily Return'].mean().unstack() 
st.subheader("Average Monthly Return")
st.dataframe(monthly_return.style.format("{:.2%}"))

yearly_return = df.groupby('Year')['Daily Return'].mean()
st.subheader("Average Yearly Return")
st.bar_chart(yearly_return)

# --- Volume and Marketcap ---
st.header("📉 Volume and Market Capitalization")
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=df['Date'], y=df['Marketcap'], name='Marketcap'))
fig6.update_layout(title="Marketcap Over Time", xaxis_title="Date")
st.plotly_chart(fig6, use_container_width=True)

fig7 = go.Figure()
fig7.add_trace(go.Scatter(x=df['Date'], y=df['Volume'], name='Volume'))
fig7.update_layout(title="Volume Over Time")
st.plotly_chart(fig7, use_container_width=True)

st.subheader("Zero Volume Days") 
st.dataframe(df[df['Volume'] == 0][['Date', 'Volume']])

# --- Anomalies ---
st.header("🧩 Anomalies and Interesting Events")
spikes = df.nlargest(3, 'Daily Return')
drops = df.nsmallest(3, 'Daily Return')

fig8 = go.Figure()
fig8.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', mode='lines'))
for row in spikes.itertuples():
    fig8.add_annotation(x=row.Date, y=row.Close, text="Spike ↑", showarrow=True, arrowhead=1, arrowcolor="green")
for row in drops.itertuples():
    fig8.add_annotation(x=row.Date, y=row.Close, text="Drop ↓", showarrow=True, arrowhead=1, arrowcolor="red")
st.plotly_chart(fig8, use_container_width=True)

# --- Feature Engineering ---
st.header("📌 Feature Engineering")
st.subheader("New Columns")
st.write("- Daily Range ")
st.write("- Price Change % ")
st.write("- Rolling Averages (MA7, MA30, MA90)")
display_df = df[['Date', 'Daily Range', 'Price Change %', 'MA7', 'MA30', 'MA90']].copy()
display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
# --- User-Driven Exploration ---
st.header("🔍 Interactive Exploration")

# 1. Custom Column Selector for DataFrame
st.subheader("Custom Data Table")
columns = st.multiselect(
    "Select columns to display",
    options=df.columns.tolist(),
    default=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
)
st.dataframe(df[columns].tail(20))

# 2. Candlestick Chart

import plotly.express as px
fig_candle = go.Figure(data=[go.Candlestick(
x=df['Date'],
open=df['Open'],
high=df['High'],
low=df['Low'],
close=df['Close'],
increasing_line_color='green',
decreasing_line_color='red')])
fig_candle.update_layout(title="Bitcoin Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_candle, use_container_width=True)

# 3. Correlation Explorer
st.subheader("Correlation Explorer")
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("X Axis", options=df.select_dtypes(include=np.number).columns, index=0)
with col2:
    y_axis = st.selectbox("Y Axis", options=df.select_dtypes(include=np.number).columns, index=1)
if x_axis != y_axis:
    fig_corr = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig_corr, use_container_width=True)

# 4. Download Data
st.subheader("Download Filtered Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download as CSV", 
    data=csv,
    file_name='filtered_bitcoin_data.csv',
    mime='text/csv'
) 


st.header("🔮 Bitcoin Closing Price Prediction")

@st.cache_resource
def load_or_train_model():
    if os.path.exists("bitcoin_model_final.pkl"):
        model = joblib.load("bitcoin_model_final.pkl")
    else:
        df = pd.read_csv("bitcoin.csv")
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date')
        df = df[df['Volume'] != 0]

        # Feature Engineering
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Weekday'] = df['Date'].dt.weekday
        df['HL_Range'] = df['High'] - df['Low']
        df['OC_Change'] = df['Close'] - df['Open']
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_14'] = df['Close'].rolling(window=14).mean()
        df['Volatility_7'] = df['Close'].rolling(window=7).std()
        df['Lag_1'] = df['Close'].shift(1)
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()

        features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap',
            'Day', 'Month', 'Year', 'Weekday',
            'HL_Range', 'OC_Change', 'MA_7', 'MA_14', 'Volatility_7', 'Lag_1'
        ]
        X = df[features]
        y = df['Target']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # GridSearchCV
        param_grid = {
            'n_estimators': [300, 500],
            'learning_rate': [0.01, 0.05],
            'max_depth': [5, 7, 9],
            'subsample': [0.7, 1.0],
            'colsample_bytree': [0.7, 1.0]
        }

        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X, y)

        best_model = grid.best_estimator_
        joblib.dump(best_model, "bitcoin_model_final.pkl")
        model = best_model

    return model

# Load model
model = load_or_train_model()

# Load and prepare data
st.subheader("📊 Prediction")
df = pd.read_csv("bitcoin.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')
df = df[df['Volume'] != 0]
df['Day'] = df['Date'].dt.day

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Weekday'] = df['Date'].dt.weekday
df['HL_Range'] = df['High'] - df['Low']
df['OC_Change'] = df['Close'] - df['Open']
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_14'] = df['Close'].rolling(window=14).mean()
df['Volatility_7'] = df['Close'].rolling(window=7).std()
df['Lag_1'] = df['Close'].shift(1)
df = df.dropna()

features = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap',
    'Day', 'Month', 'Year', 'Weekday',
    'HL_Range', 'OC_Change', 'MA_7', 'MA_14', 'Volatility_7', 'Lag_1'
]
latest = df[features].iloc[-1:]
pred = model.predict(latest)[0]
st.metric("Tomorrow's Predicted Close Price(Based on Data Set Date Range)", f"${pred:.2f}")

# Plotting
st.subheader("📉 Recent Close Price Trend")
fig, ax = plt.subplots(figsize=(12, 4))
df.tail(60).plot(x='Date', y='Close', ax=ax, color='purple')
ax.set_title("Last 60 Days of Bitcoin Close Prices")
st.pyplot(fig)

# Model Evaluation
st.subheader("📏 Model Evaluation")
y_true = df['Close'].shift(-1).dropna()
y_pred = model.predict(df[features].iloc[:-1])
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
st.write(f"**R² Score (Accuracy):** {r2:.2%}")
st.write(f"**RMSE:** ${rmse:,.2f}")

# Plot actual vs predicted
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(y_true.values, label='Actual', color='blue')
ax2.plot(y_pred, label='Predicted', color='red', alpha=0.7)
ax2.set_title("Actual vs Predicted Close Price")
ax2.legend()
st.pyplot(fig2) 

# Custom Prediction Section
# --- Sidebar Inputs --- 
st.sidebar.header("📊 Input Today's Bitcoin Data")
input_date = st.sidebar.date_input("Date", value=pd.Timestamp.today())
input_open = st.sidebar.number_input("Open", min_value=0.0, value=0.0, step=0.01)
input_high = st.sidebar.number_input("High", min_value=0.0, value=0.0, step=0.01)
input_low = st.sidebar.number_input("Low", min_value=0.0, value=0.0, step=0.01)
input_close = st.sidebar.number_input("Close", min_value=0.0, value=0.0, step=0.01)
input_volume = st.sidebar.number_input("Volume", min_value=0.0, value=0.0, step=1.0)
input_marketcap = st.sidebar.number_input("Marketcap", min_value=0.0, value=0.0, step=1.0)

# --- Feature Engineering ---
input_date_pd = pd.to_datetime(input_date)
day = input_date_pd.day
month = input_date_pd.month
year = input_date_pd.year
weekday = input_date_pd.weekday()
hl_range = input_high - input_low
oc_change = input_close - input_open

# --- Rolling Features ---
df_hist = df.copy()
df_hist = df_hist.sort_values("Date")
last_6 = df_hist.tail(6)[['Close']]
closes = last_6['Close'].tolist() + [input_close]
closes_series = pd.Series(closes)
ma_7 = closes_series.rolling(window=7).mean().iloc[-1]
ma_14 = closes_series.rolling(window=14).mean().iloc[-1] if len(closes_series) >= 14 else np.nan
vol_7 = closes_series.rolling(window=7).std().iloc[-1]
lag_1 = closes_series.iloc[-2] if len(closes_series) >= 2 else np.nan

# --- Input Feature DataFrame ---
input_features = pd.DataFrame([{
    'Open': input_open,
    'High': input_high,
    'Low': input_low,
    'Close': input_close,
    'Volume': input_volume,
    'Marketcap': input_marketcap,
    'Day': day,
    'Month': month,
    'Year': year,
    'Weekday': weekday,
    'HL_Range': hl_range,
    'OC_Change': oc_change,
    'MA_7': ma_7,
    'MA_14': ma_14,
    'Volatility_7': vol_7,
    'Lag_1': lag_1
}])

# Handle NaNs
input_features = input_features.fillna(0)

# --- Make Prediction ---
pred_custom = model.predict(input_features)[0]

# --- Display on Main Page ---
st.subheader("📈 Predicted Close Price for Tomorrow (Custom Input Data ):")
st.success(f"${pred_custom:.2f}")
# 5. Fun Fact Section
st.header("💡 Fun Bitcoin Facts")
facts = [
    "The first real-world Bitcoin transaction was for two pizzas in 2010.",
    "There will only ever be 21 million Bitcoins in existence.",
    "The smallest unit of Bitcoin is called a 'Satoshi', worth 0.00000001 BTC.",
    "Bitcoin's creator, Satoshi Nakamoto, is still anonymous.",
    "The Bitcoin blockchain has never been hacked."
]
st.info(np.random.choice(facts)) 
