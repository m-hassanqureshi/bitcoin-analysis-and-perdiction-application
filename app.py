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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

# ---- Page Setup ----
logo = Image.open("logo.png")  
st.set_page_config(
    page_title="Bitcoin EDA Dashboard",
    layout="wide",
    page_icon=logo 
)
# ---- Hide Streamlit Menu and Footer ----
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)
# ---- Sidebar Branding ----
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
st.write("**Data Set Overview:**")
st.write(df.head(15))
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

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative Return'], name='Cumulative Return'))
fig5.update_layout(title="Cumulative Return Over Time", xaxis_title="Date")
st.plotly_chart(fig5, use_container_width=True)

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
# Load or train the model
def load_or_train_model():
    if os.path.exists("bitcoin_model.pkl"):
        model = joblib.load("bitcoin_model.pkl")
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
        df['Lag_1'] = df['Close'].shift(1)
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()

        features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap',
            'Day', 'Month', 'Year', 'Weekday',
            'HL_Range', 'OC_Change', 'MA_7', 'MA_14', 'Lag_1'
        ]
        X = df[features]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)
        joblib.dump(model, "bitcoin_model.pkl")

    return model

model = load_or_train_model()

# Load dataset
df = pd.read_csv("bitcoin.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')
df = df[df['Volume'] != 0]

# Feature engineering
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Weekday'] = df['Date'].dt.weekday
df['HL_Range'] = df['High'] - df['Low']
df['OC_Change'] = df['Close'] - df['Open']
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_14'] = df['Close'].rolling(window=14).mean()
df['Lag_1'] = df['Close'].shift(1)

# Drop NA rows
df = df.dropna()

features = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap',
    'Day', 'Month', 'Year', 'Weekday',
    'HL_Range', 'OC_Change', 'MA_7', 'MA_14', 'Lag_1'
]

latest = df[features].iloc[-1:]

# Prediction
pred = model.predict(latest)[0]
st.subheader("📊 Prediction")
st.metric("Tomorrow's Predicted Close Price", f"${pred:.2f}")

# Plot recent Close prices
st.subheader("📉 Recent Close Price Trend")
fig, ax = plt.subplots(figsize=(12, 4))
df.tail(60).plot(x='Date', y='Close', ax=ax, color='orange')
ax.set_title("Last 60 Days of Bitcoin Close Prices")
st.pyplot(fig) 
# --- Predict Next Day Price Using User-Entered Latest Data ---
st.header("📝 Predict Next Day Price with Custom Input")
with st.form("predict_next_day"):
    st.write("Enter the latest Bitcoin data to predict the next day's closing price:")
    col1, col2, col3 = st.columns(3)
    with col1:
        open_val = st.number_input("Open", value=float(df['Open'].iloc[-1]))
        high_val = st.number_input("High", value=float(df['High'].iloc[-1]))
        low_val = st.number_input("Low", value=float(df['Low'].iloc[-1]))
        close_val = st.number_input("Close", value=float(df['Close'].iloc[-1]))
    with col2:
        volume_val = st.number_input("Volume", value=float(df['Volume'].iloc[-1]))
        marketcap_val = st.number_input("Marketcap", value=float(df['Marketcap'].iloc[-1]))
        day_val = st.number_input("Day", value=int(df['Date'].iloc[-1].day), step=1)
        month_val = st.number_input("Month", value=int(df['Date'].iloc[-1].month), step=1)
    with col3:
        year_val = st.number_input("Year", value=int(df['Date'].iloc[-1].year), step=1)
        weekday_val = st.number_input("Weekday (0=Mon)", value=int(df['Date'].iloc[-1].weekday()), step=1)
        hl_range_val = st.number_input("High-Low Range", value=float(df['High'].iloc[-1] - df['Low'].iloc[-1]))
        oc_change_val = st.number_input("Close-Open Change", value=float(df['Close'].iloc[-1] - df['Open'].iloc[-1]))
        ma7_val = st.number_input("MA_7", value=float(df['Close'].rolling(7).mean().iloc[-1]))
        ma14_val = st.number_input("MA_14", value=float(df['Close'].rolling(14).mean().iloc[-1]))
        lag1_val = st.number_input("Lag_1 (Prev Close)", value=float(df['Close'].iloc[-1]))

    submitted = st.form_submit_button("Predict Next Day Close Price")
    if submitted:
        input_df = pd.DataFrame([{
            'Open': open_val,
            'High': high_val,
            'Low': low_val,
            'Close': close_val,
            'Volume': volume_val,
            'Marketcap': marketcap_val,
            'Day': day_val,
            'Month': month_val,
            'Year': year_val,
            'Weekday': weekday_val,
            'HL_Range': hl_range_val,
            'OC_Change': oc_change_val,
            'MA_7': ma7_val,
            'MA_14': ma14_val,
            'Lag_1': lag1_val
        }])
        pred_custom = model.predict(input_df)[0]
        st.success(f"Predicted Next Day Close Price: ${pred_custom:.2f}")
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
