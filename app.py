import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Stock Data Extraction", layout="wide")

st.title("Technical analysis indicator")

tab1, tab2 = st.tabs(["Stock Indicator", "Portfolio Analysis"])

with tab1:
    st.write("Extract stock market prices from yahoo finance using ticker")

    st.subheader("Stock Input")
    ticker = st.text_input("Enter Ticker", "AAPL")
    start_date = st.date_input("start date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))
    get_data_button = st.button("Get Data")

    if get_data_button:
        # create ticker object
        stock = yf.Ticker(ticker)

        #  download historical prices
        df = stock.history(start = start_date, end = end_date)

        # check the data
        if df.empty:
            st.error("No Data Found. Please check the ticker symbol or date range")
        else:
            st.success(f"Data Successfully extracted for {ticker}")

            # display company info
            st.subheader("Company Information")
            info = stock.info


            company_name = info.get("longName", "N/A")
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            market_cap = info.get("marketCap", "N/A")
            website = info.get("website", "N/A")


            st.write(f"**Company Name:** {company_name}")
            st.write(f"**Sector:** {sector}")
            st.write(f"**Industry:** {industry}")
            st.write(f"**Market Cap:** {market_cap:,}")
            st.write(f"**Website:** {website}")


            st.subheader("Historical data")
            st.dataframe(df)

            st.subheader("Closing price chart")
            fig, ax = plt.subplots()
            ax.plot(df.index, df['Close'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Closing Price")
            st.pyplot(fig)

            st.subheader("Moving Averages and Trend Analysis")

            # Calculate moving averages
            df["MA20"] = df["Close"].rolling(20).mean()
            df["MA50"] = df["Close"].rolling(50).mean()
            df["MA200"] = df["Close"].rolling(200).mean()

            # Plot moving averages
            fig_ma, ax_ma = plt.subplots(figsize=(12, 6))
            ax_ma.plot(df["Close"], label="Price")
            ax_ma.plot(df["MA20"], label="MA20")
            ax_ma.plot(df["MA50"], label="MA50")
            ax_ma.plot(df["MA200"], label="MA200")
            ax_ma.set_title("Moving Averages Vs Price")
            ax_ma.set_xlabel("Date")
            ax_ma.set_ylabel("Price")
            ax_ma.legend()
            st.pyplot(fig_ma)

            # Trend analysis
            close_prices = df["Close"]
            trend = "N/A"
            if len(close_prices) >= 200: # Ensure enough data for all MAs
                current_price = close_prices.iloc[-1]
                ma_20 = df["MA20"].iloc[-1]
                ma_50 = df["MA50"].iloc[-1]
                ma_200 = df["MA200"].iloc[-1]

                st.write(f"**Current Price:** {current_price:.2f}")
                st.write(f"**20-Day Moving Average (MA20):** {ma_20:.2f}")
                st.write(f"**50-Day Moving Average (MA50):** {ma_50:.2f}")
                st.write(f"**200-Day Moving Average (MA200):** {ma_200:.2f}")

                if current_price > ma_20 and current_price > ma_50 and current_price > ma_200:
                    st.success("**Trend:** Upward trend")
                    trend = "upward"
                elif current_price < ma_20 and current_price < ma_50 and current_price < ma_200:
                    st.error("**Trend:** Downward trend")
                    trend = "downward"
                else:
                    st.info("**Trend:** Mixed trend")
                    trend = "mixed"
            else:
                st.warning("Not enough data to calculate 200-day moving average and determine a clear trend. Need at least 200 data points.")

            st.subheader("Relative Strength Index (RSI)")

            rsi_value = None
            rsi_interpretation = "N/A"
            if df.shape[0] >= 14: # Ensure enough data for RSI calculation
                # Calculate delta
                delta = df['Close'].diff(1)

                # Calculate gains and losses
                gains = delta.clip(lower=0)
                losses = -delta.clip(upper=0)

                # Calculate average gains and losses using rolling mean
                avg_gain = gains.ewm(com=13, adjust=False).mean()
                avg_loss = losses.ewm(com=13, adjust=False).mean()

                # Calculate Relative Strength (RS) and RSI
                if avg_loss.iloc[-1] == 0:
                    rs = 100 # To avoid division by zero if there are no losses
                else:
                    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]

                rsi_value = 100 - (100 / (1 + rs))

                st.write(f"**RSI (14-period):** {rsi_value:.2f}")

                if rsi_value < 30:
                    st.info("**RSI Interpretation:** Market is oversold")
                    rsi_interpretation = "oversold"
                elif rsi_value > 70:
                    st.error("**RSI Interpretation:** Market is overbought")
                    rsi_interpretation = "overbought"
                else:
                    st.success("**RSI Interpretation:** Market is neutral")
                    rsi_interpretation = "neutral"
            else:
                st.warning("Not enough data to calculate RSI. Need at least 14 data points.")

            st.subheader("Trading Signal")
            trading_signal = "No signal available"

            if trend == "upward" and rsi_interpretation == "oversold":
                trading_signal = "Strong Buy Signal"
                st.success(f"**Trading Signal:** {trading_signal}")
            elif trend == "upward" or rsi_interpretation == "oversold":
                trading_signal = "Buy Signal"
                st.success(f"**Trading Signal:** {trading_signal}")
            elif trend == "downward" and rsi_interpretation == "overbought":
                trading_signal = "Strong Sell Signal"
                st.error(f"**Trading Signal:** {trading_signal}")
            elif trend == "downward" or rsi_interpretation == "overbought":
                trading_signal = "Sell Signal"
                st.error(f"**Trading Signal:** {trading_signal}")
            elif trend == "mixed" and rsi_interpretation == "neutral":
                trading_signal = "Hold Signal"
                st.info(f"**Trading Signal:** {trading_signal}")
            else:
                st.write(f"**Trading Signal:** {trading_signal}")


            st.subheader("Volatility Analysis")

            VOLATILITY_PERIOD = 20

            if df.shape[0] >= VOLATILITY_PERIOD:
                df['Daily_Return'] = df['Close'].pct_change()
                df['Rolling_Std_Dev'] = df['Daily_Return'].rolling(window=VOLATILITY_PERIOD).std()
                df['Annualized_Volatility'] = df['Rolling_Std_Dev'] * np.sqrt(252)

                st.write("**Volatility Data (last 5 rows):**")
                st.dataframe(df[['Close', 'Daily_Return', 'Rolling_Std_Dev', 'Annualized_Volatility']].tail())

                latest_volatility = df['Annualized_Volatility'].iloc[-1]

                def categorize_volatility(volatility):
                    volatility_percent = volatility * 100
                    if volatility_percent > 40:
                        return 'High'
                    elif 25 <= volatility_percent <= 40:
                        return 'Medium'
                    else:
                        return 'Low'

                volatility_category = categorize_volatility(latest_volatility)

                st.write(f"\n**{VOLATILITY_PERIOD}-day Volatility for {ticker}:** {latest_volatility:.2%}")
                st.write(f"**Category:** {volatility_category}")
            else:
                st.warning(f"Not enough data to calculate {VOLATILITY_PERIOD}-day Volatility. Need at least {VOLATILITY_PERIOD} data points.")

            # convert dataframe to CSV for download
            csv = df.to_csv().encode("utf-8")

            #  create download button for CSV

            st.download_button(
            label = "Download Data as CSV",
            data = csv,
            file_name = f"{ticker}_stock_data.csv",
            mime = "text/csv"
            )

with tab2:
    st.write("## Portfolio Analysis")
    st.write("Enter 5 stock tickers and their respective weights (0-100%). The total weight must sum to 100%.")

    st.subheader("Portfolio Date Range")
    start_date_portfolio = st.date_input("Start Date for Portfolio", pd.to_datetime("2023-01-01"), key='portfolio_start_date')
    end_date_portfolio = st.date_input("End Date for Portfolio", pd.to_datetime("today"), key='portfolio_end_date')

    portfolio_stocks = []
    portfolio_weights = []

    for i in range(5):
        col1, col2 = st.columns(2)
        with col1:
            ticker_input = st.text_input(f"Stock {i+1} Ticker", key=f"ticker_{i}")
            portfolio_stocks.append(ticker_input)
        with col2:
            weight_input = st.number_input(f"Weight {i+1} (%)", min_value=0, max_value=100, value=0, key=f"weight_{i}")
            portfolio_weights.append(weight_input)

    if st.button("Analyze Portfolio"):
        if all(s != '' for s in portfolio_stocks) and len(portfolio_stocks) == 5:
            total_weight = sum(portfolio_weights)
            if total_weight == 100:
                st.success("Portfolio data entered successfully!")
                st.write("### Your Portfolio:")
                for i in range(5):
                    st.write(f"**{portfolio_stocks[i].upper()}**: {portfolio_weights[i]}%")

                # Start of the new code for portfolio data fetching
                st.subheader('Fetching Portfolio Data...')

                stock_data = {}
                valid_tickers = [ticker for ticker in portfolio_stocks if ticker]

                if not valid_tickers:
                    st.error("Please enter at least one stock ticker.")
                else:
                    for ticker in valid_tickers:
                        try:
                            df_portfolio = yf.download(ticker, start=start_date_portfolio, end=end_date_portfolio)
                            if df_portfolio.empty:
                                st.error(f"No Data Found for {ticker}. Please check the ticker symbol or date range.")
                            else:
                                stock_data[ticker] = df_portfolio
                                st.success(f"Data Successfully extracted for {ticker}")
                        except Exception as e:
                            st.error(f"Error fetching data for {ticker}: {e}")

                    if stock_data:
                        first_ticker = list(stock_data.keys())[0]
                        st.write(f"Preview of data for {first_ticker}:")
                        st.dataframe(stock_data[first_ticker].head())

                    # Fetch SPY data as a benchmark
                    st.subheader('Fetching Benchmark Data (SPY)...')
                    try:
                        spy_data = yf.download('SPY', start=start_date_portfolio, end=end_date_portfolio)
                        if spy_data.empty:
                            st.warning("No Data Found for SPY. Unable to fetch benchmark data.")
                        else:
                            st.success("SPY benchmark data successfully extracted.")
                            st.write("Preview of SPY data:")
                            st.dataframe(spy_data.head())
                    except Exception as e:
                        st.error(f"Error fetching SPY benchmark data: {e}")
                # End of the new code for portfolio data fetching

            else:
                st.error(f"Total weights must sum to 100%. Current total: {total_weight}%")
        else:
            st.error("Please enter all 5 stock tickers.")
            st.error("Please enter all 5 stock tickers.")
