import pandas as pd
import streamlit as st
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta

cg = CoinGeckoAPI()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_data(time_period):
    """Fetch market data for top 50 cryptocurrencies with caching"""
    try:
        market_data = cg.get_coins_markets(
            vs_currency='usd',
            order='market_cap_desc',
            per_page=50,
            sparkline=False
        )

        df = pd.DataFrame(market_data)
        df = df[['id', 'name', 'current_price', 'market_cap', 'total_volume']]
        df.columns = ['id', 'name', 'price', 'market_cap', 'volume']

        return df

    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_historical_data(coin_id):
    """Fetch historical price data for a specific coin with caching"""
    try:
        days = 365
        data = cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency='usd',
            days=days
        )

        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()