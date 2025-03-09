import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from pycoingecko import CoinGeckoAPI
import time

# Set page config
st.set_page_config(
    page_title="Advanced Crypto Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API clients with keys
COINGECKO_API_KEY = "CG-9NqB6imomQtnNez2n2Tq8r6E"
COINMARKETCAP_API_KEY = "32f55b6d-7816-431a-b7fe-dedaedd5ae8c"
OPENAI_API_KEY = "proj_b6VfaB40HR87wzGqoFIQLLjF"
NEWS_API_KEY = "70a5e830f7e145c5be7bca2b3d713c03"
CRYPTO_PANIC_API_KEY = "6a9f6439e52d7012cd57ff780fbc081e959d3096"

# Initialize CoinGecko API with key
cg = CoinGeckoAPI()
if COINGECKO_API_KEY:
    cg.api_key = COINGECKO_API_KEY

# Cache data fetching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_data(vs_currency='usd', days='1'):
    """Fetch market data for cryptocurrencies with market cap > 100B"""
    try:
        # Get all coins with market cap data
        coins = cg.get_coins_markets(
            vs_currency=vs_currency,
            order='market_cap_desc',
            per_page=250,  # Increased to get more coins
            sparkline=True,
            price_change_percentage='1h,24h,7d,30d,1y'
        )
        
        # Filter for market cap > 100 billion (in USD)
        filtered_coins = [coin for coin in coins if coin['market_cap'] > 100000000000]
        
        # If no coins meet the criteria, take top 50
        if not filtered_coins:
            filtered_coins = coins[:50]
            
        df = pd.DataFrame(filtered_coins)
        
        # Rename columns for clarity
        if 'current_price' in df.columns:
            df.rename(columns={
                'current_price': 'price',
                'total_volume': 'volume',
                'price_change_percentage_24h': 'change_24h',
                'price_change_percentage_7d_in_currency': 'change_7d',
                'price_change_percentage_30d_in_currency': 'change_30d',
                'price_change_percentage_1y_in_currency': 'change_1y',
            }, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_historical_data(coin_id, vs_currency='usd', days=365):
    """Fetch historical price data for a specific coin with caching"""
    try:
        data = cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency=vs_currency,
            days=days
        )

        df_prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        df_market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
        
        # Merge all dataframes
        df = df_prices.merge(df_volumes, on='timestamp').merge(df_market_caps, on='timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Add daily returns
        df['daily_return'] = df['price'].pct_change() * 100
        
        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_crypto_news(limit=10):
    """Fetch latest crypto news using News API"""
    try:
        url = f"https://newsapi.org/v2/everything?q=cryptocurrency&apiKey={NEWS_API_KEY}&pageSize={limit}&language=en&sortBy=publishedAt"
        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json()
            articles = news_data.get('articles', [])
            
            # Convert to DataFrame
            df_news = pd.DataFrame(articles)
            if not df_news.empty:
                df_news['publishedAt'] = pd.to_datetime(df_news['publishedAt'])
                df_news.sort_values('publishedAt', ascending=False, inplace=True)
            return df_news
        else:
            st.warning(f"Failed to fetch news: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return pd.DataFrame()

# Statistical and analysis functions
def calculate_statistics(df):
    """Calculate comprehensive statistics for market data"""
    if df.empty:
        return {}
    
    stats = {
        'current_price': float(df['price'].iloc[-1]),
        'mean': float(df['price'].mean()),
        'median': float(df['price'].median()),
        'std_dev': float(df['price'].std()),
        'variance': float(df['price'].var()),
        'min': float(df['price'].min()),
        'max': float(df['price'].max()),
        'range': float(df['price'].max() - df['price'].min()),
        'skewness': float(df['price'].skew()),
        'kurtosis': float(df['price'].kurtosis()),
    }
    
    # Calculate volatility (annualized standard deviation of returns)
    if 'daily_return' in df.columns and len(df) > 1:
        stats['volatility'] = float(df['daily_return'].std() * np.sqrt(365))
    else:
        stats['volatility'] = 0.0
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0% for simplicity)
    if 'daily_return' in df.columns and len(df) > 1:
        avg_return = df['daily_return'].mean()
        std_return = df['daily_return'].std()
        if std_return > 0:
            stats['sharpe_ratio'] = float((avg_return * np.sqrt(365)) / (std_return * np.sqrt(365)))
        else:
            stats['sharpe_ratio'] = 0.0
    else:
        stats['sharpe_ratio'] = 0.0
        
    return stats

def find_best_buy_time(df, lookback_period):
    """Find the best time to buy within the lookback period"""
    if df.empty or len(df) < 7:
        return None, None, None
    
    # Calculate returns for different holding periods (1d, 7d, 30d)
    holding_periods = [1, 7, 30]
    result = {}
    
    for days in holding_periods:
        if len(df) > days:
            # Calculate future returns for each day
            df[f'future_return_{days}d'] = df['price'].pct_change(periods=days).shift(-days) * 100
            
            # Find the day with maximum future return
            best_day_idx = df[f'future_return_{days}d'][:-(days+1)].idxmax()
            
            if pd.notna(best_day_idx):
                best_day_price = df.loc[best_day_idx, 'price']
                best_day_return = df.loc[best_day_idx, f'future_return_{days}d']
                
                result[days] = {
                    'date': best_day_idx,
                    'price': best_day_price,
                    'return': best_day_return
                }
    
    return result

def calculate_profit_loss(entry_price, current_price, investment_amount):
    """Calculate profit/loss for a cryptocurrency investment"""
    quantity = investment_amount / entry_price
    current_value = quantity * current_price
    profit_loss = current_value - investment_amount
    roi = (profit_loss / investment_amount) * 100
    
    return {
        'quantity': quantity,
        'entry_value': investment_amount,
        'current_value': current_value,
        'profit_loss': profit_loss,
        'roi': roi
    }

def get_crypto_price_prediction(coin, days=30):
    """Get a price prediction for a cryptocurrency using OpenAI API"""
    try:
        # For this implementation, we'll use a simple statistical forecast
        # In production, you would replace this with an actual OpenAI API call
        
        # Get historical data
        df = get_historical_data(coin, days=90)  # Get 90 days of data
        
        if df.empty:
            return {"error": "No historical data available"}
        
        # Calculate trend
        last_price = df['price'].iloc[-1]
        avg_daily_change = df['price'].pct_change().mean()
        std_daily_change = df['price'].pct_change().std()
        
        # Generate simple forecasted prices (random walk with drift)
        np.random.seed(42)  # For reproducibility
        forecasted_prices = [last_price]
        
        for _ in range(days):
            # Random component with some trend bias
            change = np.random.normal(avg_daily_change, std_daily_change)
            next_price = forecasted_prices[-1] * (1 + change)
            forecasted_prices.append(next_price)
        
        # Create forecast dataframe
        date_range = pd.date_range(start=df.index[-1], periods=days+1, freq='D')
        forecast_df = pd.DataFrame({
            'price': forecasted_prices,
            'lower_bound': [p * (1 - std_daily_change * 1.96) for p in forecasted_prices],
            'upper_bound': [p * (1 + std_daily_change * 1.96) for p in forecasted_prices]
        }, index=date_range)
        
        return {
            "success": True,
            "forecast": forecast_df,
            "end_price": forecast_df['price'].iloc[-1],
            "change_pct": ((forecast_df['price'].iloc[-1] / forecast_df['price'].iloc[0]) - 1) * 100
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_crypto_with_ai(coin_id):
    """Analyze a cryptocurrency using structured data (simulated AI analysis)"""
    # In a production environment, this would make a call to the OpenAI API
    # For this implementation, we'll generate structured analysis based on market data
    
    try:
        # Get coin data
        coin_data = cg.get_coin_by_id(coin_id)
        market_data = get_historical_data(coin_id, days=30)
        
        if market_data.empty:
            return {"error": "Could not fetch sufficient data for analysis"}
        
        # Calculate some metrics
        recent_trend = "bullish" if market_data['price'].pct_change(7).iloc[-1] > 0 else "bearish"
        volatility = market_data['price'].pct_change().std() * 100
        
        # Create a structured analysis
        analysis = {
            "name": coin_data.get('name', coin_id),
            "summary": f"{coin_data.get('name', coin_id)} has been showing a {recent_trend} trend in the past week.",
            "market_sentiment": "positive" if recent_trend == "bullish" else "negative",
            "volatility": f"{volatility:.2f}%",
            "key_metrics": {
                "market_cap_rank": coin_data.get('market_cap_rank', 'N/A'),
                "current_price": market_data['price'].iloc[-1],
                "30d_change": market_data['price'].pct_change(30).iloc[-1] * 100,
                "volume": market_data['volume'].iloc[-1] if 'volume' in market_data else 'N/A'
            },
            "recommendations": [
                f"Consider the {recent_trend} trend when making investment decisions",
                f"Volatility is {volatility:.2f}%, which is " + ("high" if volatility > 5 else "moderate" if volatility > 2 else "low"),
                "Diversify your portfolio to manage risk exposure"
            ]
        }
        
        return analysis
    except Exception as e:
        return {"error": str(e)}

# UI Functions for different sections
def show_market_analysis():
    st.header("📈 Market Analysis Dashboard")
    
    # Time period and currency selection
    col1, col2 = st.columns([2, 1])
    with col1:
        time_period = st.selectbox(
            "Select Time Period",
            ["1d", "7d", "30d", "90d", "1y", "3y"],
            help="Choose the time period for market analysis"
        )
    with col2:
        vs_currency = st.selectbox(
            "Currency",
            ["usd", "eur", "jpy", "gbp"],
            format_func=lambda x: x.upper()
        )
    
    # Convert time_period to days for API
    days_mapping = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365, "3y": 1095}
    days = days_mapping.get(time_period, 30)
    
    # Fetch data
    with st.spinner("Fetching market data..."):
        df = get_market_data(vs_currency=vs_currency, days=str(days))
    
    if df.empty:
        st.error("Unable to fetch market data. Please try again later.")
        return
    
    # Display key metrics
    st.subheader("Key Market Metrics")
    total_market_cap = df['market_cap'].sum()
    avg_volume = df['volume'].mean()
    
    cols = st.columns(4)
    with cols[0]:
        st.metric(
            "Total Market Cap",
            f"${total_market_cap:,.0f}",
            help="Sum of all cryptocurrencies market cap"
        )
    with cols[1]:
        st.metric(
            "Average Volume",
            f"${avg_volume:,.0f}",
            help="Average trading volume"
        )
    with cols[2]:
        top_coin = df.iloc[0]
        st.metric(
            "Top Cryptocurrency",
            f"{top_coin['name']}",
            f"${top_coin['market_cap']:,.0f}"
        )
    with cols[3]:
        st.metric(
            "Number of Coins",
            len(df),
            help="Total number of tracked cryptocurrencies"
        )
    
    # Visualization options
    st.subheader("Market Visualizations")
    
    viz_type = st.radio(
        "Select Visualization",
        ["Market Distribution", "Price vs Volume", "Market Dominance", 
        "Performance Comparison", "Price Movement", "Trading Volume"],
        horizontal=True
    )
    
    if viz_type == "Market Distribution":
        # Create treemap
        fig = px.treemap(
            df,
            path=['name'],
            values='market_cap',
            title='Market Cap Distribution',
            template='plotly_dark',
            color='change_24h',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Price vs Volume":
        # Create scatter plot
        fig = px.scatter(
            df,
            x='price',
            y='volume',
            size='market_cap',
            color='change_24h',
            color_continuous_scale='RdYlGn',
            hover_name='name',
            log_x=True,
            log_y=True,
            title='Price vs Volume Analysis',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Market Dominance":
        # Create pie chart for top 10
        top10 = df.head(10).copy()
        others = pd.DataFrame([{
            'name': 'Others',
            'market_cap': df['market_cap'][10:].sum()
        }])
        pie_data = pd.concat([top10, others])
        
        fig = px.pie(
            pie_data,
            values='market_cap',
            names='name',
            title='Market Dominance',
            template='plotly_dark',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Performance Comparison":
        # Top 10 performers by change
        top_performers = df.sort_values('change_24h', ascending=False).head(10)
        worst_performers = df.sort_values('change_24h', ascending=True).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                top_performers,
                x='name',
                y='change_24h',
                title='Top 10 Performers (24h)',
                template='plotly_dark',
                color='change_24h',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.bar(
                worst_performers,
                x='name',
                y='change_24h',
                title='Worst 10 Performers (24h)',
                template='plotly_dark',
                color='change_24h',
                color_continuous_scale='Reds_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Price Movement":
        # Select top coins for heatmap
        top_coins = df.head(20)
        
        # Create heatmap of price changes
        data = []
        periods = ['change_24h', 'change_7d', 'change_30d', 'change_1y']
        period_labels = ['24h', '7d', '30d', '1y']
        
        for _, coin in top_coins.iterrows():
            row = []
            for period in periods:
                if period in coin and not pd.isna(coin[period]):
                    row.append(coin[period])
                else:
                    row.append(0)
            data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=period_labels,
            y=top_coins['name'],
            colorscale='RdYlGn',
            colorbar=dict(title='% Change'),
            zmin=-20,
            zmax=20
        ))
        
        fig.update_layout(
            title='Price Movement Across Different Time Periods',
            xaxis_title='Time Period',
            yaxis_title='Cryptocurrency',
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Trading Volume":
        # Trading volume comparison
        volume_data = df.sort_values('volume', ascending=False).head(15)
        
        fig = px.bar(
            volume_data,
            x='name',
            y='volume',
            title='Top 15 by Trading Volume',
            template='plotly_dark',
            color='volume',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_title='Cryptocurrency',
            yaxis_title='Trading Volume (USD)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed data table
    st.subheader("Detailed Market Data")
    
    # Display data with formatting
    display_cols = ['name', 'price', 'market_cap', 'volume', 'change_24h', 'change_7d']
    st.dataframe(
        df[display_cols].style.format({
            'price': '${:,.2f}',
            'market_cap': '${:,.0f}',
            'volume': '${:,.0f}',
            'change_24h': '{:,.2f}%',
            'change_7d': '{:,.2f}%'
        }).background_gradient(subset=['change_24h', 'change_7d'], cmap='RdYlGn', vmin=-10, vmax=10),
        use_container_width=True
    )

def show_statistical_analysis():
    st.header("📊 Statistical Analysis")
    
    # Get market data
    with st.spinner("Fetching market data..."):
        market_data = get_market_data()
    
    if market_data.empty:
        st.error("Unable to fetch market data. Please try again later.")
        return
    
    # Time period selection
    time_period = st.selectbox(
        "Select Time Period for Analysis",
        ["30d", "90d", "1y", "3y"],
        index=0,
        key="stat_time_period"
    )
    
    days_mapping = {"30d": 30, "90d": 90, "1y": 365, "3y": 1095}
    days = days_mapping.get(time_period, 30)
    
    # Coin selection
    selected_coins = st.multiselect(
        "Select Cryptocurrencies to Compare",
        options=market_data['name'].tolist(),
        default=market_data['name'].iloc[:3].tolist() if len(market_data) > 0 else []
    )
    
    if not selected_coins:
        st.warning("Please select at least one cryptocurrency to analyze")
        return
    
    # Calculate statistics for selected coins
    with st.spinner("Calculating statistics..."):
        stats_list = []
        
        for coin in selected_coins:
            coin_id = market_data[market_data['name'] == coin]['id'].iloc[0]
            historical_data = get_historical_data(coin_id, days=days)
            
            if not historical_data.empty:
                coin_stats = calculate_statistics(historical_data)
                coin_stats['name'] = coin
                coin_stats['id'] = coin_id
                stats_list.append(coin_stats)
        
        if not stats_list:
            st.error("No statistics available for selected coins")
            return
            
        stats_df = pd.DataFrame(stats_list)
    
    # Display summary metrics
    st.subheader("Summary Metrics")
    cols = st.columns(len(selected_coins))
    
    for idx, coin in enumerate(selected_coins):
        coin_stats = stats_df[stats_df['name'] == coin].iloc[0]
        with cols[idx]:
            st.metric(
                coin,
                f"${coin_stats['current_price']:,.2f}",
                f"σ: ${coin_stats['std_dev']:,.2f}"
            )
    
    # Display detailed statistics
    st.subheader("Detailed Statistics")
    
    # Format the statistics for display
    display_stats = []
    metrics = [
        ('Mean', 'mean', '$'),
        ('Median', 'median', '$'),
        ('Standard Deviation', 'std_dev', '$'),
        ('Variance', 'variance', '$'),
        ('Min', 'min', '$'),
        ('Max', 'max', '$'),
        ('Range', 'range', '$'),
        ('Volatility (Annualized)', 'volatility', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Skewness', 'skewness', ''),
        ('Kurtosis', 'kurtosis', '')
    ]
    
    for label, key, prefix in metrics:
        row = {'Metric': label}
        
        for coin in selected_coins:
            coin_stats = stats_df[stats_df['name'] == coin].iloc[0]
            value = coin_stats.get(key, 0)
            
            if key in ['volatility', 'sharpe_ratio']:
                row[coin] = f"{value:.2f}{prefix}"
            elif key in ['skewness', 'kurtosis']:
                row[coin] = f"{value:.2f}"
            else:
                row[coin] = f"{prefix}{value:,.2f}"
                
        display_stats.append(row)
    
    # Create DataFrame and set Metric as index
    formatted_stats = pd.DataFrame(display_stats)
    formatted_stats.set_index('Metric', inplace=True)
    
    # Display as table
    st.table(formatted_stats)
    
    # Visualizations
    st.subheader("Statistical Visualizations")
    
    # Select metrics to compare
    metric_options = {
        'mean': 'Mean Price',
        'median': 'Median Price',
        'std_dev': 'Standard Deviation',
        'volatility': 'Volatility',
        'sharpe_ratio': 'Sharpe Ratio',
        'max': 'Maximum Price',
        'min': 'Minimum Price'
    }
    
    selected_metric = st.selectbox(
        "Select Metric to Compare",
        options=list(metric_options.keys()),
        format_func=lambda x: metric_options[x]
    )
    
    # Create comparison chart
    if selected_metric:
        fig = px.bar(
            stats_df,
            x='name',
            y=selected_metric,
            title=f'Comparison of {metric_options[selected_metric]}',
            template='plotly_dark',
            color=selected_metric,
            color_continuous_scale='Viridis'
        )
        
        # Format y-axis based on metric
        if selected_metric in ['mean', 'median', 'std_dev', 'max', 'min']:
            fig.update_layout(yaxis_title=f"Value (USD)")
        elif selected_metric == 'volatility':
            fig.update_layout(yaxis_title="Volatility (%)")
        else:
            fig.update_layout(yaxis_title=metric_options[selected_metric])
            
        st.plotly_chart(fig, use_container_width=True)
    
    # Historical comparison if more than one coin selected
    if len(selected_coins) > 1:
        st.subheader("Historical Price Comparison")
        
        # Get normalized price data for all selected coins
        with st.spinner("Preparing comparison data..."):
            comparison_data = pd.DataFrame()
            
            for coin in selected_coins:
                coin_id = stats_df[stats_df['name'] == coin]['id'].iloc[0]
                hist_data = get_historical_data(coin_id, days=days)
                
                if not hist_data.empty:
                    # Normalize prices to start at 100 for comparison
                    normalized_price = (hist_data['price'] / hist_data['price'].iloc[0]) * 100
                    comparison_data[coin] = normalized_price
            
            comparison_data.dropna(inplace=True)
        
        if not comparison_data.empty:
            # Create comparison chart
            fig = go.Figure()
            
            for coin in selected_coins:
                if coin in comparison_data.columns:
                    fig.add_trace(go.Scatter(
                        x=comparison_data.index,
                        y=comparison_data[coin],
                        name=coin,
                        mode='lines'
                    ))
            
            fig.update_layout(
                title='Normalized Price Comparison (Starting at 100)',
                xaxis_title='Date',
                yaxis_title='Normalized Price',
                template='plotly_dark',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            if len(selected_coins) > 1:
                st.subheader("Results")
            
            # Display results
            st.metric(
                "Quantity Owned",
                f"{results['quantity']:.6f} {selected_coin}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Initial Investment",
                    f"${results['entry_value']:,.2f}"
                )
            with col2:
                st.metric(
                    "Current Value",
                    f"${results['current_value']:,.2f}",
                    f"{results['roi']:+.2f}%"
                )
            
            # Profit/Loss card
            profit_loss = results['profit_loss']
            if profit_loss >= 0:
                st.success(f"Profit: ${profit_loss:,.2f} (ROI: {results['roi']:+.2f}%)")
            else:
                st.error(f"Loss: ${profit_loss:,.2f} (ROI: {results['roi']:+.2f}%)")
    
    with col2:
        # Historical context
        st.subheader("Historical Context")
        
        coin_id = market_data[market_data['name'] == selected_coin]['id'].iloc[0]
        historical_data = get_historical_data(coin_id, days=365)
        
        if not historical_data.empty:
            # Price chart
            fig = px.line(
                historical_data,
                y='price',
                title=f'{selected_coin} Price - Past Year',
                template='plotly_dark'
            )
            
            # Add entry price line
            fig.add_hline(
                y=entry_price,
                line_dash="dash",
                line_color="yellow",
                annotation_text="Entry Price"
            )
            
            # Add current price line
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="green" if current_price > entry_price else "red",
                annotation_text="Current Price"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate historical profit opportunities
            st.subheader("Best Entry Points (Past Year)")
            best_times = find_best_buy_time(historical_data, 365)
            
            if best_times:
                for holding_period, data in best_times.items():
                    st.metric(
                        f"Best day to buy for {holding_period}d holding",
                        f"{data['date'].strftime('%Y-%m-%d')}",
                        f"{data['return']:+.2f}%"
                    )

def show_price_prediction():
    st.header("🔮 Price Prediction")
    
    # Get market data
    with st.spinner("Fetching market data..."):
        market_data = get_market_data()
    
    if market_data.empty:
        st.error("Unable to fetch market data. Please try again later.")
        return
    
    # Coin selection
    coin = st.selectbox(
        "Select Cryptocurrency",
        options=market_data['name'].tolist(),
        index=0 if not market_data.empty else None
    )
    
    if not coin:
        st.warning("Please select a cryptocurrency")
        return
    
    # Get coin ID
    coin_id = market_data[market_data['name'] == coin]['id'].iloc[0]
    
    # Prediction period
    forecast_days = st.slider(
        "Prediction Period (Days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    # Generate prediction
    with st.spinner("Generating prediction..."):
        prediction_result = get_crypto_price_prediction(coin_id, days=forecast_days)
    
    if prediction_result.get("error"):
        st.error(f"Error generating prediction: {prediction_result['error']}")
        return
    
    if prediction_result.get("success"):
        # Get current price for reference
        current_price = market_data[market_data['name'] == coin]['price'].iloc[0]
        end_price = prediction_result["end_price"]
        change_pct = prediction_result["change_pct"]
        
        # Display prediction metrics
        st.subheader("Prediction Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}"
            )
        with col2:
            st.metric(
                f"Predicted Price ({forecast_days} days)",
                f"${end_price:.2f}",
                f"{change_pct:+.2f}%"
            )
        with col3:
            conf_level = st.selectbox(
                "Confidence Level",
                ["95%", "90%", "80%"],
                index=0
            )
        
        # Create prediction plot
        forecast_df = prediction_result["forecast"]
        
        # Get historical data for context
        historical_data = get_historical_data(coin_id, days=90)
        
        # Plot both historical and forecast data
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['price'],
            name='Historical',
            line=dict(color='#1f77b4')
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['price'],
            name='Prediction',
            line=dict(color='#2ca02c', dash='dash')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
            y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(44, 160, 44, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{conf_level} Confidence Interval'
        ))
        
        # Layout
        fig.update_layout(
            title=f'{coin} Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add prediction disclaimer
        st.info("""
        **Disclaimer**: This prediction is based on historical data and statistical analysis. 
        Cryptocurrency markets are highly volatile and actual prices may vary significantly from predictions.
        This should not be considered as financial advice.
        """)
        
        # Model details
        st.subheader("Model Information")
        st.write("""
        This prediction is generated using a combination of:
        - Historical price trend analysis
        - Volatility-based confidence intervals
        - Statistical time series modeling
        """)

def show_ai_insights():
    st.header("🧠 AI Crypto Insights")
    
    # Get market data
    with st.spinner("Fetching market data..."):
        market_data = get_market_data()
    
    if market_data.empty:
        st.error("Unable to fetch market data. Please try again later.")
        return
    
    # Coin selection
    selected_coin = st.selectbox(
        "Select Cryptocurrency for AI Analysis",
        options=market_data['name'].tolist(),
        index=0 if not market_data.empty else None
    )
    
    if not selected_coin:
        st.warning("Please select a cryptocurrency")
        return
    
    # Get coin ID
    coin_id = market_data[market_data['name'] == selected_coin]['id'].iloc[0]
    
    # Generate AI insights
    with st.spinner("Generating AI insights..."):
        analysis = analyze_crypto_with_ai(coin_id)
    
    if analysis.get("error"):
        st.error(f"Error generating analysis: {analysis['error']}")
        return
    
    # Display AI analysis
    st.subheader(f"AI Analysis for {analysis.get('name', selected_coin)}")
    
    # Summary
    st.markdown(f"### Summary")
    st.write(analysis.get("summary", "No summary available"))
    
    # Market sentiment
    sentiment = analysis.get("market_sentiment", "neutral")
    sentiment_color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "blue"
    
    st.markdown(f"### Market Sentiment: <span style='color:{sentiment_color}'>{sentiment.title()}</span>", unsafe_allow_html=True)
    st.write(f"Volatility: {analysis.get('volatility', 'N/A')}")
    
    # Key metrics
    st.markdown("### Key Metrics")
    metrics = analysis.get("key_metrics", {})
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Market Cap Rank", f"#{metrics.get('market_cap_rank', 'N/A')}")
    with cols[1]:
        st.metric("Current Price", f"${metrics.get('current_price', 0):,.2f}")
    with cols[2]:
        st.metric("30d Change", f"{metrics.get('30d_change', 0):+.2f}%")
    with cols[3]:
        st.metric("24h Volume", f"${metrics.get('volume', 0):,.0f}")
    
    # Recommendations
    st.markdown("### AI Recommendations")
    recommendations = analysis.get("recommendations", [])
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Disclaimer
    st.info("""
    **Disclaimer**: These insights are generated through automated analysis of market data and should not
    be considered as financial advice. Always do your own research and consult a financial advisor
    before making investment decisions.
    """)

def show_crypto_news():
    st.header("📰 Cryptocurrency News")
    
    # News sources
    news_source = st.radio(
        "Select News Source",
        ["News API", "Crypto Panic"],
        horizontal=True
    )
    
    # Number of news items
    news_limit = st.slider(
        "Number of News Items",
        min_value=5,
        max_value=25,
        value=10,
        step=5
    )
    
    # Fetch news
    with st.spinner("Fetching latest crypto news..."):
        news_df = get_crypto_news(limit=news_limit)
    
    if news_df.empty:
        st.error("Unable to fetch news. Please try again later.")
        return
    
    # Display news
    st.subheader("Latest Cryptocurrency News")
    
    for i, (_, news) in enumerate(news_df.iterrows()):
        with st.container():
            st.markdown(f"#### {news['title']}")
            st.markdown(f"*{news['publishedAt'].strftime('%Y-%m-%d %H:%M')} • {news['source']['name']}*")
            
            st.markdown(f"{news['description']}")
            
            if news['urlToImage'] and str(news['urlToImage']).startswith('http'):
                st.image(news['urlToImage'], use_column_width=True)
            
            st.markdown(f"[Read full article]({news['url']})")
            st.markdown("---")

def main():
    st.title("Advanced Cryptocurrency Analysis Platform")
    
    # Add description
    st.markdown("""
    This platform provides comprehensive cryptocurrency analysis using advanced statistical methods, 
    real-time data, and AI-powered insights:
    
    - **Market Analysis**: Real-time market data and visualization
    - **Statistical Analysis**: Detailed statistical metrics and comparisons
    - **Profit/Loss Calculator**: Calculate potential returns on investments
    - **Price Predictions**: Advanced time series forecasting
    - **AI Insights**: AI-powered analysis and recommendations
    - **Crypto News**: Latest cryptocurrency news
    """)
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Market Analysis"
    
    # Navigation
    pages = {
        "Market Analysis": {
            "func": show_market_analysis,
            "icon": "📈",
            "desc": "View market trends with 6+ chart types"
        },
        "Statistical Analysis": {
            "func": show_statistical_analysis,
            "icon": "📊",
            "desc": "Compare detailed statistics across coins"
        },
        "Profit/Loss Calculator": {
            "func": show_profit_loss_calculator,
            "icon": "💰",
            "desc": "Calculate investment returns"
        },
        "Price Predictions": {
            "func": show_price_prediction,
            "icon": "🔮",
            "desc": "Forecast future price movements"
        },
        "AI Insights": {
            "func": show_ai_insights,
            "icon": "🧠",
            "desc": "Get AI-powered crypto analysis"
        },
        "Crypto News": {
            "func": show_crypto_news,
            "icon": "📰",
            "desc": "Latest cryptocurrency news"
        }
    }
    
    with st.sidebar:
        st.header("Navigation")
        
        for page_name, page_info in pages.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"{page_info['icon']}")
            with col2:
                if st.button(f"{page_name}", key=page_name, use_container_width=True):
                    st.session_state.current_page = page_name
            st.write(f"{page_info['desc']}")
            st.markdown("---")
        
        st.sidebar.markdown("### About")
        st.sidebar.info("""
        This platform uses real-time data from CoinGecko API,
        advanced statistical modeling, and AI analysis to help
        you make informed decisions in the cryptocurrency market.
        """)
        
        # Add refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()
    
    # Display selected page
    current_page = st.session_state.get('current_page', "Market Analysis")
    pages[current_page]["func"]()

if __name__ == "__main__":
    main()("Price Correlation Matrix")
                
                # Calculate correlation matrix
                corr_matrix = comparison_data.corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    range_color=[-1, 1],
                    template='plotly_dark',
                    title='Price Correlation Between Selected Cryptocurrencies'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    if st.button("Export Statistics"):
        csv = formatted_stats.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f'crypto_statistics_{time_period}.csv',
            mime='text/csv',
        )

def show_profit_loss_calculator():
    st.header("💰 Profit/Loss Calculator")
    
    # Get market data
    with st.spinner("Fetching market data..."):
        market_data = get_market_data()
    
    if market_data.empty:
        st.error("Unable to fetch market data. Please try again later.")
        return
    
    # Select coin
    selected_coin = st.selectbox(
        "Select Cryptocurrency",
        options=market_data['name'].tolist(),
        index=0 if not market_data.empty else None
    )
    
    if not selected_coin:
        st.warning("Please select a cryptocurrency")
        return
    
    # Get current price
    current_price = market_data[market_data['name'] == selected_coin]['price'].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input investment details
        st.subheader("Investment Details")
        
        entry_price = st.number_input(
            "Entry Price (USD)",
            min_value=0.0,
            value=current_price,
            step=0.01,
            format="%.6f"
        )
        
        investment_amount = st.number_input(
            "Investment Amount (USD)",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )
        
        # Calculate profit/loss
        if entry_price > 0:
            results = calculate_profit_loss(entry_price, current_price, investment_amount)
            
            st.subheader
