import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.data_fetcher import get_market_data
from utils.analysis import calculate_statistics, calculate_correlation, calculate_relative_performance

def create_market_cap_chart(df):
    """Create an interactive market cap chart"""
    fig = px.treemap(
        df,
        path=['name'],
        values='market_cap',
        title='Market Cap Distribution',
        template='plotly_dark'
    )
    return fig

def create_price_volume_scatter(df):
    """Create price vs volume scatter plot"""
    fig = px.scatter(
        df,
        x='price',
        y='volume',
        size='market_cap',
        color='name',
        title='Price vs Volume Analysis',
        template='plotly_dark',
        hover_data=['name', 'market_cap']
    )
    return fig

def create_market_dominance(df):
    """Create market dominance pie chart"""
    fig = px.pie(
        df.head(10),
        values='market_cap',
        names='name',
        title='Top 10 Cryptocurrencies by Market Cap',
        template='plotly_dark'
    )
    return fig

def create_comparison_chart(df, metric):
    """Create comparison bar chart"""
    fig = px.bar(
        df.head(15),
        x='name',
        y=metric,
        title=f'Top 15 by {metric.replace("_", " ").title()}',
        template='plotly_dark'
    )
    return fig

def show():
    st.header("Market Analysis Dashboard")

    # Time period selection
    col1, col2 = st.columns([2, 1])
    with col1:
        time_period = st.selectbox(
            "Select Time Period",
            ["24h", "7d", "30d", "1y", "3y"],
            help="Choose the time period for market analysis"
        )
    with col2:
        metric = st.selectbox(
            "Select Metric",
            ["market_cap", "volume", "price"],
            format_func=lambda x: x.replace("_", " ").title()
        )

    # Fetch data
    with st.spinner("Fetching market data..."):
        df = get_market_data(time_period)

    if df.empty:
        st.error("Unable to fetch market data. Please try again later.")
        return

    # Calculate statistics
    stats = calculate_statistics(df)

    # Display key metrics
    st.subheader("Key Market Metrics")
    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "Total Market Cap",
            f"${df['market_cap'].sum():,.0f}",
            help="Sum of all cryptocurrencies market cap"
        )
    with cols[1]:
        st.metric(
            "Average Volume",
            f"${df['volume'].mean():,.0f}",
            help="Average trading volume"
        )
    with cols[2]:
        top_gain = df.nlargest(1, 'market_cap')
        st.metric(
            "Largest Market Cap",
            f"${top_gain['market_cap'].iloc[0]:,.0f}",
            f"{top_gain['name'].iloc[0]}"
        )
    with cols[3]:
        st.metric(
            "Number of Coins",
            len(df),
            help="Total number of tracked cryptocurrencies"
        )

    # Visualizations
    st.subheader("Market Visualizations")

    viz_type = st.radio(
        "Select Visualization",
        ["Market Distribution", "Price vs Volume", "Market Dominance", "Comparison"],
        horizontal=True
    )

    if viz_type == "Market Distribution":
        st.plotly_chart(create_market_cap_chart(df), use_container_width=True)
    elif viz_type == "Price vs Volume":
        st.plotly_chart(create_price_volume_scatter(df), use_container_width=True)
    elif viz_type == "Market Dominance":
        st.plotly_chart(create_market_dominance(df), use_container_width=True)
    else:
        st.plotly_chart(create_comparison_chart(df, metric), use_container_width=True)

    # Detailed data table
    st.subheader("Detailed Market Data")
    st.dataframe(
        df[['name', 'price', 'market_cap', 'volume']].style.format({
            'price': '${:,.2f}',
            'market_cap': '${:,.0f}',
            'volume': '${:,.0f}'
        }),
        use_container_width=True
    )