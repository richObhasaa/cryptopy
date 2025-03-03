import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_fetcher import get_market_data
from utils.analysis import calculate_statistics, calculate_correlation, calculate_relative_performance

def create_comparison_chart(df, selected_coins, metric):
    """Create a comparison chart for selected coins"""
    df_filtered = df[df['name'].isin(selected_coins)].copy()

    fig = px.bar(
        df_filtered,
        x='name',
        y=metric,
        title=f'{metric.title()} Comparison',
        labels={'name': 'Cryptocurrency', metric: metric.title()},
        template='plotly_dark'
    )
    return fig

def create_correlation_heatmap(df, selected_coins):
    """Create a correlation heatmap for selected coins"""
    corr_matrix = []
    for coin1 in selected_coins:
        row = []
        for coin2 in selected_coins:
            corr = calculate_correlation(df, coin1, coin2)
            row.append(corr)
        corr_matrix.append(row)

    fig = px.imshow(
        corr_matrix,
        x=selected_coins,
        y=selected_coins,
        title='Correlation Heatmap',
        template='plotly_dark',
        color_continuous_scale='RdBu'
    )
    return fig

def show():
    st.header("Statistical Analysis")

    # Time period selection
    time_period = st.selectbox(
        "Select Time Period",
        ["24h", "7d", "30d", "1y", "3y"]
    )

    # Get market data
    df = get_market_data(time_period)

    if df.empty:
        st.error("Unable to fetch market data. Please try again later.")
        return

    # Coin selection
    selected_coins = st.multiselect(
        "Select Cryptocurrencies to Compare",
        options=df['name'].tolist(),
        default=df['name'].iloc[:3].tolist() if len(df) > 0 else []
    )

    if not selected_coins:
        st.warning("Please select at least one cryptocurrency to analyze")
        return

    # Calculate statistics for selected coins
    stats_list = []
    for coin in selected_coins:
        coin_data = df[df['name'] == coin]
        if not coin_data.empty:
            stats = calculate_statistics(coin_data)
            stats['name'] = coin
            stats_list.append(stats)

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
                f"${coin_stats['market_cap']:,.0f}",
                f"σ: ${coin_stats['std']:,.0f}"
            )

    # Create detailed statistics DataFrame
    st.subheader("Detailed Statistics")
    detailed_stats = []

    for coin in selected_coins:
        coin_stats = stats_df[stats_df['name'] == coin].iloc[0]
        detailed_stats.append({
            'Cryptocurrency': coin,
            'Market Cap': f"${coin_stats['market_cap']:,.2f}",
            'Mean': f"${coin_stats['mean']:,.2f}",
            'Median': f"${coin_stats['median']:,.2f}",
            'Std Dev': f"${coin_stats['std']:,.2f}",
            'CV (%)': f"{coin_stats['coefficient_of_variation']:.2f}%",
            'Skewness': f"{coin_stats['skewness']:.2f}",
            'Kurtosis': f"{coin_stats['kurtosis']:.2f}"
        })

    formatted_stats = pd.DataFrame(detailed_stats)
    formatted_stats.set_index('Cryptocurrency', inplace=True)
    st.table(formatted_stats)

    # Visualization options
    st.subheader("Statistical Visualizations")

    # Metric comparison
    metric = st.selectbox(
        "Select Metric to Compare",
        ['market_cap', 'mean', 'median', 'std', 'coefficient_of_variation'],
        format_func=lambda x: x.replace('_', ' ').title()
    )

    if metric:
        st.plotly_chart(create_comparison_chart(stats_df, selected_coins, metric))

    # Correlation analysis
    if len(selected_coins) > 1:
        st.subheader("Correlation Analysis")
        st.plotly_chart(create_correlation_heatmap(df, selected_coins))

    # Relative performance
    if len(selected_coins) > 1:
        st.subheader("Relative Performance")
        base_coin = st.selectbox(
            "Select Base Cryptocurrency",
            selected_coins
        )

        if base_coin:
            df_relative = calculate_relative_performance(df[df['name'].isin(selected_coins)], base_coin)
            fig = px.bar(
                df_relative,
                x='name',
                y='relative_performance',
                title=f'Performance Relative to {base_coin} (%)',
                template='plotly_dark'
            )
            st.plotly_chart(fig)

    # Export functionality
    if st.button("Export Statistics"):
        csv = formatted_stats.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='crypto_statistics_comparison.csv',
            mime='text/csv',
        )