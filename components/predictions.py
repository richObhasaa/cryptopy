import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_fetcher import get_historical_data
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

@st.cache_data(ttl=3600)
def preprocess_data(df):
    """Preprocess the time series data with caching"""
    try:
        # Handle missing values
        df = df.ffill().bfill()

        # Remove outliers using IQR method
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR)))]

        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return df

def find_best_arima_params(data):
    """Find best ARIMA parameters using simplified grid search"""
    best_params = (1, 1, 1)  # Default parameters
    best_aic = float('inf')

    # Simplified parameter grid
    for p in [1, 2]:  # Reduced parameter space
        for q in [1, 2]:
            try:
                model = ARIMA(data, order=(p, 1, q))  # Keep d=1 fixed
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (p, 1, q)
            except:
                continue

    return best_params

def create_prediction_plot(historical_data, predictions, confidence_intervals, coin, mape):
    """Create prediction plot with confidence intervals"""
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['price'],
        name='Historical',
        line=dict(color='#1f77b4')
    ))

    # Predictions
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions,
        name=f'Prediction (MAPE: {mape:.2f}%)',
        line=dict(color='#2ca02c', dash='dash')
    ))

    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=predictions.index.tolist() + predictions.index.tolist()[::-1],
        y=confidence_intervals['upper'].tolist() + confidence_intervals['lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))

    fig.update_layout(
        title=f'{coin.title()} Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        hovermode='x unified'
    )

    return fig

def show():
    st.header("Price Predictions")

    # Coin selection
    coin = st.selectbox(
        "Select Cryptocurrency",
        ["bitcoin", "ethereum", "binancecoin", "cardano", "solana"]
    )

    # Prediction period
    forecast_days = st.slider(
        "Prediction Period (Days)",
        min_value=7,
        max_value=30,  # Reduced maximum prediction period
        value=14,
        step=7
    )

    # Get historical data
    historical_data = get_historical_data(coin)

    if historical_data.empty:
        st.error("Unable to fetch historical data. Please try again later.")
        return

    # Preprocess data
    processed_data = preprocess_data(historical_data)

    with st.spinner("Generating prediction..."):
        try:
            # Find optimal parameters
            best_params = find_best_arima_params(processed_data['price'])

            # Fit ARIMA model
            model = ARIMA(processed_data['price'],
                         order=best_params)
            model_fit = model.fit()

            # Generate predictions
            forecast = model_fit.forecast(steps=forecast_days)
            forecast_index = pd.date_range(
                start=processed_data.index[-1] + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )

            # Calculate confidence intervals
            forecast_conf = model_fit.get_forecast(forecast_days)
            confidence_intervals = pd.DataFrame({
                'lower': forecast_conf.conf_int()['lower'],
                'upper': forecast_conf.conf_int()['upper']
            }, index=forecast_index)

            # Calculate MAPE
            predictions = pd.Series(forecast, index=forecast_index)
            mape = mean_absolute_percentage_error(
                processed_data['price'][-forecast_days:],
                model_fit.fittedvalues[-forecast_days:]
            ) * 100

            # Display prediction plot
            st.plotly_chart(create_prediction_plot(
                processed_data,
                predictions,
                confidence_intervals,
                coin,
                mape
            ))

            # Display model information
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAPE", f"{mape:.2f}%",
                         help="Mean Absolute Percentage Error - lower is better")
            with col2:
                st.metric("Confidence Level", "95%",
                         help="Prediction confidence interval")

            # Model details
            st.info(f"""
            Model Configuration:
            - ARIMA Order: {best_params}
            - Training Data: {len(processed_data)} days
            - Prediction Period: {forecast_days} days
            """)

        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")