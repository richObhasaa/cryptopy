import pandas as pd
import numpy as np

def calculate_statistics(df):
    """Calculate various statistical metrics for market data"""
    try:
        if df.empty:
            return {}

        df_copy = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        stats = {
            'market_cap': float(df_copy['market_cap'].iloc[0]),  # Current market cap
            'mean': float(df_copy['market_cap'].mean()),
            'median': float(df_copy['market_cap'].median()),
            'std': float(df_copy['market_cap'].std()),
            'variance': float(df_copy['market_cap'].var()),
            'skewness': float(df_copy['market_cap'].skew()),
            'kurtosis': float(df_copy['market_cap'].kurtosis()),
            'range': float(df_copy['market_cap'].max() - df_copy['market_cap'].min())
        }

        # Calculate coefficient of variation safely
        mean_value = stats['mean']
        if mean_value != 0:
            stats['coefficient_of_variation'] = (stats['std'] / mean_value) * 100
        else:
            stats['coefficient_of_variation'] = 0.0

        return stats
    except Exception as e:
        print(f"Error calculating statistics: {str(e)}")
        return {}

def calculate_correlation(df, coin1, coin2):
    """Calculate correlation between two cryptocurrencies"""
    try:
        df_copy = df.copy()
        coin1_data = df_copy[df_copy['name'] == coin1]['market_cap']
        coin2_data = df_copy[df_copy['name'] == coin2]['market_cap']

        if len(coin1_data) > 0 and len(coin2_data) > 0:
            correlation = coin1_data.corr(coin2_data)
            return float(correlation) if not np.isnan(correlation) else 0.0
        return 0.0
    except Exception as e:
        print(f"Error calculating correlation: {str(e)}")
        return 0.0

def calculate_relative_performance(df, base_coin):
    """Calculate relative performance against a base cryptocurrency"""
    try:
        df_copy = df.copy()
        base_market_cap = df_copy[df_copy['name'] == base_coin]['market_cap'].iloc[0]

        if base_market_cap > 0:
            df_copy.loc[:, 'relative_performance'] = (df_copy['market_cap'] / base_market_cap * 100).round(2)
        else:
            df_copy.loc[:, 'relative_performance'] = 0.0

        return df_copy
    except Exception as e:
        print(f"Error calculating relative performance: {str(e)}")
        df_copy = df.copy()
        df_copy['relative_performance'] = 0.0
        return df_copy