import streamlit as st
from components import market_analysis, statistics, predictions

st.set_page_config(
    page_title="Crypto Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Cryptocurrency Analysis Platform")

    # Add description
    st.markdown("""
    This platform provides comprehensive cryptocurrency analysis using advanced statistical methods and machine learning:
    - **Market Analysis**: Real-time market data and visualization
    - **Statistical Analysis**: Detailed statistical metrics and comparisons
    - **Price Predictions**: Advanced time series forecasting
    """)

    # Navigation
    pages = {
        "Market Analysis": {
            "func": market_analysis.show,
            "icon": "📈",
            "desc": "View market trends and comparisons"
        },
        "Statistical Analysis": {
            "func": statistics.show,
            "icon": "📊",
            "desc": "Analyze detailed statistics and correlations"
        },
        "Price Predictions": {
            "func": predictions.show,
            "icon": "🔮",
            "desc": "Forecast future price movements"
        }
    }

    with st.sidebar:
        st.header("Navigation")

        for page_name, page_info in pages.items():
            st.write(f"{page_info['icon']} **{page_name}**")
            st.write(page_info['desc'])
            if st.button(f"Go to {page_name}", key=page_name):
                st.session_state.current_page = page_name

        st.sidebar.markdown("---")
        st.sidebar.markdown("### About")
        st.sidebar.info("""
        This platform uses advanced statistical modeling and machine learning
        to analyze cryptocurrency markets and predict future trends.
        """)

    # Display selected page
    current_page = st.session_state.get('current_page', "Market Analysis")
    pages[current_page]["func"]()

if __name__ == "__main__":
    main()