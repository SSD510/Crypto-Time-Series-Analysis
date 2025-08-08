"""
Cryptocurrency Time Series Analysis Dashboard
Main Streamlit application for analyzing cryptocurrency data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# Import custom modules
from data.data_fetcher import CryptoDataFetcher
from data.data_processor import DataProcessor
from analysis.eda import ExploratoryAnalysis
from analysis.time_series_analysis import TimeSeriesAnalysis
from models.arima_model import ARIMAForecaster
from models.prophet_model import ProphetForecaster
from utils.helpers import format_currency, calculate_returns

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crypto Time Series Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class CryptoDashboard:
    """Main dashboard class for cryptocurrency analysis"""
    
    def __init__(self):
        self.data_fetcher = CryptoDataFetcher()
        self.data_processor = DataProcessor()
        self.eda = ExploratoryAnalysis()
        self.ts_analysis = TimeSeriesAnalysis()
        self.arima_forecaster = ARIMAForecaster()
        self.prophet_forecaster = ProphetForecaster()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'crypto_data' not in st.session_state:
            st.session_state.crypto_data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
    
    def sidebar_controls(self):
        """Create sidebar controls"""
        st.sidebar.header("🎛️ Analysis Controls")
        
        # Cryptocurrency selection
        crypto_options = {
            'Bitcoin': 'bitcoin',
            'Ethereum': 'ethereum',
            'Binance Coin': 'binancecoin',
            'Cardano': 'cardano',
            'Solana': 'solana',
            'Polkadot': 'polkadot',
            'Dogecoin': 'dogecoin',
            'Avalanche': 'avalanche-2'
        }
        
        selected_crypto = st.sidebar.selectbox(
            "Select Cryptocurrency",
            list(crypto_options.keys()),
            index=0
        )
        
        # Date range selection
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now().date()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
        
        # Data loading button
        if st.sidebar.button("🔄 Load Data", type="primary"):
            self.load_data(crypto_options[selected_crypto], start_date, end_date)
        
        return selected_crypto, crypto_options[selected_crypto], start_date, end_date
    
    def load_data(self, crypto_id, start_date, end_date):
        """Load and process cryptocurrency data"""
        try:
            with st.spinner(f"Loading {crypto_id} data..."):
                # Fetch raw data
                raw_data = self.data_fetcher.get_crypto_data(
                    crypto_id, start_date, end_date
                )
                
                if raw_data is not None and not raw_data.empty:
                    # Process data
                    processed_data = self.data_processor.clean_data(raw_data)
                    processed_data = self.data_processor.add_technical_indicators(processed_data)
                    
                    # Store in session state
                    st.session_state.crypto_data = raw_data
                    st.session_state.processed_data = processed_data
                    st.session_state.data_loaded = True
                    
                    st.sidebar.success("✅ Data loaded successfully!")
                else:
                    st.sidebar.error("❌ Failed to load data. Please try again.")
                    
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")
    
    def main_header(self, crypto_name):
        """Display main header"""
        st.markdown('<h1 class="main-header">📈 Crypto Time Series Analysis</h1>', 
                   unsafe_allow_html=True)
        
        if st.session_state.data_loaded:
            data = st.session_state.processed_data
            latest_price = data['close'].iloc[-1]
            price_change = ((data['close'].iloc[-1] / data['close'].iloc[-2]) - 1) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label=f"{crypto_name} Price",
                    value=format_currency(latest_price),
                    delta=f"{price_change:.2f}%"
                )
            
            with col2:
                st.metric(
                    label="24h Volume",
                    value=format_currency(data['volume'].iloc[-1])
                )
            
            with col3:
                st.metric(
                    label="Market Cap",
                    value=format_currency(data.get('market_cap', [0]).iloc[-1] if 'market_cap' in data.columns else 0)
                )
            
            with col4:
                volatility = calculate_returns(data['close']).std() * np.sqrt(365) * 100
                st.metric(
                    label="Annual Volatility",
                    value=f"{volatility:.2f}%"
                )
    
    def overview_tab(self):
        """Overview tab content"""
        if not st.session_state.data_loaded:
            st.info("👆 Please load data using the sidebar controls first.")
            return
        
        data = st.session_state.processed_data
        
        # Price chart
        st.subheader("📊 Price History")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Historical Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.subheader("📈 Trading Volume")
        
        fig_vol = px.bar(
            x=data.index,
            y=data['volume'],
            title="Trading Volume Over Time"
        )
        fig_vol.update_layout(template='plotly_white')
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Basic statistics
        st.subheader("📋 Basic Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Price Statistics**")
            price_stats = data['close'].describe()
            st.dataframe(price_stats)
        
        with col2:
            st.write("**Volume Statistics**")
            volume_stats = data['volume'].describe()
            st.dataframe(volume_stats)
    
    def technical_analysis_tab(self):
        """Technical analysis tab content"""
        if not st.session_state.data_loaded:
            st.info("👆 Please load data using the sidebar controls first.")
            return
        
        data = st.session_state.processed_data
        
        # Moving averages chart
        st.subheader("📈 Moving Averages")
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=['Price with Moving Averages', 'Volume'],
                           row_heights=[0.7, 0.3],
                           vertical_spacing=0.1)
        
        # Price and moving averages
        fig.add_trace(go.Scatter(
            x=data.index, y=data['close'],
            mode='lines', name='Close Price',
            line=dict(color='#1f77b4')
        ), row=1, col=1)
        
        if 'sma_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['sma_20'],
                mode='lines', name='SMA 20',
                line=dict(color='orange', dash='dash')
            ), row=1, col=1)
        
        if 'sma_50' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['sma_50'],
                mode='lines', name='SMA 50',
                line=dict(color='red', dash='dot')
            ), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=data.index, y=data['volume'],
            name='Volume', marker_color='lightblue'
        ), row=2, col=1)
        
        fig.update_layout(height=600, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators
        st.subheader("🎯 Technical Indicators")
        
        if 'rsi' in data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index, y=data['rsi'],
                    mode='lines', name='RSI',
                    line=dict(color='purple')
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title="RSI (14)", yaxis_title="RSI", template='plotly_white')
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                # Latest RSI value
                latest_rsi = data['rsi'].iloc[-1]
                if latest_rsi > 70:
                    rsi_signal = "🔴 Overbought"
                elif latest_rsi < 30:
                    rsi_signal = "🟢 Oversold"
                else:
                    rsi_signal = "🔵 Neutral"
                
                st.metric("Current RSI", f"{latest_rsi:.2f}", rsi_signal)
    
    def time_series_tab(self):
        """Time series analysis tab content"""
        if not st.session_state.data_loaded:
            st.info("👆 Please load data using the sidebar controls first.")
            return
        
        data = st.session_state.processed_data
        
        # Time series decomposition
        st.subheader("🔍 Time Series Decomposition")
        
        try:
            decomposition_fig = self.ts_analysis.decompose_time_series(data['close'])
            st.plotly_chart(decomposition_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in decomposition: {str(e)}")
        
        # Stationarity test
        st.subheader("📊 Stationarity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            adf_result = self.ts_analysis.adf_test(data['close'])
            
            st.write("**Augmented Dickey-Fuller Test Results:**")
            st.write(f"ADF Statistic: {adf_result['adf_stat']:.4f}")
            st.write(f"p-value: {adf_result['p_value']:.4f}")
            st.write(f"Critical Values: {adf_result['critical_values']}")
            
            if adf_result['p_value'] < 0.05:
                st.success("✅ Series is stationary")
            else:
                st.warning("⚠️ Series is non-stationary")
        
        with col2:
            # Autocorrelation plots
            acf_fig = self.ts_analysis.plot_acf_pacf(data['close'])
            st.plotly_chart(acf_fig, use_container_width=True)
    
    def forecasting_tab(self):
        """Forecasting tab content"""
        if not st.session_state.data_loaded:
            st.info("👆 Please load data using the sidebar controls first.")
            return
        
        data = st.session_state.processed_data
        
        st.subheader("🔮 Price Forecasting")
        
        # Forecast parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_days = st.selectbox("Forecast Period", [7, 14, 30, 60], index=2)
        
        with col2:
            model_type = st.selectbox("Model Type", ["ARIMA", "Prophet"], index=1)
        
        with col3:
            if st.button("🚀 Generate Forecast"):
                self.generate_forecast(data, model_type, forecast_days)
        
        # Display forecast results if available
        if hasattr(st.session_state, 'forecast_results'):
            self.display_forecast_results()
    
    def generate_forecast(self, data, model_type, forecast_days):
        """Generate forecast using selected model"""
        try:
            with st.spinner(f"Generating {model_type} forecast..."):
                if model_type == "ARIMA":
                    forecast_result = self.arima_forecaster.forecast(
                        data['close'], forecast_days
                    )
                else:  # Prophet
                    forecast_result = self.prophet_forecaster.forecast(
                        data[['close']], forecast_days
                    )
                
                st.session_state.forecast_results = forecast_result
                st.success("✅ Forecast generated successfully!")
                
        except Exception as e:
            st.error(f"❌ Error generating forecast: {str(e)}")
    
    def display_forecast_results(self):
        """Display forecast results"""
        forecast_data = st.session_state.forecast_results
        
        # Forecast chart
        st.plotly_chart(forecast_data['chart'], use_container_width=True)
        
        # Model performance metrics
        if 'metrics' in forecast_data:
            st.subheader("📊 Model Performance")
            metrics = forecast_data['metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            
            with col2:
                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
            
            with col3:
                st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
    
    def run(self):
        """Main application runner"""
        self.initialize_session_state()
        
        # Sidebar controls
        crypto_name, crypto_id, start_date, end_date = self.sidebar_controls()
        
        # Main header
        self.main_header(crypto_name)
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "🎯 Technical Analysis", 
            "🔍 Time Series Analysis", 
            "🔮 Forecasting"
        ])
        
        with tab1:
            self.overview_tab()
        
        with tab2:
            self.technical_analysis_tab()
        
        with tab3:
            self.time_series_tab()
        
        with tab4:
            self.forecasting_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "📈 **Crypto Time Series Analysis Dashboard** | "
            "Built with Streamlit | "
            "⚠️ For educational purposes only"
        )

if __name__ == "__main__":
    # Run the dashboard
    dashboard = CryptoDashboard()
    dashboard.run()