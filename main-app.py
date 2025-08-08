"""
Cryptocurrency Time Series Analysis Dashboard
Complete self-contained Streamlit application for analyzing cryptocurrency data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import requests
import time
from typing import Optional, Dict, List
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import io
import base64

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


class CryptoDataFetcher:
    """Cryptocurrency data fetcher using CoinGecko API"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoAnalysisApp/1.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0
        
        # Cryptocurrency mapping
        self.crypto_mapping = {
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'binancecoin': 'binancecoin',
            'cardano': 'cardano',
            'solana': 'solana',
            'polkadot': 'polkadot',
            'dogecoin': 'dogecoin',
            'avalanche-2': 'avalanche-2'
        }
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @st.cache_data(ttl=3600)
    def get_crypto_data(_self, crypto_id: str, start_date: datetime, 
                       end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data from CoinGecko API"""
        try:
            _self._rate_limit()
            
            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            # CoinGecko API endpoint
            url = f"{_self.base_url}/coins/{crypto_id}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': start_timestamp,
                'to': end_timestamp
            }
            
            response = _self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'prices' not in data:
                return None
            
            # Convert to DataFrame
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            
            # Merge data
            df = pd.merge(prices_df, volumes_df, on='timestamp')
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            
            # Add OHLC data (approximated from close prices for daily data)
            df['open'] = df['close'].shift(1)
            df['high'] = df[['open', 'close']].max(axis=1)
            df['low'] = df[['open', 'close']].min(axis=1)
            
            # Fill NaN values
            df['open'].fillna(df['close'], inplace=True)
            
            # Reorder columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None


class DataProcessor:
    """Data processing and technical indicators"""
    
    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        df = data.copy()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Handle missing values
        df.fillna(method='forward', inplace=True)
        df.fillna(method='backward', inplace=True)
        
        # Remove outliers (prices that are 0 or negative)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df = df[df[col] > 0]
        
        return df
    
    @staticmethod
    def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        df['rsi'] = DataProcessor._calculate_rsi(df['close'])
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Daily returns
        df['returns'] = df['close'].pct_change()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class TimeSeriesAnalysis:
    """Time series analysis methods"""
    
    @staticmethod
    def decompose_time_series(data: pd.Series, period: int = 30):
        """Decompose time series into trend, seasonal, and residual components"""
        try:
            # Ensure we have enough data points
            if len(data) < 2 * period:
                period = max(2, len(data) // 4)
            
            # Perform decomposition
            decomposition = seasonal_decompose(
                data.dropna(), 
                model='multiplicative', 
                period=period,
                extrapolate_trend='freq'
            )
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                vertical_spacing=0.08
            )
            
            # Original data
            fig.add_trace(
                go.Scatter(x=data.index, y=data.values, name='Original', 
                          line=dict(color='blue')), row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, 
                          name='Trend', line=dict(color='red')), row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, 
                          name='Seasonal', line=dict(color='green')), row=3, col=1
            )
            
            # Residual
            fig.add_trace(
                go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, 
                          name='Residual', line=dict(color='orange')), row=4, col=1
            )
            
            fig.update_layout(
                height=800,
                title_text="Time Series Decomposition",
                showlegend=False,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error in decomposition: {str(e)}")
            return None
    
    @staticmethod
    def adf_test(timeseries: pd.Series) -> Dict:
        """Perform Augmented Dickey-Fuller test"""
        try:
            result = adfuller(timeseries.dropna())
            
            return {
                'adf_stat': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except Exception as e:
            return {
                'adf_stat': None,
                'p_value': None,
                'critical_values': None,
                'is_stationary': False,
                'error': str(e)
            }
    
    @staticmethod
    def plot_acf_pacf(data: pd.Series, lags: int = 40):
        """Plot ACF and PACF"""
        try:
            # Create matplotlib plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # ACF plot
            plot_acf(data.dropna(), lags=lags, ax=ax1, title='Autocorrelation Function')
            
            # PACF plot  
            plot_pacf(data.dropna(), lags=lags, ax=ax2, title='Partial Autocorrelation Function')
            
            plt.tight_layout()
            
            # Convert to Plotly
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Convert to base64 for display
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            st.error(f"Error creating ACF/PACF plots: {str(e)}")
            return None


class ARIMAForecaster:
    """ARIMA model for time series forecasting"""
    
    @staticmethod
    def forecast(data: pd.Series, forecast_days: int = 30) -> Dict:
        """Generate ARIMA forecast"""
        try:
            # Prepare data
            train_data = data.dropna()
            
            if len(train_data) < 50:
                raise ValueError("Insufficient data for ARIMA modeling")
            
            # Fit ARIMA model (using auto parameters for simplicity)
            model = ARIMA(train_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_result = fitted_model.forecast(steps=forecast_days)
            conf_int = fitted_model.get_forecast(steps=forecast_days).conf_int()
            
            # Create forecast dates
            last_date = train_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'forecast': forecast_result,
                'lower_ci': conf_int.iloc[:, 0],
                'upper_ci': conf_int.iloc[:, 1]
            }, index=forecast_dates)
            
            # Create chart
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=train_data.index[-100:],  # Last 100 points
                y=train_data.values[-100:],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['upper_ci'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['lower_ci'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(width=0),
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title=f'ARIMA Forecast ({forecast_days} days)',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_white'
            )
            
            # Calculate basic metrics (on training data)
            residuals = fitted_model.resid
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            
            return {
                'chart': fig,
                'forecast_data': forecast_df,
                'model_summary': fitted_model.summary(),
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
            }
            
        except Exception as e:
            st.error(f"Error in ARIMA forecasting: {str(e)}")
            return None


def format_currency(value: float) -> str:
    """Format currency values"""
    if abs(value) >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate price returns"""
    return prices.pct_change().dropna()


class CryptoDashboard:
    """Main dashboard class for cryptocurrency analysis"""
    
    def __init__(self):
        self.data_fetcher = CryptoDataFetcher()
        self.data_processor = DataProcessor()
        self.ts_analysis = TimeSeriesAnalysis()
        self.arima_forecaster = ARIMAForecaster()
        
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
            
            if len(data) > 1:
                price_change = ((data['close'].iloc[-1] / data['close'].iloc[-2]) - 1) * 100
            else:
                price_change = 0
            
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
                high_52w = data['high'].rolling(window=min(365, len(data))).max().iloc[-1]
                st.metric(
                    label="52W High",
                    value=format_currency(high_52w)
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
                if pd.notna(latest_rsi):
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
            if decomposition_fig:
                st.plotly_chart(decomposition_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in decomposition: {str(e)}")
        
        # Stationarity test
        st.subheader("📊 Stationarity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            adf_result = self.ts_analysis.adf_test(data['close'])
            
            st.write("**Augmented Dickey-Fuller Test Results:**")
            if 'error' not in adf_result:
                st.write(f"ADF Statistic: {adf_result['adf_stat']:.4f}")
                st.write(f"p-value: {adf_result['p_value']:.4f}")
                st.write(f"Critical Values: {adf_result['critical_values']}")
                
                if adf_result['p_value'] < 0.05:
                    st.success("✅ Series is stationary")
                else:
                    st.warning("⚠️ Series is non-stationary")
            else:
                st.error(f"Error in ADF test: {adf_result['error']}")
        
        with col2:
            # Autocorrelation plots
            acf_img = self.ts_analysis.plot_acf_pacf(data['close'])
            if acf_img:
                st.image(f"data:image/png;base64,{acf_img}")
    
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
            model_type = st.selectbox("Model Type", ["ARIMA"], index=0)
        
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
                forecast_result = self.arima_forecaster.forecast(
                    data['close'], forecast_days
                )
                
                if forecast_result:
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
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            
            with col2:
                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
            
            with col3:
                st.metric("AIC", f"{metrics.get('aic', 0):.2f}")
            
            with col4:
                st.metric("BIC", f"{metrics.get('bic', 0):.2f}")
        
        # Download forecast data
        if st.button("📥 Download Forecast Data"):
            forecast_df = forecast_data['forecast_data']
            csv = forecast_df.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"crypto_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
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