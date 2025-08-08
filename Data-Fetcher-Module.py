"""
Cryptocurrency Data Fetcher Module
Handles data acquisition from various cryptocurrency data sources
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
import streamlit as st
from typing import Optional, Dict, List


class CryptoDataFetcher:
    """
    Cryptocurrency data fetcher class supporting multiple data sources
    """
    
    def __init__(self):
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoAnalysisApp/1.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        
        # Cryptocurrency mapping for different sources
        self.crypto_mapping = {
            'bitcoin': {'yf': 'BTC-USD', 'cg': 'bitcoin'},
            'ethereum': {'yf': 'ETH-USD', 'cg': 'ethereum'},
            'binancecoin': {'yf': 'BNB-USD', 'cg': 'binancecoin'},
            'cardano': {'yf': 'ADA-USD', 'cg': 'cardano'},
            'solana': {'yf': 'SOL-USD', 'cg': 'solana'},
            'polkadot': {'yf': 'DOT-USD', 'cg': 'polkadot'},
            'dogecoin': {'yf': 'DOGE-USD', 'cg': 'dogecoin'},
            'avalanche-2': {'yf': 'AVAX-USD', 'cg': 'avalanche-2'}
        }
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_crypto_data(self, crypto_id: str, start_date: datetime, 
                       end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch cryptocurrency data from multiple sources with fallback
        
        Args:
            crypto_id: Cryptocurrency identifier
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Try Yahoo Finance first (more reliable)
            data = self._fetch_from_yahoo_finance(crypto_id, start_date, end_date)
            
            if data is not None and not data.empty:
                return data
            
            # Fallback to CoinGecko
            st.warning("Yahoo Finance failed, trying CoinGecko...")
            data = self._fetch_from_coingecko(crypto_id, start_date, end_date)
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def _fetch_from_yahoo_finance(self, crypto_id: str, start_date: datetime, 
                                 end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance using yfinance
        
        Args:
            crypto_id: Cryptocurrency identifier
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            if crypto_id not in self.crypto_mapping:
                return None
            
            yf_symbol = self.crypto_mapping[crypto_id]['yf']
            
            # Download data
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return None
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only required columns
            columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
            data = data[columns_to_keep]
            
            # Handle timezone-aware index
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            return data
            
        except Exception as e:
            print(f"Yahoo Finance error: {str(e)}")
            return None
    
    def _fetch_from_coingecko(self, crypto_id: str, start_date: datetime, 
                             end_date: datetime) -> Optional[pd.DataFrame]:
        """