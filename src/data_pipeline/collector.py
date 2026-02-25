"""
Data Collection Pipeline
Multi-source financial data with fallback strategy
Primary: yfinance (free)
Fallback: Alpha Vantage (if needed)
Cache: Local storage for reliability
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class DataCollector:
    """
    Collect financial data from multiple sources with fallback
    Implements resilient data pipeline
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = 24
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get historical stock data with multi-source fallback
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            DataFrame with OHLCV data
        """
        
        logger.info(f"Fetching data for {ticker} ({period})")
        
        # Try primary source (yfinance)
        try:
            data = self._fetch_yfinance(ticker, period)
            if self._validate_data(data):
                self._cache_data(ticker, data, 'historical')
                logger.info(f"✓ Successfully fetched {ticker} from yfinance")
                return data
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {e}")
        
        # Try cache as fallback
        cached_data = self._get_cached_data(ticker, 'historical')
        if cached_data is not None:
            logger.info(f"✓ Loaded {ticker} from cache (stale data)")
            return cached_data
        
        logger.error(f"✗ Failed to fetch data for {ticker}")
        return None
    
    def get_company_info(self, ticker: str) -> Optional[Dict]:
        """
        Get company fundamental information
        
        Returns:
            Dict with company metrics
        """
        
        logger.info(f"Fetching company info for {ticker}")
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'symbol' not in info:
                raise ValueError("Invalid company data")
            
            # Extract key metrics
            company_data = {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'business_summary': info.get('longBusinessSummary', ''),
                'updated_at': datetime.now()
            }
            
            self._cache_data(ticker, company_data, 'company_info')
            logger.info(f"✓ Successfully fetched info for {ticker}")
            
            return company_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch company info for {ticker}: {e}")
            
            # Try cache
            cached = self._get_cached_data(ticker, 'company_info')
            if cached:
                logger.info(f"✓ Loaded {ticker} info from cache")
                return cached
            
            return None
    
    def get_multiple_stocks(self, tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Returns:
            Dict of {ticker: DataFrame}
        """
        
        logger.info(f"Fetching data for {len(tickers)} stocks")
        
        results = {}
        for ticker in tickers:
            data = self.get_stock_data(ticker, period)
            if data is not None:
                results[ticker] = data
        
        logger.info(f"✓ Successfully fetched {len(results)}/{len(tickers)} stocks")
        return results
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculate daily returns from price data
        
        Args:
            prices: DataFrame with 'Close' column
        
        Returns:
            Series of daily returns
        """
        
        if 'Close' not in prices.columns:
            raise ValueError("Price data must have 'Close' column")
        
        returns = prices['Close'].pct_change().dropna()
        return returns
    
    def get_market_benchmark(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get S&P 500 data as market benchmark
        
        Returns:
            DataFrame with SPY data
        """
        
        return self.get_stock_data("SPY", period)
    
    def generate_sample_esg_data(self, ticker: str, sector: str) -> Dict:
        """
        Generate realistic ESG data for demo purposes
        Based on sector averages
        
        In production, this would integrate with:
        - MSCI ESG API
        - Sustainalytics
        - Bloomberg ESG data
        - SEC filings scraper
        """
        
        # Sector-based baseline scores
        sector_baselines = {
            'Technology': {'E': 65, 'S': 70, 'G': 75},
            'Energy': {'E': 45, 'S': 55, 'G': 60},
            'Healthcare': {'E': 60, 'S': 75, 'G': 70},
            'Financials': {'E': 55, 'S': 65, 'G': 80},
            'Consumer Discretionary': {'E': 60, 'S': 65, 'G': 65},
            'Industrials': {'E': 50, 'S': 60, 'G': 65},
            'Materials': {'E': 45, 'S': 55, 'G': 60},
            'Utilities': {'E': 50, 'S': 60, 'G': 70},
        }
        
        baseline = sector_baselines.get(sector, {'E': 60, 'S': 60, 'G': 60})
        
        # Add some realistic variation
        np.random.seed(hash(ticker) % 2**32)
        
        esg_data = {
            # Environmental
            'carbon_intensity': np.random.uniform(30, 150),
            'renewable_energy_pct': baseline['E'] + np.random.uniform(-15, 15),
            'water_usage': np.random.uniform(200, 800),
            'waste_recycling_pct': baseline['E'] + np.random.uniform(-10, 20),
            'environmental_innovations': int(np.random.uniform(0, 10)),
            
            # Social
            'employee_satisfaction': baseline['S'] + np.random.uniform(-10, 15),
            'diversity_score': baseline['S'] + np.random.uniform(-15, 10),
            'female_employees_pct': np.random.uniform(25, 50),
            'employee_turnover_rate': np.random.uniform(5, 20),
            'training_hours_per_employee': np.random.uniform(20, 60),
            'community_investment': np.random.uniform(1000000, 10000000),
            'labor_practices_score': baseline['S'] + np.random.uniform(-10, 10),
            'human_rights_score': baseline['S'] + np.random.uniform(-5, 15),
            
            # Governance
            'board_independence': np.random.uniform(50, 85),
            'board_diversity': np.random.uniform(30, 70),
            'female_board_members': np.random.uniform(15, 45),
            'executive_compensation_ratio': np.random.uniform(50, 250),
            'shareholder_rights_score': baseline['G'] + np.random.uniform(-10, 10),
            'anti_corruption_score': baseline['G'] + np.random.uniform(-5, 15),
            'tax_transparency_score': baseline['G'] + np.random.uniform(-10, 10),
            
            # Controversies
            'esg_controversies': int(np.random.binomial(5, 0.1))
        }
        
        # Ensure all values are within valid ranges
        for key in ['renewable_energy_pct', 'waste_recycling_pct', 'employee_satisfaction',
                    'diversity_score', 'labor_practices_score', 'human_rights_score',
                    'shareholder_rights_score', 'anti_corruption_score', 'tax_transparency_score']:
            esg_data[key] = max(0, min(100, esg_data[key]))
        
        logger.info(f"Generated ESG data for {ticker} ({sector})")
        
        return esg_data
    
    def _fetch_yfinance(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch data from yfinance"""
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fetched data"""
        if data is None or data.empty:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        if len(data) < 10:  # At least 10 data points
            return False
        
        return True
    
    def _cache_data(self, ticker: str, data, data_type: str):
        """Cache data locally"""
        try:
            cache_file = self.cache_dir / f"{ticker}_{data_type}.json"
            
            cache_obj = {
                'ticker': ticker,
                'data_type': data_type,
                'timestamp': datetime.now().isoformat(),
                'data': data.to_dict() if isinstance(data, pd.DataFrame) else data
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_obj, f, default=str)
            
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    def _get_cached_data(self, ticker: str, data_type: str):
        """Retrieve cached data if not expired"""
        try:
            cache_file = self.cache_dir / f"{ticker}_{data_type}.json"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r') as f:
                cache_obj = json.load(f)
            
            # Check if cache is expired
            cached_time = datetime.fromisoformat(cache_obj['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.cache_expiry_hours):
                logger.info(f"Cache expired for {ticker}")
                return None
            
            # Convert back to DataFrame if needed
            data = cache_obj['data']
            if data_type == 'historical' and isinstance(data, dict):
                data = pd.DataFrame(data)
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None


# Example usage
if __name__ == "__main__":
    collector = DataCollector()
    
    # Test single stock
    print("\n=== Testing Single Stock ===")
    data = collector.get_stock_data("AAPL", period="6mo")
    if data is not None:
        print(f"✓ Fetched {len(data)} days of data for AAPL")
        print(data.head())
    
    # Test company info
    print("\n=== Testing Company Info ===")
    info = collector.get_company_info("AAPL")
    if info:
        print(f"✓ Company: {info['company_name']}")
        print(f"  Sector: {info['sector']}")
        print(f"  Market Cap: ${info['market_cap']:,.0f}")
    
    # Test ESG data generation
    print("\n=== Testing ESG Data ===")
    esg = collector.generate_sample_esg_data("AAPL", "Technology")
    print(f"✓ Generated ESG data")
    print(f"  Renewable Energy: {esg['renewable_energy_pct']:.1f}%")
    print(f"  Employee Satisfaction: {esg['employee_satisfaction']:.1f}/100")