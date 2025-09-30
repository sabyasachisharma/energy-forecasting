"""
SQLAlchemy models for TimescaleDB Tiger Cloud.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class EnergyPrice(Base):
    """Energy price data model for TimescaleDB hypertable (time-series data)."""
    __tablename__ = 'energy_prices'
    
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    price = Column(Float, nullable=False)

class HistoricalEnergyPrices(Base):
    """Historical energy prices - only essential hourly columns."""
    __tablename__ = 'historical_energy_prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    delivery_day = Column(String, nullable=False, unique=True, index=True)
    
    hour_1 = Column(Float)
    hour_2 = Column(Float)
    hour_3a = Column(Float)
    hour_3b = Column(Float)
    hour_4 = Column(Float)
    hour_5 = Column(Float)
    hour_6 = Column(Float)
    hour_7 = Column(Float)
    hour_8 = Column(Float)
    hour_9 = Column(Float)
    hour_10 = Column(Float)
    hour_11 = Column(Float)
    hour_12 = Column(Float)
    hour_13 = Column(Float)
    hour_14 = Column(Float)
    hour_15 = Column(Float)
    hour_16 = Column(Float)
    hour_17 = Column(Float)
    hour_18 = Column(Float)
    hour_19 = Column(Float)
    hour_20 = Column(Float)
    hour_21 = Column(Float)
    hour_22 = Column(Float)
    hour_23 = Column(Float)
    hour_24 = Column(Float)
    
    def to_hourly_series(self):
        """Convert daily row to hourly time series data with DST handling."""
        from datetime import datetime, timedelta
        import pandas as pd
        
        base_date = datetime.strptime(self.delivery_day, '%Y-%m-%d')
        
        hourly_data = []
        
        for hour in range(1, 3):
            price = getattr(self, f'hour_{hour}', None)
            if price is not None:
                timestamp = base_date + timedelta(hours=hour-1)
                hourly_data.append({
                    'timestamp': timestamp,
                    'price': price
                })
        
        price_3a = getattr(self, 'hour_3a', None)
        price_3b = getattr(self, 'hour_3b', None)
        
        if price_3a is not None:
            timestamp_3a = base_date + timedelta(hours=2)
            hourly_data.append({
                'timestamp': timestamp_3a,
                'price': price_3a
            })
        
        if price_3b is not None and price_3b != 0:
            timestamp_3b = base_date + timedelta(hours=2, minutes=30)
            hourly_data.append({
                'timestamp': timestamp_3b,
                'price': price_3b,
                'dst_duplicate': True
            })
        
        for hour in range(4, 25):
            price = getattr(self, f'hour_{hour}', None)
            if price is not None:
                timestamp = base_date + timedelta(hours=hour-1)
                hourly_data.append({
                    'timestamp': timestamp,
                    'price': price
                })
        
        return pd.DataFrame(hourly_data).set_index('timestamp') if hourly_data else pd.DataFrame()
    
    def get_peak_hours(self, threshold_percentile=75):
        """Get peak hour indicators based on price threshold."""
        hourly_prices = []
        
        for hour in range(1, 3):
            price = getattr(self, f'hour_{hour}', None)
            if price is not None:
                hourly_prices.append((hour, price))
        
        price_3a = getattr(self, 'hour_3a', None)
        if price_3a is not None:
            hourly_prices.append((3, price_3a))
        
        price_3b = getattr(self, 'hour_3b', None)
        if price_3b is not None and price_3b != 0:
            hourly_prices.append((3.5, price_3b))
        
        for hour in range(4, 25):
            price = getattr(self, f'hour_{hour}', None)
            if price is not None:
                hourly_prices.append((hour, price))
        
        if not hourly_prices:
            return []
        
        import numpy as np
        prices = [p[1] for p in hourly_prices]
        threshold = np.percentile(prices, threshold_percentile)
        
        return [hour for hour, price in hourly_prices if price >= threshold]
