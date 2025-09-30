
import pandas as pd
import numpy as np
import os
import logging
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Optional
from .database import load_prices_from_db, check_db_connection, load_historical_prices_from_db
from .config import use_database

def load_prices() -> pd.DataFrame:
    """
    Load energy price data from TimescaleDB Tiger Cloud.
    
    Returns:
        DataFrame with timestamp index and price column
    """
    if not use_database() or not check_db_connection():
        raise Exception("TimescaleDB Tiger Cloud connection required but not available")
    
    try:
        logging.info("Loading prices from TimescaleDB Tiger Cloud")
        df = load_prices_from_db()
        if not df.empty:
            return df
        else:
            logging.info("No time-series data found, trying historical data conversion")
            df = convert_historical_to_timeseries()
            if not df.empty:
                return df
            else:
                raise Exception("No data found in TimescaleDB Tiger Cloud")
    except Exception as e:
        raise Exception(f"Failed to load data from TimescaleDB Tiger Cloud: {e}")

def load_recent_prices(hours: int = 168) -> pd.DataFrame:
    """
    Load recent price data for forecasting from TimescaleDB Tiger Cloud.
    
    Args:
        hours: Number of recent hours to load (default: 1 week)
        
    Returns:
        DataFrame with recent price data
    """
    if not use_database() or not check_db_connection():
        raise Exception("TimescaleDB Tiger Cloud connection required but not available")
    
    try:
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        df = load_prices_from_db(start_time=start_time, end_time=end_time)
        if not df.empty:
            return df
        else:
            logging.info("No time-series data found, using recent historical data")
            days = max(7, hours // 24 + 1)
            df = convert_historical_to_timeseries(days=days)
            if not df.empty:
                return df.tail(hours) if len(df) > hours else df
            else:
                raise Exception("No recent data available in TimescaleDB Tiger Cloud")
    except Exception as e:
        raise Exception(f"Failed to load recent prices from TimescaleDB Tiger Cloud: {e}")

def seasonal_naive_forecast(df: pd.DataFrame, horizon_hours: int = 24, season: int = 24) -> pd.Series:
    """Repeat the value from same hour last season."""
    y = df["price"]
    last_idx = y.index[-1]
    future_idx = pd.date_range(start=last_idx + pd.Timedelta(hours=1), periods=horizon_hours, freq="h", tz=y.index.tz)
    y_extended = pd.concat([y, pd.Series(index=future_idx, dtype=float)])
    yhat = y_extended.shift(season)
    return yhat.reindex(future_idx)

def sarimax_forecast(df: pd.DataFrame, horizon_hours: int = 24, s: int = 24):
    """Fit a simple SARIMAX and forecast. Falls back to seasonal naive if fit fails."""
    y = df["price"]
    try:
        model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,s), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=horizon_hours).predicted_mean
        return fc
    except Exception as e:
        return seasonal_naive_forecast(df, horizon_hours, s)

def recommend_slots(forecast: pd.Series, k: int = 3, run_hours: int = 2):
    """
    Find k cheapest non-overlapping windows of length run_hours.
    Returns list of dicts with start, end, avg_price.
    """
    prices = forecast.copy()
    windows = []
    roll = prices.rolling(run_hours).mean().dropna()
    used = set()
    for _ in range(k):
        if roll.empty:
            break
        best_start = roll.idxmin()
        best_mean = roll.loc[best_start]
        best_end = best_start + pd.Timedelta(hours=run_hours - 1)
        windows.append({"start": best_start.isoformat(), "end": best_end.isoformat(), "avg_price": float(round(best_mean, 2))})
        mask = (roll.index >= best_start - pd.Timedelta(hours=run_hours-1)) & (roll.index <= best_end)
        roll = roll[~mask]
    return windows

def convert_historical_to_timeseries(days: int = 30) -> pd.DataFrame:
    """
    Convert historical daily data to hourly time-series format for forecasting.
    
    Args:
        days: Number of recent days to convert
        
    Returns:
        DataFrame with hourly time-series data
    """
    try:
        df_daily = load_historical_prices_from_db(limit=days)
        if df_daily.empty:
            return pd.DataFrame()
        
        all_hourly = []
        
        for _, row in df_daily.iterrows():
            base_date = pd.to_datetime(row['delivery_day'])
            
            for hour in range(1, 25):
                if hour == 3:
                    if pd.notna(row.get('hour_3a')):
                        timestamp = base_date + pd.Timedelta(hours=2)
                        all_hourly.append({'timestamp': timestamp, 'price': float(row['hour_3a'])})
                    
                    if pd.notna(row.get('hour_3b')) and row.get('hour_3b', 0) != 0:
                        timestamp = base_date + pd.Timedelta(hours=2, minutes=30)
                        all_hourly.append({'timestamp': timestamp, 'price': float(row['hour_3b'])})
                else:
                    hour_col = f'hour_{hour}'
                    if hour_col in row and pd.notna(row[hour_col]):
                        timestamp = base_date + pd.Timedelta(hours=hour-1)
                        all_hourly.append({'timestamp': timestamp, 'price': float(row[hour_col])})
        
        if not all_hourly:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_hourly)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/Berlin')
        
        df = df.sort_values('timestamp').set_index('timestamp')
        df = df.asfreq('h', method='ffill')
        
        logging.info(f"Converted {len(df_daily)} daily records to {len(df)} hourly records for forecasting")
        return df
        
    except Exception as e:
        logging.error(f"Failed to convert historical data to time-series: {e}")
        return pd.DataFrame()
