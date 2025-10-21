import pandas as pd
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .database import load_prices_from_db, check_db_connection, load_historical_prices_from_db
from .config import use_database

try:
    from .ml_models import ModelManager, XGBoostForecaster, LSTMForecaster, train_ensemble_models
    HAS_ML_MODELS = True
except ImportError as e:
    HAS_ML_MODELS = False
    logging.warning(f"ML models not available: {e}")

def load_prices() -> pd.DataFrame:
    """Load energy price data from TimescaleDB"""
    if not use_database() or not check_db_connection():
        raise Exception("TimescaleDB connection required but not available")
    
    try:
        logging.info("Loading prices from TimescaleDB")
        df = load_prices_from_db()
        if not df.empty:
            return df
        else:
            raise Exception("No data found in TimescaleDB")
    except Exception as e:
        raise Exception(f"Failed to load data from TimescaleDB: {e}")

def load_recent_prices(hours: int = 168) -> pd.DataFrame:
    """Load recent price data for forecasting from TimescaleDB"""
    if not use_database() or not check_db_connection():
        raise Exception("TimescaleDB connection required but not available")
    
    try:
        from datetime import datetime, timedelta
        
        # Calculate time window based on hours parameter
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        logging.info(f"Loading recent prices from TimescaleDB from {start_time} to {end_time} ({hours} hours)")
        
        df = load_prices_from_db(start_time=start_time, end_time=end_time)
        if not df.empty:
            logging.info(f"Found {len(df)} hourly records in recent {hours} hours")
            return df
        else:
            # If no data in the time range, get recent historical data
            days = max(7, hours // 24 + 1)
            logging.info(f"No recent data found, loading most recent {days} days from database")
            df = load_prices_from_db(limit=days)
            if not df.empty:
                logging.info(f"Loaded {len(df)} hourly records from historical data")
                return df
            else:
                raise Exception("No recent data available in TimescaleDB")
    except Exception as e:
        raise Exception(f"Failed to load recent prices from TimescaleDB: {e}")

def seasonal_naive_forecast(df: pd.DataFrame, horizon_hours: int = 24, season: int = 24) -> pd.Series:
    """Repeat the value from same hour last season"""
    y = df["price"]
    last_idx = y.index[-1]
    future_idx = pd.date_range(start=last_idx + pd.Timedelta(hours=1), periods=horizon_hours, freq="h", tz=y.index.tz)
    y_extended = pd.concat([y, pd.Series(index=future_idx, dtype=float)])
    yhat = y_extended.shift(season)
    return yhat.reindex(future_idx)

def sarimax_forecast(df: pd.DataFrame, horizon_hours: int = 24, s: int = 24):
    """Fit a simple SARIMAX and forecast. Falls back to seasonal naive if fit fails"""
    y = df["price"]
    try:
        model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,s), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=horizon_hours).predicted_mean
        return fc
    except Exception:
        return seasonal_naive_forecast(df, horizon_hours, s)

def recommend_slots(forecast: pd.Series, k: int = 3, run_hours: int = 2):
    """Find k cheapest non-overlapping windows of length run_hours"""
    prices = forecast.copy()
    windows = []
    roll = prices.rolling(run_hours).mean().dropna()
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

def ml_forecast(df: pd.DataFrame, horizon_hours: int = 24, model_type: str = 'auto') -> pd.Series:
    """Generate forecasts using ML models"""
    if not HAS_ML_MODELS:
        logging.warning("ML models not available, falling back to SARIMAX")
        return sarimax_forecast(df, horizon_hours, s=24)
    
    try:
        manager = ModelManager()
        
        if model_type == 'auto':
            # Use the best available model
            model = manager.get_best_model(metric='rmse')
            if model is None:
                logging.info("No trained models found, training new ensemble")
                train_ensemble_models(df)
                model = manager.get_best_model(metric='rmse')
        elif model_type == 'xgboost':
            # Train new XGBoost model
            model = XGBoostForecaster()
            model.train(df)
        elif model_type == 'lstm':
            # Train new LSTM model
            model = LSTMForecaster()
            model.train(df, epochs=50)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model is None:
            logging.warning("No ML model available, falling back to SARIMAX")
            return sarimax_forecast(df, horizon_hours, s=24)
        
        # Generate forecast
        forecast = model.forecast(df, horizon_hours)
        return forecast
        
    except Exception as e:
        logging.error(f"ML forecasting failed: {e}, falling back to SARIMAX")
        return sarimax_forecast(df, horizon_hours, s=24)

def hybrid_forecast(df: pd.DataFrame, horizon_hours: int = 24) -> pd.Series:
    """Hybrid forecast combining SARIMAX trend with ML residuals"""
    try:
        # Get SARIMAX baseline
        sarimax_pred = sarimax_forecast(df, horizon_hours, s=24)
        
        if not HAS_ML_MODELS:
            return sarimax_pred
        
        # Calculate SARIMAX residuals on training data
        y = df["price"]
        try:
            model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,24), 
                          enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            fitted_values = res.fittedvalues
            residuals = y - fitted_values
            
            # Create residuals DataFrame
            residuals_df = pd.DataFrame({'price': residuals}, index=residuals.index)
            residuals_df = residuals_df.dropna()
            
            if len(residuals_df) > 100:  # Need enough data for ML
                # Train ML model on residuals
                xgb_model = XGBoostForecaster()
                xgb_model.train(residuals_df, target_col='price')
                
                # Forecast residuals
                residual_forecast = xgb_model.forecast(residuals_df, horizon_hours, target_col='price')
                
                # Combine SARIMAX + ML residuals
                combined_forecast = sarimax_pred + residual_forecast
                return combined_forecast
            
        except Exception as e:
            logging.warning(f"Hybrid forecasting failed: {e}")
        
        return sarimax_pred
        
    except Exception as e:
        logging.error(f"Hybrid forecasting error: {e}")
        return sarimax_forecast(df, horizon_hours, s=24)
