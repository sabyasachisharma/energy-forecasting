"""
TimescaleDB Tiger Cloud database operations for energy pricing data.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging
from sqlalchemy import create_engine, text, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from .config import get_database_url
from .models import Base, EnergyPrice, HistoricalEnergyPrices

_engine = None
_SessionLocal = None

def get_engine():
    """Get SQLAlchemy engine for TimescaleDB Tiger Cloud."""
    global _engine
    if _engine is None:
        database_url = get_database_url()
        _engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=False
        )
        logging.info("Created TimescaleDB Tiger Cloud engine")
    return _engine

def get_session_local():
    """Get SQLAlchemy session local for Tiger Cloud."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logging.info("Created TimescaleDB session local")
    return _SessionLocal

def get_db_session() -> Session:
    """Get database session."""
    SessionLocal = get_session_local()
    return SessionLocal()

def init_database() -> bool:
    """
    Initialize TimescaleDB Tiger Cloud with hypertables and tables.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        engine = get_engine()
        
        Base.metadata.create_all(bind=engine)
        
        with engine.connect() as conn:
            conn.execute(text("""
                SELECT create_hypertable('energy_prices', 'timestamp', if_not_exists => TRUE);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_historical_energy_prices_delivery_day 
                ON historical_energy_prices (delivery_day);
            """))
            
            conn.commit()
        
        logging.info("TimescaleDB Tiger Cloud initialized successfully with historical_energy_prices table")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize TimescaleDB Tiger Cloud: {e}")
        return False

def check_db_connection() -> bool:
    """Check if TimescaleDB Tiger Cloud connection is working."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            
        logging.info("TimescaleDB Tiger Cloud connection successful")
        return True
        
    except Exception as e:
        logging.error(f"TimescaleDB Tiger Cloud connection failed: {e}")
        return False

def load_prices_from_db(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load energy prices from TimescaleDB Tiger Cloud.
    
    Args:
        start_time: Start timestamp (optional)
        end_time: End timestamp (optional) 
        limit: Maximum number of records (optional)
    
    Returns:
        DataFrame with timestamp index and price column
    """
    try:
        session = get_db_session()
        
        query = session.query(EnergyPrice)
        
        if start_time:
            query = query.filter(EnergyPrice.timestamp >= start_time)
        if end_time:
            query = query.filter(EnergyPrice.timestamp <= end_time)
        
        query = query.order_by(EnergyPrice.timestamp)
        
        if limit:
            query = query.limit(limit)
        
        results = query.all()
        session.close()
        
        if not results:
            logging.warning("No price data found in TimescaleDB Tiger Cloud")
            return pd.DataFrame(columns=['price'])
        
        df = pd.DataFrame([
            {'timestamp': record.timestamp, 'price': record.price}
            for record in results
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/Berlin')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Berlin')
        
        df = df.sort_values('timestamp').set_index('timestamp')
        df = df.asfreq('h', method='ffill')
        df['price'] = df['price'].astype(float)
        
        logging.info(f"Loaded {len(df)} price records from TimescaleDB Tiger Cloud")
        return df
        
    except Exception as e:
        logging.error(f"Failed to load prices from TimescaleDB Tiger Cloud: {e}")
        return pd.DataFrame(columns=['price'])

def insert_prices_to_db(df: pd.DataFrame) -> bool:
    """
    Insert price data into TimescaleDB Tiger Cloud using upsert.
    
    Args:
        df: DataFrame with timestamp index and price column
        
    Returns:
        True if successful, False otherwise
    """
    try:
        session = get_db_session()
        
        records = []
        for timestamp, row in df.iterrows():
            records.append({
                'timestamp': timestamp,
                'price': float(row['price'])
            })
        
        session.execute(text("""
            INSERT INTO energy_prices (timestamp, price) 
            VALUES (:timestamp, :price)
            ON CONFLICT (timestamp) 
            DO UPDATE SET price = EXCLUDED.price
        """), records)
        
        session.commit()
        session.close()
        
        logging.info(f"Successfully inserted {len(records)} price records to TimescaleDB Tiger Cloud")
        return True
        
    except Exception as e:
        logging.error(f"Failed to insert prices to TimescaleDB Tiger Cloud: {e}")
        return False

def get_price_count() -> int:
    """Get total number of price records in TimescaleDB Tiger Cloud."""
    try:
        session = get_db_session()
        count = session.query(EnergyPrice).count()
        session.close()
        return count
        
    except Exception as e:
        logging.error(f"Failed to get price count from TimescaleDB Tiger Cloud: {e}")
        return 0

def get_latest_timestamp() -> Optional[datetime]:
    """Get the timestamp of the latest price record in TimescaleDB Tiger Cloud."""
    try:
        session = get_db_session()
        result = session.query(EnergyPrice.timestamp).order_by(EnergyPrice.timestamp.desc()).first()
        session.close()
        
        return result[0] if result else None
        
    except Exception as e:
        logging.error(f"Failed to get latest timestamp from TimescaleDB Tiger Cloud: {e}")
        return None

def get_price_stats() -> dict:
    """Get comprehensive statistics about the price data in TimescaleDB Tiger Cloud."""
    try:
        session = get_db_session()
        
        from sqlalchemy import func
        
        stats_query = session.query(
            func.count(EnergyPrice.timestamp).label('count'),
            func.min(EnergyPrice.timestamp).label('min_timestamp'),
            func.max(EnergyPrice.timestamp).label('max_timestamp'),
            func.min(EnergyPrice.price).label('min_price'),
            func.max(EnergyPrice.price).label('max_price'),
            func.avg(EnergyPrice.price).label('avg_price')
        ).first()
        
        if stats_query.count == 0:
            session.close()
            return {}
        
        std_result = session.execute(text("""
            SELECT stddev(price) as price_std FROM energy_prices
        """)).fetchone()
        
        session.close()
        
        stats = {
            "total_records": int(stats_query.count),
            "date_range": {
                "start": stats_query.min_timestamp,
                "end": stats_query.max_timestamp
            },
            "price_stats": {
                "min": float(stats_query.min_price),
                "max": float(stats_query.max_price),
                "mean": float(stats_query.avg_price),
                "std": float(std_result[0]) if std_result and std_result[0] else 0.0
            }
        }
        
        return stats
        
    except Exception as e:
        logging.error(f"Failed to get price stats from TimescaleDB Tiger Cloud: {e}")
        return {}

def optimize_database():
    """Optimize the TimescaleDB Tiger Cloud for better performance."""
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_energy_prices_timestamp 
                ON energy_prices (timestamp DESC);
            """))
            
            conn.execute(text("ANALYZE energy_prices;"))
            
            conn.commit()
        
        logging.info("TimescaleDB Tiger Cloud optimization completed")
        return True
        
    except Exception as e:
        logging.error(f"Failed to optimize TimescaleDB Tiger Cloud: {e}")
        return False

def cleanup_old_data(days_to_keep: int = 365):
    """
    Clean up old data from TimescaleDB Tiger Cloud.
    
    Args:
        days_to_keep: Number of days of data to keep
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        session = get_db_session()
        deleted_count = session.query(EnergyPrice).filter(
            EnergyPrice.timestamp < cutoff_date
        ).delete()
        
        session.commit()
        session.close()
        
        logging.info(f"Cleaned up {deleted_count} old price records from TimescaleDB Tiger Cloud")
        return deleted_count
        
    except Exception as e:
        logging.error(f"Failed to cleanup old data from TimescaleDB Tiger Cloud: {e}")
        return 0

# Daily Spot Price Operations

def insert_historical_prices_to_db(df: pd.DataFrame) -> bool:
    """
    Insert historical energy prices into TimescaleDB Tiger Cloud.
    
    Args:
        df: DataFrame with historical energy price data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        session = get_db_session()
        
        records_inserted = 0
        records_updated = 0
        
        for _, row in df.iterrows():
            existing = session.query(HistoricalEnergyPrices).filter(
                HistoricalEnergyPrices.delivery_day == str(row['delivery_day'])
            ).first()
            
            if existing:
                for column in df.columns:
                    if column != 'id' and hasattr(existing, column):
                        value = None if pd.isna(row[column]) else row[column]
                        setattr(existing, column, value)
                records_updated += 1
            else:
                record_data = {}
                for column in df.columns:
                    if column != 'id':
                        value = None if pd.isna(row[column]) else row[column]
                        record_data[column] = value
                
                new_record = HistoricalEnergyPrices(**record_data)
                session.add(new_record)
                records_inserted += 1
        
        session.commit()
        session.close()
        
        logging.info(f"Historical prices: inserted {records_inserted}, updated {records_updated} records")
        return True
        
    except Exception as e:
        logging.error(f"Failed to insert historical prices to TimescaleDB Tiger Cloud: {e}")
        if session:
            session.rollback()
            session.close()
        return False

def load_historical_prices_from_db(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load historical energy prices from TimescaleDB Tiger Cloud.
    
    Args:
        start_date: Start date (YYYY-MM-DD format, optional)
        end_date: End date (YYYY-MM-DD format, optional) 
        limit: Maximum number of records (optional)
    
    Returns:
        DataFrame with historical energy price data
    """
    try:
        session = get_db_session()
        
        query = session.query(HistoricalEnergyPrices)
        
        if start_date:
            query = query.filter(HistoricalEnergyPrices.delivery_day >= start_date)
        if end_date:
            query = query.filter(HistoricalEnergyPrices.delivery_day <= end_date)
        
        query = query.order_by(HistoricalEnergyPrices.delivery_day.desc())
        
        if limit:
            query = query.limit(limit)
        
        results = query.all()
        session.close()
        
        if not results:
            logging.warning("No historical price data found in TimescaleDB Tiger Cloud")
            return pd.DataFrame()
        
        data = []
        for record in results:
            row = {}
            for column in HistoricalEnergyPrices.__table__.columns:
                row[column.name] = getattr(record, column.name)
            data.append(row)
        
        df = pd.DataFrame(data)
        df['delivery_day'] = pd.to_datetime(df['delivery_day'])
        
        logging.info(f"Loaded {len(df)} historical price records from TimescaleDB Tiger Cloud")
        return df
        
    except Exception as e:
        logging.error(f"Failed to load historical prices from TimescaleDB Tiger Cloud: {e}")
        return pd.DataFrame()

def convert_historical_to_hourly(delivery_day: str) -> pd.DataFrame:
    """
    Convert a single historical record to hourly time-series format.
    
    Args:
        delivery_day: Date in YYYY-MM-DD format
        
    Returns:
        DataFrame with hourly time-series data
    """
    try:
        session = get_db_session()
        
        record = session.query(HistoricalEnergyPrices).filter(
            HistoricalEnergyPrices.delivery_day == delivery_day
        ).first()
        
        session.close()
        
        if not record:
            return pd.DataFrame()
        
        return record.to_hourly_series()
        
    except Exception as e:
        logging.error(f"Failed to convert historical to hourly data: {e}")
        return pd.DataFrame()

def get_historical_price_count() -> int:
    """Get total number of historical price records."""
    try:
        session = get_db_session()
        count = session.query(HistoricalEnergyPrices).count()
        session.close()
        return count
        
    except Exception as e:
        logging.error(f"Failed to get historical price count: {e}")
        return 0

def get_historical_price_stats() -> dict:
    """Get comprehensive statistics about historical price data."""
    try:
        session = get_db_session()
        
        from sqlalchemy import func
        
        stats_query = session.query(
            func.count(HistoricalEnergyPrices.id).label('count'),
            func.min(HistoricalEnergyPrices.delivery_day).label('min_date'),
            func.max(HistoricalEnergyPrices.delivery_day).label('max_date')
        ).first()
        
        session.close()
        
        if stats_query.count == 0:
            return {}
        
        stats = {
            "total_records": int(stats_query.count),
            "date_range": {
                "start": stats_query.min_date,
                "end": stats_query.max_date
            }
        }
        
        return stats
        
    except Exception as e:
        logging.error(f"Failed to get historical price stats: {e}")
        return {}

def get_peak_hours_analysis(days: int = 30) -> dict:
    """
    Analyze peak hours patterns from recent historical data.
    
    Args:
        days: Number of recent days to analyze
        
    Returns:
        Dictionary with peak hours analysis
    """
    try:
        session = get_db_session()
        
        records = session.query(HistoricalEnergyPrices).order_by(
            HistoricalEnergyPrices.delivery_day.desc()
        ).limit(days).all()
        
        session.close()
        
        if not records:
            return {}
        
        hourly_averages = {}
        hourly_peaks = {}
        
        for hour in range(1, 25):
            hour_col = f'hour_{hour}' if hour != 3 else 'hour_3a'
            prices = []
            peak_count = 0
            
            for record in records:
                price = getattr(record, hour_col, None)
                if price is not None:
                    prices.append(price)
                    if hour in record.get_peak_hours():
                        peak_count += 1
            
            if prices:
                hourly_averages[f'hour_{hour}'] = sum(prices) / len(prices)
                hourly_peaks[f'hour_{hour}'] = peak_count / len(records) * 100
        
        return {
            "days_analyzed": len(records),
            "hourly_averages": hourly_averages,
            "peak_frequency_percent": hourly_peaks,
            "most_expensive_hours": sorted(hourly_averages.items(), key=lambda x: x[1], reverse=True)[:5],
            "cheapest_hours": sorted(hourly_averages.items(), key=lambda x: x[1])[:5]
        }
        
    except Exception as e:
        logging.error(f"Failed to analyze peak hours: {e}")
        return {}

logging.basicConfig(level=logging.INFO)
