import pandas as pd
from datetime import datetime
from typing import Optional
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from .config import get_database_url
from .models import Base, HistoricalEnergyPrices

_engine = None
_SessionLocal = None

def get_engine():
    """Get SQLAlchemy engine for TimescaleDB"""
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
        logging.info("Created TimescaleDB engine")
    return _engine

def get_session_local():
    """Get SQLAlchemy session local for Tiger Cloud"""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logging.info("Created TimescaleDB session local")
    return _SessionLocal

def get_db_session() -> Session:
    """Get database session"""
    SessionLocal = get_session_local()
    return SessionLocal()

def init_database() -> bool:
    """Initialize TimescaleDB with tables"""
    try:
        engine = get_engine()
        
        Base.metadata.create_all(bind=engine)
        
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_historical_energy_prices_delivery_day 
                ON historical_energy_prices (delivery_day);
            """))
            
            conn.commit()
        
        logging.info("TimescaleDB initialized successfully with historical_energy_prices table")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize TimescaleDB: {e}")
        return False

def check_db_connection() -> bool:
    """Check if TimescaleDB connection is working"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            
        logging.info("TimescaleDB connection successful")
        return True
        
    except Exception as e:
        logging.error(f"TimescaleDB connection failed: {e}")
        return False

def load_prices_from_db(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """Load energy prices from historical_energy_prices and convert to time-series format"""
    try:
        start_date = start_time.strftime('%Y-%m-%d') if start_time else None
        end_date = end_time.strftime('%Y-%m-%d') if end_time else None
        
        df_historical = load_historical_prices_from_db(
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        if df_historical.empty:
            return pd.DataFrame()
        
        # Convert to time-series format
        all_hourly = []
        
        for _, row in df_historical.iterrows():
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
        
        # Apply time filters if specified - ensure timezone compatibility
        if start_time:
            if start_time.tzinfo is None:
                start_time = pd.to_datetime(start_time).tz_localize('Europe/Berlin')
            else:
                start_time = pd.to_datetime(start_time).tz_convert('Europe/Berlin')
            df = df[df.index >= start_time]
            
        if end_time:
            if end_time.tzinfo is None:
                end_time = pd.to_datetime(end_time).tz_localize('Europe/Berlin')
            else:
                end_time = pd.to_datetime(end_time).tz_convert('Europe/Berlin')
            df = df[df.index <= end_time]
        
        logging.info(f"Converted {len(df_historical)} daily records to {len(df)} hourly records")
        return df
        
    except Exception as e:
        logging.error(f"Failed to load prices from TimescaleDB: {e}")
        return pd.DataFrame()

def insert_prices_to_db(df: pd.DataFrame) -> bool:
    """Insert time-series data by converting to historical format and using insert_historical_prices_to_db"""
    try:
        if df.empty:
            logging.warning("Empty DataFrame provided for insertion")
            return True
        
        # Convert time-series data to daily historical format
        df_daily = convert_timeseries_to_daily(df)
        
        if df_daily.empty:
            logging.warning("No daily data could be created from time-series")
            return True
        
        # Use the historical insertion function
        return insert_historical_prices_to_db(df_daily)
        
    except Exception as e:
        logging.error(f"Failed to insert prices into TimescaleDB: {e}")
        return False

def convert_timeseries_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Convert time-series data to daily historical format"""
    try:
        if df.empty or 'price' not in df.columns:
            return pd.DataFrame()
        
        # Group by date
        df_copy = df.copy()
        df_copy['date'] = df_copy.index.date
        df_copy['hour'] = df_copy.index.hour + 1  # Convert to 1-24 hour format
        
        daily_data = []
        
        for date, group in df_copy.groupby('date'):
            row_data = {'delivery_day': date.strftime('%Y-%m-%d')}
            
            for _, record in group.iterrows():
                hour = record['hour']
                price = record['price']
                
                if 1 <= hour <= 24:
                    if hour == 3:
                        # Handle DST - if we have multiple hour 3 entries, use 3a and 3b
                        if f'hour_3a' not in row_data:
                            row_data['hour_3a'] = price
                        else:
                            row_data['hour_3b'] = price
                    else:
                        row_data[f'hour_{hour}'] = price
            
            daily_data.append(row_data)
        
        return pd.DataFrame(daily_data)
        
    except Exception as e:
        logging.error(f"Failed to convert time-series to daily format: {e}")
        return pd.DataFrame()

def load_historical_prices_from_db(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 30
) -> pd.DataFrame:
    """Load historical energy prices from TimescaleDB"""
    try:
        session = get_db_session()
        
        query = session.query(HistoricalEnergyPrices)
        logging.info(f"Loading historical energy prices from TimescaleDB from {start_date} to {end_date} with limit {limit}")
        if start_date:
            query = query.filter(HistoricalEnergyPrices.delivery_day >= start_date)
        if end_date:
            query = query.filter(HistoricalEnergyPrices.delivery_day <= end_date)
        
        query = query.order_by(HistoricalEnergyPrices.delivery_day.desc())
        
        if limit:
            query = query.limit(limit)
        
        result = query.all()
        session.close()
        
        if not result:
            logging.info("No historical energy price data found in TimescaleDB")
            return pd.DataFrame()
        
        data = []
        for record in result:
            row_data = {
                'delivery_day': pd.to_datetime(record.delivery_day)
            }
            
            for hour in range(1, 25):
                if hour == 3:
                    if hasattr(record, 'hour_3a') and record.hour_3a is not None:
                        row_data['hour_3a'] = record.hour_3a
                    if hasattr(record, 'hour_3b') and record.hour_3b is not None:
                        row_data['hour_3b'] = record.hour_3b
                else:
                    hour_attr = f'hour_{hour}'
                    if hasattr(record, hour_attr):
                        value = getattr(record, hour_attr)
                        if value is not None:
                            row_data[hour_attr] = value
            
            data.append(row_data)
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            date_range = f"from {df['delivery_day'].min().strftime('%Y-%m-%d')} to {df['delivery_day'].max().strftime('%Y-%m-%d')}"
            logging.info(f"Loaded {len(df)} historical energy price records from TimescaleDB {date_range}")
        else:
            logging.info("No historical energy price records found in TimescaleDB")
        return df
        
    except Exception as e:
        logging.error(f"Failed to load historical prices from TimescaleDB: {e}")
        return pd.DataFrame()

def insert_historical_prices_to_db(df: pd.DataFrame) -> bool:
    """Insert historical energy price data into TimescaleDB"""
    try:
        if df.empty:
            logging.warning("Empty DataFrame provided for historical price insertion")
            return True
        
        init_database()
        
        session = get_db_session()
        
        for _, row in df.iterrows():
            delivery_day = row['delivery_day']
            
            existing = session.query(HistoricalEnergyPrices).filter(
                HistoricalEnergyPrices.delivery_day == delivery_day
            ).first()
            
            if existing:
                logging.debug(f"Skipping existing delivery_day: {delivery_day}")
                continue
            
            record_data = {'delivery_day': delivery_day}
            
            for hour in range(1, 25):
                if hour == 3:
                    if 'hour_3a' in row and pd.notna(row['hour_3a']):
                        record_data['hour_3a'] = float(row['hour_3a'])
                    if 'hour_3b' in row and pd.notna(row['hour_3b']):
                        record_data['hour_3b'] = float(row['hour_3b'])
                else:
                    hour_col = f'hour_{hour}'
                    if hour_col in row and pd.notna(row[hour_col]):
                        record_data[hour_col] = float(row[hour_col])
            
            new_record = HistoricalEnergyPrices(**record_data)
            session.add(new_record)
        
        session.commit()
        session.close()
        
        logging.info(f"Successfully inserted historical energy price data")
        return True
        
    except Exception as e:
        logging.error(f"Failed to insert historical prices: {e}")
        if 'session' in locals():
            session.rollback()
            session.close()
        return False

def get_price_count() -> int:
    """Get total count of hourly price records from historical_energy_prices"""
    try:
        session = get_db_session()
        
        # Count all hourly records from historical data
        result = session.query(HistoricalEnergyPrices).all()
        session.close()
        
        total_hours = 0
        for record in result:
            for hour in range(1, 25):
                if hour == 3:
                    if hasattr(record, 'hour_3a') and getattr(record, 'hour_3a') is not None:
                        total_hours += 1
                    if hasattr(record, 'hour_3b') and getattr(record, 'hour_3b') is not None:
                        total_hours += 1
                else:
                    hour_attr = f'hour_{hour}'
                    if hasattr(record, hour_attr) and getattr(record, hour_attr) is not None:
                        total_hours += 1
        
        return total_hours
    except Exception as e:
        logging.error(f"Error getting price count: {e}")
        return 0

def get_historical_price_count() -> int:
    """Get total count of historical energy price records"""
    try:
        session = get_db_session()
        count = session.query(HistoricalEnergyPrices).count()
        session.close()
        return count
    except Exception as e:
        logging.error(f"Error getting historical price count: {e}")
        return 0

def get_latest_timestamp() -> Optional[datetime]:
    """Get the latest timestamp from historical_energy_prices table"""
    try:
        session = get_db_session()
        result = session.query(HistoricalEnergyPrices.delivery_day).order_by(
            HistoricalEnergyPrices.delivery_day.desc()
        ).first()
        session.close()
        
        if result:
            # Convert delivery_day to datetime and add 23 hours to get end of day
            latest_date = pd.to_datetime(result[0])
            latest_timestamp = latest_date + pd.Timedelta(hours=23)
            return latest_timestamp.to_pydatetime()
        
        return None
    except Exception as e:
        logging.error(f"Error getting latest timestamp: {e}")
        return None

def get_price_stats():
    """Get basic statistics for energy prices from historical data"""
    try:
        # Use the historical price stats function as it provides the same information
        return get_historical_price_stats()
        
    except Exception as e:
        logging.error(f"Error getting price statistics: {e}")
        return None

def get_historical_price_stats():
    """Get basic statistics for historical energy prices"""
    try:
        session = get_db_session()
        
        result = session.query(HistoricalEnergyPrices).all()
        session.close()
        
        if not result:
            return None
        
        dates = [pd.to_datetime(r.delivery_day) for r in result]
        all_prices = []
        
        for record in result:
            for hour in range(1, 25):
                if hour == 3:
                    if hasattr(record, 'hour_3a') and record.hour_3a is not None:
                        all_prices.append(float(record.hour_3a))
                    if hasattr(record, 'hour_3b') and record.hour_3b is not None:
                        all_prices.append(float(record.hour_3b))
                else:
                    hour_attr = f'hour_{hour}'
                    if hasattr(record, hour_attr):
                        value = getattr(record, hour_attr)
                        if value is not None:
                            all_prices.append(float(value))
        
        if not all_prices:
            return {
                'count': len(result),
                'date_range': {
                    'start': min(dates).strftime('%Y-%m-%d') if dates else None,
                    'end': max(dates).strftime('%Y-%m-%d') if dates else None
                }
            }
        
        import numpy as np
        
        stats = {
            'count': len(result),
            'date_range': {
                'start': min(dates).strftime('%Y-%m-%d'),
                'end': max(dates).strftime('%Y-%m-%d')
            },
            'price_stats': {
                'min': float(np.min(all_prices)),
                'max': float(np.max(all_prices)),
                'mean': float(np.mean(all_prices)),
                'std': float(np.std(all_prices))
            }
        }
        
        return stats
        
    except Exception as e:
        logging.error(f"Error getting historical price statistics: {e}")
        return None

def get_peak_hours_analysis(days: int = 30):
    """Analyze peak hours patterns from recent historical data"""
    try:
        session = get_db_session()
        
        query = session.query(HistoricalEnergyPrices).order_by(
            HistoricalEnergyPrices.delivery_day.desc()
        ).limit(days)
        
        result = query.all()
        session.close()
        
        if not result:
            return None
        
        hourly_data = {}
        for hour in range(1, 25):
            hourly_data[hour] = []
        
        hourly_data[3] = []
        
        peak_count_by_hour = {}
        for hour in range(1, 25):
            peak_count_by_hour[hour] = 0
        peak_count_by_hour[3] = 0
        
        for record in result:
            daily_prices = []
            
            for hour in range(1, 25):
                if hour == 3:
                    if hasattr(record, 'hour_3a') and record.hour_3a is not None:
                        hourly_data[3].append(float(record.hour_3a))
                        daily_prices.append((3, float(record.hour_3a)))
                    if hasattr(record, 'hour_3b') and record.hour_3b is not None:
                        daily_prices.append((3, float(record.hour_3b)))
                else:
                    hour_attr = f'hour_{hour}'
                    if hasattr(record, hour_attr):
                        value = getattr(record, hour_attr)
                        if value is not None:
                            hourly_data[hour].append(float(value))
                            daily_prices.append((hour, float(value)))
            
            if daily_prices:
                max_price = max(daily_prices, key=lambda x: x[1])
                peak_hour = max_price[0]
                peak_count_by_hour[peak_hour] += 1
        
        hourly_averages = {}
        for hour in range(1, 25):
            if hourly_data[hour]:
                hourly_averages[f"hour_{hour}"] = round(
                    sum(hourly_data[hour]) / len(hourly_data[hour]), 2
                )
        
        total_days_analyzed = len(result)
        peak_frequency_percent = {}
        for hour in range(1, 25):
            if peak_count_by_hour[hour] > 0:
                peak_frequency_percent[f"hour_{hour}"] = round(
                    (peak_count_by_hour[hour] / total_days_analyzed) * 100, 1
                )
        
        sorted_by_frequency = sorted(
            peak_frequency_percent.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        sorted_by_price = sorted(
            hourly_averages.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            'days_analyzed': total_days_analyzed,
            'hourly_averages': hourly_averages,
            'peak_frequency_percent': peak_frequency_percent,
            'most_expensive_hours': [item[0] for item in sorted_by_price[:5]],
            'cheapest_hours': [item[0] for item in sorted_by_price[-5:]]
        }
        
    except Exception as e:
        logging.error(f"Error analyzing peak hours: {e}")
        return None

logging.basicConfig(level=logging.INFO)