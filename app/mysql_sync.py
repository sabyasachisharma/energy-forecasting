import pandas as pd
import logging
from datetime import datetime, date
from typing import Optional, List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from .config import get_mysql_config, sync_from_mysql
from .database import insert_prices_to_db, get_latest_timestamp

_mysql_engine = None
_mysql_session_local = None

def get_mysql_engine():
    """Get SQLAlchemy engine for MySQL"""
    global _mysql_engine
    if _mysql_engine is None:
        mysql_config = get_mysql_config()
        _mysql_engine = create_engine(
            mysql_config.connection_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=False
        )
        logging.info("Created MySQL engine")
    return _mysql_engine

def get_mysql_session_local():
    """Get SQLAlchemy session local for MySQL"""
    global _mysql_session_local
    if _mysql_session_local is None:
        engine = get_mysql_engine()
        _mysql_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logging.info("Created MySQL session local")
    return _mysql_session_local

def get_mysql_session() -> Session:
    """Get MySQL database session"""
    SessionLocal = get_mysql_session_local()
    return SessionLocal()

def check_mysql_connection() -> bool:
    """Check if MySQL connection is working"""
    try:
        engine = get_mysql_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        
        logging.info("MySQL connection successful")
        return True
        
    except Exception as e:
        logging.error(f"MySQL connection failed: {e}")
        return False

def get_mysql_delivery_dates() -> List[date]:
    """Get all delivery dates from MySQL database"""
    try:
        session = get_mysql_session()
        
        query = text("""
            SELECT DISTINCT delivery_day 
            FROM spot_price_daily 
            ORDER BY delivery_day DESC
        """)
        
        result = session.execute(query)
        dates = [row[0] for row in result.fetchall()]
        session.close()
        
        logging.info(f"Found {len(dates)} delivery dates in MySQL")
        return dates
        
    except Exception as e:
        logging.error(f"Failed to get delivery dates from MySQL: {e}")
        return []

def load_missing_dates_from_mysql(missing_dates: List[date]) -> pd.DataFrame:
    """Load energy price data for specific dates from MySQL"""
    try:
        if not missing_dates:
            return pd.DataFrame()
        
        session = get_mysql_session()
        
        date_strings = [f"'{date_obj.strftime('%Y-%m-%d')}'" for date_obj in missing_dates]
        dates_clause = ','.join(date_strings)
        
        describe_query = text("DESCRIBE spot_price_daily")
        describe_result = session.execute(describe_query)
        columns_info = describe_result.fetchall()
        available_columns = [row[0] for row in columns_info]
        
        logging.info(f"Available columns in spot_price_daily: {available_columns}")
        
        select_columns = ["delivery_day"]
        
        for hour in range(1, 25):
            if hour == 3:
                if "hour_3a" in available_columns:
                    select_columns.append("hour_3a")
                if "hour_3b" in available_columns:
                    select_columns.append("hour_3b")
                elif f"hour_{hour}" in available_columns:
                    select_columns.append(f"hour_{hour}")
            else:
                hour_col = f"hour_{hour}"
                if hour_col in available_columns:
                    select_columns.append(hour_col)
        
        columns_clause = ", ".join(select_columns)
        query = text(f"""
            SELECT {columns_clause}
            FROM spot_price_daily 
            WHERE delivery_day IN ({dates_clause})
            ORDER BY delivery_day
        """)
        
        logging.info(f"Executing query with {len(select_columns)} columns: {select_columns}")
        
        result = session.execute(query)
        rows = result.fetchall()
        session.close()
        
        if not rows:
            logging.warning(f"No data found in MySQL for dates: {missing_dates}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=select_columns)
        
        logging.info(f"Loaded {len(df)} records from MySQL for {len(missing_dates)} dates")
        logging.info(f"DataFrame shape: {df.shape}, Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        logging.error(f"Failed to load data from MySQL: {e}")
        return pd.DataFrame()

def convert_mysql_to_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MySQL hourly data format to TimescaleDB time-series format"""
    try:
        if df.empty:
            return pd.DataFrame()
        
        all_hourly = []
        
        for _, row in df.iterrows():
            base_date = pd.to_datetime(row['delivery_day'])
            
            for hour in range(1, 25):
                hour_col = f'hour_{hour}'
                if hour_col in row and pd.notna(row[hour_col]):
                    timestamp = base_date + pd.Timedelta(hours=hour-1)
                    all_hourly.append({
                        'timestamp': timestamp, 
                        'price': float(row[hour_col])
                    })
        
        if not all_hourly:
            return pd.DataFrame()
        
        ts_df = pd.DataFrame(all_hourly)
        ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'])
        if ts_df['timestamp'].dt.tz is None:
            ts_df['timestamp'] = ts_df['timestamp'].dt.tz_localize('Europe/Berlin')
        
        ts_df = ts_df.sort_values('timestamp').set_index('timestamp')
        
        logging.info(f"Converted {len(df)} daily records to {len(ts_df)} hourly records")
        return ts_df
        
    except Exception as e:
        logging.error(f"Failed to convert MySQL data to time-series: {e}")
        return pd.DataFrame()

def sync_missing_data_from_mysql() -> dict:
    """Sync missing data from MySQL to TimescaleDB historical_energy_prices table"""
    try:
        if not sync_from_mysql():
            return {"synced": False, "reason": "MySQL sync disabled"}
        
        if not check_mysql_connection():
            return {"synced": False, "reason": "MySQL connection failed"}
        
        from .database import get_historical_price_stats
        
        latest_delivery_day = None
        hist_stats = get_historical_price_stats()
        if hist_stats and hist_stats.get("date_range", {}).get("end"):
            latest_delivery_day = hist_stats["date_range"]["end"]
            logging.info(f"Latest delivery_day in TimescaleDB: {latest_delivery_day}")
        else:
            logging.info("No existing data in TimescaleDB historical_energy_prices table")
        
        mysql_data = fetch_mysql_data_since_date(latest_delivery_day)
        
        if mysql_data.empty:
            return {
                "synced": True, 
                "reason": "No new data found in MySQL", 
                "records_synced": 0
            }
        
        from .database import insert_historical_prices_to_db
        
        success = insert_historical_prices_to_db(mysql_data)
        
        if success:
            return {
                "synced": True, 
                "records_synced": len(mysql_data),
                "date_range": {
                    "start": mysql_data['delivery_day'].min(),
                    "end": mysql_data['delivery_day'].max()
                },
                "source_table": "spot_price_daily",
                "target_table": "historical_energy_prices"
            }
        else:
            return {"synced": False, "reason": "Failed to insert data into TimescaleDB"}
        
    except Exception as e:
        logging.error(f"Data sync failed: {e}")
        return {"synced": False, "reason": f"Sync error: {str(e)}"}

def fetch_mysql_data_since_date(since_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch data from MySQL spot_price_daily table since a specific date"""
    try:
        session = get_mysql_session()
        
        describe_query = text("DESCRIBE spot_price_daily")
        describe_result = session.execute(describe_query)
        columns_info = describe_result.fetchall()
        available_columns = [row[0] for row in columns_info]
        
        logging.info(f"Available columns in spot_price_daily: {available_columns}")
        
        select_columns = ["delivery_day"]
        
        for hour in range(1, 25):
            if hour == 3:
                if "hour_3a" in available_columns:
                    select_columns.append("hour_3a")
                if "hour_3b" in available_columns:
                    select_columns.append("hour_3b")
                elif f"hour_{hour}" in available_columns:
                    select_columns.append(f"hour_{hour}")
            else:
                hour_col = f"hour_{hour}"
                if hour_col in available_columns:
                    select_columns.append(hour_col)
        
        columns_clause = ", ".join(select_columns)
        
        if since_date:
            query = text(f"""
                SELECT {columns_clause}
                FROM spot_price_daily 
                WHERE delivery_day >= '{since_date}'
                ORDER BY delivery_day
            """)
            logging.info(f"Fetching MySQL data since {since_date}")
        else:
            query = text(f"""
                SELECT {columns_clause}
                FROM spot_price_daily 
                ORDER BY delivery_day
            """)
            logging.info("Fetching all MySQL data")
        
        result = session.execute(query)
        rows = result.fetchall()
        session.close()
        
        if not rows:
            logging.info("No new data found in MySQL")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=select_columns)
        
        df['delivery_day'] = pd.to_datetime(df['delivery_day']).dt.strftime('%Y-%m-%d')
        
        logging.info(f"Fetched {len(df)} records from MySQL spot_price_daily")
        logging.info(f"Date range: {df['delivery_day'].min()} to {df['delivery_day'].max()}")
        
        return df
        
    except Exception as e:
        logging.error(f"Failed to fetch data from MySQL: {e}")
        return pd.DataFrame()

logging.basicConfig(level=logging.INFO)