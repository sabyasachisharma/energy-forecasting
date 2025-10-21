import paramiko
import pandas as pd
import logging
from datetime import datetime, date
from typing import Optional, List
from io import StringIO
from .config import get_sftp_config

def get_sftp_connection():
    """Create SFTP connection to EPEX SPOT server"""
    try:
        sftp_config = get_sftp_config()
        
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        ssh.connect(
            hostname=sftp_config.host,
            port=sftp_config.port,
            username=sftp_config.username,
            password=sftp_config.password
        )
        
        sftp = ssh.open_sftp()
        
        logging.info(f"SFTP connection established to {sftp_config.host}")
        return ssh, sftp
        
    except Exception as e:
        logging.error(f"SFTP connection failed: {e}")
        return None, None

def list_available_files(date_filter: Optional[str] = None) -> List[str]:
    """List available spot price files on SFTP server"""
    try:
        ssh, sftp = get_sftp_connection()
        
        if not sftp:
            return []
        
        files = sftp.listdir('.')
        
        spot_files = [f for f in files if 'spot' in f.lower() and f.endswith('.csv')]
        
        if date_filter:
            spot_files = [f for f in spot_files if date_filter in f]
        
        sftp.close()
        ssh.close()
        
        logging.info(f"Found {len(spot_files)} spot price files")
        return sorted(spot_files)
        
    except Exception as e:
        logging.error(f"Failed to list SFTP files: {e}")
        return []

def download_spot_price_file(filename: str) -> Optional[pd.DataFrame]:
    """Download and parse a spot price file from SFTP"""
    try:
        ssh, sftp = get_sftp_connection()
        
        if not sftp:
            return None
        
        with sftp.file(filename, 'r') as remote_file:
            file_content = remote_file.read()
        
        sftp.close()
        ssh.close()
        
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        
        logging.info(f"Downloaded and parsed {filename}, {len(df)} records")
        return df
        
    except Exception as e:
        logging.error(f"Failed to download {filename}: {e}")
        return None

def fetch_latest_spot_prices(days: int = 7) -> pd.DataFrame:
    """Fetch the latest spot price data from SFTP"""
    try:
        files = list_available_files()
        
        if not files:
            logging.warning("No spot price files found on SFTP server")
            return pd.DataFrame()
        
        recent_files = files[-days:] if len(files) >= days else files
        
        all_data = []
        
        for filename in recent_files:
            df = download_spot_price_file(filename)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if not all_data:
            logging.warning("No data could be downloaded from SFTP")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        if 'delivery_date' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['delivery_date'])
        
        logging.info(f"Fetched {len(combined_df)} records from {len(recent_files)} files")
        return combined_df
        
    except Exception as e:
        logging.error(f"Failed to fetch spot prices from SFTP: {e}")
        return pd.DataFrame()

def sync_from_sftp() -> dict:
    """Sync spot price data from SFTP server"""
    try:
        sftp_data = fetch_latest_spot_prices(days=30)
        
        if sftp_data.empty:
            return {"synced": False, "reason": "No data available from SFTP"}
        
        processed_data = process_epex_data(sftp_data)
        
        if processed_data.empty:
            return {"synced": False, "reason": "Failed to process SFTP data"}
        
        from .database import insert_prices_to_db
        
        success = insert_prices_to_db(processed_data)
        
        if success:
            return {
                "synced": True,
                "records_synced": len(processed_data),
                "source": "EPEX SPOT SFTP"
            }
        else:
            return {"synced": False, "reason": "Failed to insert SFTP data into TimescaleDB"}
        
    except Exception as e:
        logging.error(f"SFTP sync failed: {e}")
        return {"synced": False, "reason": f"SFTP sync error: {str(e)}"}

def process_epex_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process EPEX SPOT data format to TimescaleDB format"""
    try:
        if 'Date' in df.columns and 'Hour' in df.columns and 'Price' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')
            result_df = df[['timestamp', 'Price']].rename(columns={'Price': 'price'})
            
        elif 'delivery_date' in df.columns:
            all_hourly = []
            
            for _, row in df.iterrows():
                base_date = pd.to_datetime(row['delivery_date'])
                
                for hour in range(1, 25):
                    hour_col = f'hour_{hour}'
                    if hour_col in row and pd.notna(row[hour_col]):
                        timestamp = base_date + pd.Timedelta(hours=hour-1)
                        all_hourly.append({
                            'timestamp': timestamp,
                            'price': float(row[hour_col])
                        })
            
            result_df = pd.DataFrame(all_hourly)
            
        else:
            logging.error("Unknown EPEX data format")
            return pd.DataFrame()
        
        if not result_df.empty:
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            if result_df['timestamp'].dt.tz is None:
                result_df['timestamp'] = result_df['timestamp'].dt.tz_localize('Europe/Berlin')
            
            result_df = result_df.sort_values('timestamp').set_index('timestamp')
            result_df['price'] = result_df['price'].astype(float)
        
        logging.info(f"Processed EPEX data: {len(result_df)} hourly records")
        return result_df
        
    except Exception as e:
        logging.error(f"Failed to process EPEX data: {e}")
        return pd.DataFrame()

logging.basicConfig(level=logging.INFO)