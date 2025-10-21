import pandas as pd
import numpy as np
import logging
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib

# ML Libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    HAS_BOOSTING = True
except ImportError:
    HAS_BOOSTING = False
    logging.warning("Boosting libraries not available. Install: pip install xgboost lightgbm catboost")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch not available. Install: pip install torch")

try:
    from darts import TimeSeries
    from darts.models import XGBModel, LightGBMModel, NBEATSModel, TFTModel
    HAS_DARTS = True
except ImportError:
    HAS_DARTS = False
    logging.warning("Darts not available. Install: pip install darts")

from .database import get_db_session
from .models import Base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean

class ModelMetadata(Base):
    """Model metadata tracking table"""
    __tablename__ = 'model_metadata'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'xgboost', 'lstm', 'transformer', etc.
    hyperparameters = Column(Text)  # JSON string
    training_data_start = Column(DateTime)
    training_data_end = Column(DateTime)
    training_samples = Column(Integer)
    validation_rmse = Column(Float)
    validation_mape = Column(Float)
    validation_mae = Column(Float)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_path = Column(String(500))  # Path to saved model file

class FeatureEngineer:
    """Feature engineering for time series forecasting"""
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, price_col: str = 'price', lags: List[int] = None) -> pd.DataFrame:
        """Create lag features"""
        if lags is None:
            lags = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h, 2h, 3h, 6h, 12h, 1d, 2d, 1w
        
        df_features = df.copy()
        
        for lag in lags:
            df_features[f'price_lag_{lag}'] = df_features[price_col].shift(lag)
        
        return df_features
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, price_col: str = 'price', windows: List[int] = None) -> pd.DataFrame:
        """Create rolling statistics features"""
        if windows is None:
            windows = [3, 6, 12, 24, 48, 168]  # 3h, 6h, 12h, 1d, 2d, 1w
        
        df_features = df.copy()
        
        for window in windows:
            df_features[f'price_rolling_mean_{window}'] = df_features[price_col].rolling(window).mean()
            df_features[f'price_rolling_std_{window}'] = df_features[price_col].rolling(window).std()
            df_features[f'price_rolling_min_{window}'] = df_features[price_col].rolling(window).min()
            df_features[f'price_rolling_max_{window}'] = df_features[price_col].rolling(window).max()
        
        return df_features
    
    @staticmethod
    def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df_features = df.copy()
        
        # Ensure index is datetime
        if not isinstance(df_features.index, pd.DatetimeIndex):
            df_features.index = pd.to_datetime(df_features.index)
        
        # Time features
        df_features['hour'] = df_features.index.hour
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['day_of_month'] = df_features.index.day
        df_features['month'] = df_features.index.month
        df_features['quarter'] = df_features.index.quarter
        df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        return df_features
    
    @staticmethod
    def create_all_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """Create all features at once"""
        df_features = FeatureEngineer.create_time_features(df)
        df_features = FeatureEngineer.create_lag_features(df_features, price_col)
        df_features = FeatureEngineer.create_rolling_features(df_features, price_col)
        
        return df_features

class XGBoostForecaster:
    """XGBoost-based forecasting model"""
    
    def __init__(self, **xgb_params):
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            **xgb_params
        }
        self.model = None
        self.feature_columns = None
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'price') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        df_features = FeatureEngineer.create_all_features(df, target_col)
        
        # Remove rows with NaN values (due to lags and rolling windows)
        df_clean = df_features.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df_clean.columns if col != target_col]
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        self.feature_columns = feature_cols
        
        return X, y
    
    def train(self, df: pd.DataFrame, target_col: str = 'price', validation_split: float = 0.2):
        """Train XGBoost model"""
        if not HAS_BOOSTING:
            raise ImportError("XGBoost not available")
        
        X, y = self.prepare_data(df, target_col)
        
        # Train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Calculate metrics
        y_pred = self.model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        mae = np.mean(np.abs(y_val - y_pred))
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
        
        return {'rmse': rmse, 'mae': mae, 'mape': mape}
    
    def forecast(self, df: pd.DataFrame, horizon: int, target_col: str = 'price') -> pd.Series:
        """Generate forecasts"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        df_features = FeatureEngineer.create_all_features(df, target_col)
        
        forecasts = []
        current_data = df_features.copy()
        
        for step in range(horizon):
            # Get the last row for prediction
            last_row = current_data.iloc[-1:][self.feature_columns]
            
            # Make prediction
            pred = self.model.predict(last_row)[0]
            forecasts.append(pred)
            
            # Create next timestamp
            next_timestamp = current_data.index[-1] + pd.Timedelta(hours=1)
            
            # Create new row with prediction
            new_row = pd.DataFrame(index=[next_timestamp])
            new_row[target_col] = pred
            
            # Add to current data and regenerate features
            current_data = pd.concat([current_data, new_row])
            current_data = FeatureEngineer.create_all_features(current_data, target_col)
        
        # Create forecast series
        forecast_index = pd.date_range(
            start=df.index[-1] + pd.Timedelta(hours=1),
            periods=horizon,
            freq='h'
        )
        
        return pd.Series(forecasts, index=forecast_index)

class LSTMForecaster:
    """LSTM-based forecasting model"""
    
    def __init__(self, sequence_length: int = 168, hidden_size: int = 128, num_layers: int = 2):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.scaler = None
        
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, target_col: str = 'price', epochs: int = 100, validation_split: float = 0.2):
        """Train LSTM model"""
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")
        
        from sklearn.preprocessing import MinMaxScaler
        
        # Prepare data
        data = df[target_col].values.reshape(-1, 1)
        
        # Scale data
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(data_scaled.flatten())
        
        # Train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val).unsqueeze(-1)
        y_val = torch.FloatTensor(y_val)
        
        # Create model
        self.model = LSTMModel(1, self.hidden_size, self.num_layers, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs.squeeze(), y_val)
                    logging.info(f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            val_pred = self.model(X_val).squeeze().numpy()
            val_true = y_val.numpy()
            
            # Inverse transform
            val_pred_orig = self.scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
            val_true_orig = self.scaler.inverse_transform(val_true.reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(np.mean((val_true_orig - val_pred_orig) ** 2))
            mae = np.mean(np.abs(val_true_orig - val_pred_orig))
            mape = np.mean(np.abs((val_true_orig - val_pred_orig) / val_true_orig)) * 100
        
        return {'rmse': rmse, 'mae': mae, 'mape': mape}
    
    def forecast(self, df: pd.DataFrame, horizon: int, target_col: str = 'price') -> pd.Series:
        """Generate forecasts"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained")
        
        # Get last sequence
        data = df[target_col].values[-self.sequence_length:].reshape(-1, 1)
        data_scaled = self.scaler.transform(data).flatten()
        
        forecasts = []
        current_sequence = data_scaled.copy()
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                # Prepare input
                X = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1)
                
                # Make prediction
                pred = self.model(X).item()
                forecasts.append(pred)
                
                # Update sequence
                current_sequence = np.append(current_sequence[1:], pred)
        
        # Inverse transform
        forecasts_orig = self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        
        # Create forecast series
        forecast_index = pd.date_range(
            start=df.index[-1] + pd.Timedelta(hours=1),
            periods=horizon,
            freq='h'
        )
        
        return pd.Series(forecasts_orig, index=forecast_index)

class LSTMModel(nn.Module):
    """LSTM neural network model"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class ModelManager:
    """Manage multiple forecasting models"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models = {}
        
    def register_model(self, name: str, model: Any, model_type: str, hyperparams: Dict, 
                      metrics: Dict, training_data_info: Dict):
        """Register a trained model"""
        # Save model
        model_path = self.model_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        if hasattr(model, 'model') and hasattr(model.model, 'save_model'):
            # XGBoost/LightGBM
            model.model.save_model(str(model_path))
        else:
            # Generic pickle save
            joblib.dump(model, model_path)
        
        # Save metadata to database
        session = get_db_session()
        
        metadata = ModelMetadata(
            model_name=name,
            model_version=datetime.now().strftime('%Y%m%d_%H%M%S'),
            model_type=model_type,
            hyperparameters=json.dumps(hyperparams),
            training_data_start=training_data_info.get('start'),
            training_data_end=training_data_info.get('end'),
            training_samples=training_data_info.get('samples'),
            validation_rmse=metrics.get('rmse'),
            validation_mape=metrics.get('mape'),
            validation_mae=metrics.get('mae'),
            model_path=str(model_path),
            is_active=False
        )
        
        session.add(metadata)
        session.commit()
        session.close()
        
        # Store in memory
        self.models[name] = {
            'model': model,
            'metadata': metadata,
            'metrics': metrics
        }
        
        logging.info(f"Registered model {name} with RMSE: {metrics.get('rmse', 'N/A'):.4f}")
    
    def get_best_model(self, metric: str = 'rmse') -> Optional[Any]:
        """Get the best model based on validation metric"""
        session = get_db_session()
        
        if metric == 'rmse':
            best = session.query(ModelMetadata).order_by(ModelMetadata.validation_rmse.asc()).first()
        elif metric == 'mape':
            best = session.query(ModelMetadata).order_by(ModelMetadata.validation_mape.asc()).first()
        elif metric == 'mae':
            best = session.query(ModelMetadata).order_by(ModelMetadata.validation_mae.asc()).first()
        else:
            best = session.query(ModelMetadata).order_by(ModelMetadata.created_at.desc()).first()
        
        session.close()
        
        if best and Path(best.model_path).exists():
            try:
                model = joblib.load(best.model_path)
                return model
            except Exception as e:
                logging.error(f"Failed to load model {best.model_name}: {e}")
        
        return None
    
    def list_models(self) -> List[Dict]:
        """List all registered models"""
        session = get_db_session()
        models = session.query(ModelMetadata).order_by(ModelMetadata.created_at.desc()).all()
        session.close()
        
        return [{
            'name': m.model_name,
            'version': m.model_version,
            'type': m.model_type,
            'rmse': m.validation_rmse,
            'mape': m.validation_mape,
            'mae': m.validation_mae,
            'created_at': m.created_at,
            'is_active': m.is_active
        } for m in models]

def train_ensemble_models(df: pd.DataFrame, target_col: str = 'price') -> Dict[str, Any]:
    """Train multiple models and return the best one"""
    manager = ModelManager()
    results = {}
    
    training_info = {
        'start': df.index.min(),
        'end': df.index.max(),
        'samples': len(df)
    }
    
    # Train XGBoost
    if HAS_BOOSTING:
        try:
            xgb_model = XGBoostForecaster()
            xgb_metrics = xgb_model.train(df, target_col)
            manager.register_model(
                'xgboost_v1', xgb_model, 'xgboost',
                xgb_model.xgb_params, xgb_metrics, training_info
            )
            results['xgboost'] = xgb_metrics
            logging.info(f"XGBoost trained - RMSE: {xgb_metrics['rmse']:.4f}")
        except Exception as e:
            logging.error(f"XGBoost training failed: {e}")
    
    # Train LSTM
    if HAS_TORCH and len(df) > 200:  # Need sufficient data for LSTM
        try:
            lstm_model = LSTMForecaster()
            lstm_metrics = lstm_model.train(df, target_col, epochs=50)
            manager.register_model(
                'lstm_v1', lstm_model, 'lstm',
                {'sequence_length': lstm_model.sequence_length, 'hidden_size': lstm_model.hidden_size},
                lstm_metrics, training_info
            )
            results['lstm'] = lstm_metrics
            logging.info(f"LSTM trained - RMSE: {lstm_metrics['rmse']:.4f}")
        except Exception as e:
            logging.error(f"LSTM training failed: {e}")
    
    return results

logging.basicConfig(level=logging.INFO)
