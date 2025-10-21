import numpy as np
import logging
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum
try:
    from app.forecasting import load_recent_prices, seasonal_naive_forecast, sarimax_forecast, recommend_slots, ml_forecast, hybrid_forecast
    HAS_ML_FORECASTING = True
except ImportError as e:
    from app.forecasting import load_recent_prices, seasonal_naive_forecast, sarimax_forecast, recommend_slots
    HAS_ML_FORECASTING = False
    logging.warning(f"ML forecasting not available: {e}")
from app import rag_index
from app.database import (
    check_db_connection, get_price_count, get_latest_timestamp, get_price_stats,
    get_historical_price_count, get_historical_price_stats, load_historical_prices_from_db,
    get_peak_hours_analysis
)
from app.mysql_sync import sync_missing_data_from_mysql, check_mysql_connection
from app.sftp_ops import sync_from_sftp, list_available_files
from app.config import get_config, use_database, get_index_dir, sync_from_mysql

config = get_config()

app = FastAPI(
    title="Forecasting Energy Pricing API",
    description="Energy Price Forecasting & Scheduling",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastMethod(str, Enum):
    sarimax = "sarimax"
    seasonal = "seasonal"
    xgboost = "xgboost"
    lstm = "lstm"
    hybrid = "hybrid"
    auto_ml = "auto_ml"

class AskPayload(BaseModel):
    question: str
    k: int = 5

@app.get("/health", tags=["System"])
def health():
    """System health and database status"""
    status = {"ok": True, "timestamp": datetime.now().isoformat()}
    
    if use_database():
        db_connected = check_db_connection()
        mysql_connected = check_mysql_connection() if sync_from_mysql() else False
        
        status["database"] = {
            "connected": db_connected,
            "type": "TimescaleDB" if db_connected else "unavailable",
            "config": {
                "host": config.database.host,
                "port": config.database.port,
                "name": config.database.name,
                "user": config.database.user,
                "ssl_mode": config.database.ssl_mode
            }
        }
        
        status["mysql"] = {
            "connected": mysql_connected,
            "sync_enabled": sync_from_mysql(),
            "config": {
                "host": config.mysql.host,
                "port": config.mysql.port,
                "name": config.mysql.name,
                "user": config.mysql.user
            } if sync_from_mysql() else None
        }
        
        if db_connected:
            try:
                record_count = get_price_count()
                latest_timestamp = get_latest_timestamp()
                stats = get_price_stats()
                
                status["database"]["records"] = record_count
                status["database"]["latest_data"] = latest_timestamp.isoformat() if latest_timestamp else None
                
                if stats:
                    status["database"]["stats"] = {
                        "date_range": {
                            "start": stats["date_range"]["start"].isoformat() if stats.get("date_range", {}).get("start") else None,
                            "end": stats["date_range"]["end"].isoformat() if stats.get("date_range", {}).get("end") else None
                        },
                    "price_stats": stats.get("price_stats", {})
                }

                historical_count = get_historical_price_count()
                historical_stats = get_historical_price_stats()
                
                status["database"]["historical_data"] = {
                    "records": historical_count,
                    "stats": historical_stats
                }
                
            except Exception as e:
                status["database"]["error"] = str(e)
    else:
        status["database"] = {"type": "unavailable", "error": "Database connection required"}
    
    return status

@app.get("/forecast", tags=["Forecasting"])
def get_forecast(
    h: int = Query(24, ge=1, le=168, description="Forecast horizon in hours"),
    method: ForecastMethod = Query(ForecastMethod.sarimax, description="Forecasting method")
):
    """Generate electricity price forecasts"""
    try:
        df = load_recent_prices(hours=min(2000, h * 10))
            
        if df.empty:
            raise HTTPException(status_code=500, detail="No price data available")
        
        if method == "seasonal":
            fc = seasonal_naive_forecast(df, h, season=24)
        elif method == "sarimax":
            fc = sarimax_forecast(df, h, s=24)
        elif method == "xgboost":
            if HAS_ML_FORECASTING:
                fc = ml_forecast(df, h, model_type='xgboost')
            else:
                fc = sarimax_forecast(df, h, s=24)
        elif method == "lstm":
            if HAS_ML_FORECASTING:
                fc = ml_forecast(df, h, model_type='lstm')
            else:
                fc = sarimax_forecast(df, h, s=24)
        elif method == "auto_ml":
            if HAS_ML_FORECASTING:
                fc = ml_forecast(df, h, model_type='auto')
            else:
                fc = sarimax_forecast(df, h, s=24)
        elif method == "hybrid":
            if HAS_ML_FORECASTING:
                fc = hybrid_forecast(df, h)
            else:
                fc = sarimax_forecast(df, h, s=24)
        else:
            fc = sarimax_forecast(df, h, s=24)
        out = []
        for ts, p in fc.items():
            if not np.isnan(p) and np.isfinite(p):
                out.append({"timestamp": ts.isoformat(), "price": round(float(p), 2)})
            else:
                out.append({"timestamp": ts.isoformat(), "price": 0.0})
        
        return {
            "horizon_hours": h, 
            "method": method, 
            "forecast": out,
            "data_source": "TimescaleDB",
            "training_records": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/recommendations", tags=["Scheduling"])
def get_recommendations(
    h: int = Query(24, ge=2, le=168, description="Forecast horizon in hours"),
    k: int = Query(3, ge=1, le=10, description="Number of cheapest time slots"),
    run_hours: int = Query(2, ge=1, le=8, description="Device runtime duration in hours")
):
    """Get optimal time slot recommendations for device scheduling"""
    try:
        df = load_recent_prices(hours=min(2000, h * 10))
            
        if df.empty:
            raise HTTPException(status_code=500, detail="No price data available")
        
        fc = sarimax_forecast(df, h, s=24)
        recs = recommend_slots(fc, k=k, run_hours=run_hours)
        
        return {
            "horizon_hours": h, 
            "k": k, 
            "run_hours": run_hours, 
            "recommendations": recs,
            "data_source": "TimescaleDB",
            "training_records": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendations generation failed: {str(e)}")

@app.post("/index/build", tags=["RAG"])
def build_index():
    """Build FAISS index from energy pricing data"""
    df = load_recent_prices(hours=8760)
    rag_index.build_index(df, out_dir=get_index_dir())
    return {"ok": True, "message": "Index built"}

@app.post("/rag/ask", tags=["RAG"])
def rag_ask(payload: AskPayload):
    """Ask natural language questions about energy data"""
    ctx = rag_index.retrieve(payload.question, k=payload.k, out_dir=get_index_dir())
    answer = generate_answer(payload.question, ctx)
    return {"answer": answer, "context": ctx}

@app.get("/historical/prices", tags=["Historical Data"])
def get_historical_prices(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(30, ge=1, le=365, description="Maximum number of records")
):
    """Get historical energy price data with hourly breakdowns"""
    try:
        if not use_database():
            raise HTTPException(status_code=501, detail="Historical data requires database connection")
        
        df = load_historical_prices_from_db(start_date=start_date, end_date=end_date, limit=limit)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No historical price data found")
        
        records = []
        for _, row in df.iterrows():
            record = {
                "delivery_day": row['delivery_day'].strftime('%Y-%m-%d'),
                "hourly_prices": {},
                "metadata": {}
            }
            
            for hour in range(1, 25):
                if hour == 3:
                    if 'hour_3a' in row and pd.notna(row['hour_3a']):
                        record["hourly_prices"]["hour_03a"] = round(float(row['hour_3a']), 2)
                    if 'hour_3b' in row and pd.notna(row['hour_3b']) and row['hour_3b'] != 0:
                        record["hourly_prices"]["hour_03b"] = round(float(row['hour_3b']), 2)
                        record["metadata"]["dst_transition"] = "fall_back"
                        record["metadata"]["total_hours"] = 25
                else:
                    hour_col = f'hour_{hour}'
                    if hour_col in row and pd.notna(row[hour_col]):
                        record["hourly_prices"][f"hour_{hour:02d}"] = round(float(row[hour_col]), 2)
            
            records.append(record)
        
        return {
            "records": records,
            "count": len(records),
            "date_range": {
                "start": records[-1]["delivery_day"] if records else None,
                "end": records[0]["delivery_day"] if records else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve historical prices: {str(e)}")

@app.get("/historical/peak-analysis", tags=["Historical Data"])
def get_peak_hours_analysis_endpoint(days: int = Query(30, ge=1, le=365)):
    """Analyze peak hours patterns from recent historical data"""
    try:
        if not use_database():
            raise HTTPException(status_code=501, detail="Peak analysis requires database connection")
        
        analysis = get_peak_hours_analysis(days)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="No data available for peak analysis")
        
        return {
            "analysis_period": {
                "days": analysis["days_analyzed"],
                "description": f"Analysis based on last {analysis['days_analyzed']} days of data"
            },
            "hourly_patterns": {
                "average_prices": analysis["hourly_averages"],
                "peak_frequency_percent": analysis["peak_frequency_percent"]
            },
            "insights": {
                "most_expensive_hours": analysis["most_expensive_hours"],
                "cheapest_hours": analysis["cheapest_hours"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze peak hours: {str(e)}")

@app.get("/historical/summary", tags=["Historical Data"])
def get_historical_summary():
    """Get summary statistics for historical energy price data"""
    try:
        if not use_database():
            raise HTTPException(status_code=501, detail="Historical summary requires database connection")
        
        stats = get_historical_price_stats()
        count = get_historical_price_count()
        
        if not stats or count == 0:
            raise HTTPException(status_code=404, detail="No historical price data available")
        
        return {
            "total_days": count,
            "date_range": stats.get("date_range", {}),
            "price_statistics": stats.get("price_stats", {}),
            "data_source": "TimescaleDB"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get historical summary: {str(e)}")

@app.post("/sync/mysql", tags=["Data Sync"])
def sync_data_from_mysql():
    """Sync missing data from MySQL to TimescaleDB"""
    try:
        if not sync_from_mysql():
            raise HTTPException(status_code=400, detail="MySQL sync is disabled. Set SYNC_FROM_MYSQL=true")
        
        result = sync_missing_data_from_mysql()
        
        if result["synced"]:
            return {
                "success": True,
                "message": "Data sync completed successfully",
                **result
            }
        else:
            raise HTTPException(status_code=500, detail=f"Sync failed: {result['reason']}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync operation failed: {str(e)}")

@app.post("/sync/sftp", tags=["Data Sync"])
def sync_data_from_sftp():
    """Sync data from EPEX SPOT SFTP server"""
    try:
        result = sync_from_sftp()
        
        if result["synced"]:
            return {
                "success": True,
                "message": "SFTP sync completed successfully",
                **result
            }
        else:
            raise HTTPException(status_code=500, detail=f"SFTP sync failed: {result['reason']}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SFTP sync operation failed: {str(e)}")

@app.get("/sync/status", tags=["Data Sync"])
def get_sync_status():
    """Get data synchronization status"""
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "timescaledb": {
                "connected": check_db_connection(),
                "latest_data": get_latest_timestamp().isoformat() if get_latest_timestamp() else None,
                "record_count": get_price_count()
            },
            "mysql": {
                "sync_enabled": sync_from_mysql(),
                "connected": check_mysql_connection() if sync_from_mysql() else False
            },
            "sftp": {
                "available_files": len(list_available_files()),
                "server": config.sftp.host
            }
        }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sync status: {str(e)}")

@app.get("/debug/tables", tags=["Debug"])
def debug_tables():
    """Debug database tables and data"""
    try:
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "timescaledb": {},
            "mysql": {}
        }
        
        if check_db_connection():
            try:
                ts_count = get_price_count()
                ts_latest = get_latest_timestamp()
                hist_count = get_historical_price_count()
                hist_stats = get_historical_price_stats()
                
                debug_info["timescaledb"] = {
                    "connected": True,
                    "historical_energy_prices_table": {
                        "record_count": hist_count,
                        "hourly_records_equivalent": ts_count,
                        "latest_timestamp": ts_latest.isoformat() if ts_latest else None,
                        "date_range": hist_stats.get("date_range", {}) if hist_stats else {}
                    }
                }
            except Exception as e:
                debug_info["timescaledb"]["error"] = str(e)
        else:
            debug_info["timescaledb"]["connected"] = False
        
        if sync_from_mysql():
            try:
                from app.mysql_sync import check_mysql_connection, get_mysql_delivery_dates
                
                if check_mysql_connection():
                    mysql_dates = get_mysql_delivery_dates()
                    debug_info["mysql"] = {
                        "connected": True,
                        "spot_price_daily_table": {
                            "delivery_dates_count": len(mysql_dates),
                            "latest_date": mysql_dates[0].isoformat() if mysql_dates else None,
                            "oldest_date": mysql_dates[-1].isoformat() if mysql_dates else None
                        }
                    }
                else:
                    debug_info["mysql"]["connected"] = False
            except Exception as e:
                debug_info["mysql"]["error"] = str(e)
        else:
            debug_info["mysql"]["sync_disabled"] = True
        
        return debug_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug check failed: {str(e)}")

@app.post("/convert/historical-to-timeseries", tags=["Data Management"])
def convert_historical_to_timeseries():
    """Convert historical daily data to hourly time-series format"""
    try:
        from app.database import load_prices_from_db
        
        df = load_prices_from_db(limit=1368)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No historical data found to convert")
        
        from app.database import insert_prices_to_db
        success = insert_prices_to_db(df)
        
        if success:
            return {
                "success": True,
                "message": "Historical data converted to time-series successfully",
                "records_converted": len(df),
                "date_range": {
                    "start": df.index.min().isoformat(),
                    "end": df.index.max().isoformat()
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to insert converted data")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

@app.post("/models/train", tags=["ML Models"])
def train_models():
    """Train ensemble of ML models on historical data"""
    if not HAS_ML_FORECASTING:
        raise HTTPException(status_code=501, detail="ML models not available. Install required packages: pip install xgboost torch darts")
    
    try:
        from app.ml_models import train_ensemble_models
        
        df = load_recent_prices(hours=24*90)  # 90 days of data
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No training data available")
        
        if len(df) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data for training (need at least 100 samples)")
        
        results = train_ensemble_models(df)
        
        return {
            "success": True,
            "message": "Models trained successfully",
            "training_samples": len(df),
            "date_range": {
                "start": df.index.min().isoformat(),
                "end": df.index.max().isoformat()
            },
            "results": results
        }
        
    except ImportError:
        raise HTTPException(status_code=501, detail="ML models not available. Install required packages.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.get("/models/list", tags=["ML Models"])
def list_models():
    """List all trained models with their performance metrics"""
    if not HAS_ML_FORECASTING:
        raise HTTPException(status_code=501, detail="ML models not available")
    
    try:
        from app.ml_models import ModelManager
        
        manager = ModelManager()
        models = manager.list_models()
        
        return {
            "models": models,
            "count": len(models)
        }
        
    except ImportError:
        raise HTTPException(status_code=501, detail="ML models not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/models/best", tags=["ML Models"])
def get_best_model(metric: str = "rmse"):
    """Get the best performing model"""
    if not HAS_ML_FORECASTING:
        raise HTTPException(status_code=501, detail="ML models not available")
    
    try:
        from app.ml_models import ModelManager
        
        manager = ModelManager()
        models = manager.list_models()
        
        if not models:
            raise HTTPException(status_code=404, detail="No trained models found")
        
        # Find best model by metric
        valid_models = [m for m in models if m.get(metric) is not None]
        if not valid_models:
            raise HTTPException(status_code=404, detail=f"No models with {metric} metric found")
        
        best_model = min(valid_models, key=lambda x: x[metric])
        
        return {
            "best_model": best_model,
            "metric_used": metric
        }
        
    except ImportError:
        raise HTTPException(status_code=501, detail="ML models not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get best model: {str(e)}")

@app.post("/models/compare", tags=["ML Models"])
def compare_models(horizon: int = 24):
    """Compare all available forecasting methods"""
    try:
        df = load_recent_prices(hours=horizon * 4)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for comparison")
        
        methods = ["seasonal", "sarimax", "xgboost", "lstm", "hybrid", "auto_ml"]
        results = {}
        
        for method in methods:
            try:
                if method == "seasonal":
                    fc = seasonal_naive_forecast(df, horizon, season=24)
                elif method == "sarimax":
                    fc = sarimax_forecast(df, horizon, s=24)
                elif method == "xgboost":
                    fc = ml_forecast(df, horizon, model_type='xgboost')
                elif method == "lstm":
                    fc = ml_forecast(df, horizon, model_type='lstm')
                elif method == "auto_ml":
                    fc = ml_forecast(df, horizon, model_type='auto')
                elif method == "hybrid":
                    fc = hybrid_forecast(df, horizon)
                
                forecast_data = []
                for ts, p in fc.items():
                    if not pd.isna(p) and np.isfinite(p):
                        forecast_data.append({
                            "timestamp": ts.isoformat(),
                            "price": round(float(p), 2)
                        })
                
                results[method] = {
                    "success": True,
                    "forecast": forecast_data,
                    "count": len(forecast_data)
                }
                
            except Exception as e:
                results[method] = {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "comparison": results,
            "horizon_hours": horizon,
            "training_samples": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

def generate_answer(question: str, context: list) -> str:
    try:
        import os
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            sys = "You are an assistant that answers user questions about electricity prices using only the provided context. If numerical computations are needed, base them on the context."
            ctx_text = "\\n".join(f"- {c}" for c in context)
            prompt = f"Context:\\n{ctx_text}\\n\\nQuestion: {question}\\nAnswer briefly with specifics and cite timestamps from context."
            resp = client.chat.completions.create(model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"), messages=[
                {"role":"system","content":sys},
                {"role":"user","content":prompt}
            ])
            return resp.choices[0].message.content.strip()
    except Exception:
        pass
    head = "Based on the retrieved price facts:\\n" + "\\n".join(context[:5])
    tail = f"\\n\\nYour question: {question}\\n(For a richer answer, set OPENAI_API_KEY to enable the LLM.)"
    return head + tail