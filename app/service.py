
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum
from app.forecasting import load_prices, load_recent_prices, seasonal_naive_forecast, sarimax_forecast, recommend_slots
from app import rag_index
from app.database import (
    check_db_connection, get_price_count, get_latest_timestamp, get_price_stats,
    get_historical_price_count, get_historical_price_stats, load_historical_prices_from_db,
    get_peak_hours_analysis
)
from app.config import get_config, use_database, get_index_dir

config = get_config()

app = FastAPI(
    title="Forecasting Energy Pricing API",
    description="""
    Energy Price Forecasting & Scheduling
    
    Advanced electricity price forecasting with intelligent device scheduling recommendations.
    Built for energy optimization, cost savings, and smart grid integration.
    
    Core Features:
    - SARIMAX & Seasonal forecasting models
    - Optimal time slot recommendations
    - TimescaleDB Tiger Cloud integration
    - Historical data analysis
    
    - Smart home energy management
    - EV charging optimization  
    - Energy trading analysis
    """,
    version="1.0.0",
    contact={
        "name": "Energy Pricing API",
        "url": "https://github.com/sabyasachisharma/rag_energy_pricing",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastMethod(str, Enum):
    """Available forecasting methods"""
    sarimax = "sarimax"
    seasonal = "seasonal"

class AskPayload(BaseModel):
    question: str
    k: int = 5

@app.get("/health",
         summary="System Health Check",
         description="Get comprehensive system status including database connectivity and data availability",
         tags=["System"])
def health():
    """
    **System Health & Status**
    
    Provides detailed information about system components and data availability.
    
    **Checks:**
    - API server status
    - Database connectivity (TimescaleDB Tiger Cloud)
    - Available data records and date ranges
    - Price statistics and data quality
    
    **Use This To:**
    - Verify system is operational before making forecasts
    - Check data freshness and coverage
    - Troubleshoot connection issues
    - Monitor database performance
    
    **Returns:**
    - Connection status for all components
    - Data record counts and date ranges
    - Database configuration details
    - Error messages if issues detected
    """
    status = {"ok": True, "timestamp": datetime.now().isoformat()}
    
    if use_database():
        db_connected = check_db_connection()
        status["database"] = {
            "connected": db_connected,
            "type": "TimescaleDB Tiger Cloud" if db_connected else "unavailable",
            "config": {
                "host": config.database.host,
                "port": config.database.port,
                "name": config.database.name,
                "user": config.database.user,
                "ssl_mode": config.database.ssl_mode
            }
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

@app.get("/forecast", 
         summary="Generate Energy Price Forecast",
         description="Generate electricity price forecasts using advanced time series models",
         tags=["Forecasting"])
def get_forecast(
    h: int = Query(
        24, 
        ge=1, 
        le=168, 
        description="Forecast horizon in hours (1-168). Recommended: 24h for daily planning, 48h for weekend planning, 168h for weekly planning",
        example=48
    ),
    method: ForecastMethod = Query(
        ForecastMethod.sarimax, 
        description="Forecasting method: 'sarimax' for accurate ARIMA-based modeling (slower), 'seasonal' for fast naive seasonal patterns"
    )
):
    """
    **Generate Electricity Price Forecasts**
    
    Predicts future energy prices using historical data and advanced time series modeling.
    
    **Methods:**
    - **SARIMAX**: Advanced seasonal ARIMA model with external regressors (recommended for accuracy)
    - **Seasonal**: Fast naive method repeating seasonal patterns (recommended for speed)
    
    **Use Cases:**
    - Plan energy-intensive tasks during low-price periods
    - Schedule device operations (washing machines, EV charging, etc.)
    - Energy trading and market analysis
    
    **Data Source:**
    - TimescaleDB Tiger Cloud (real-time data)
    
    **Returns:**
    - Hourly price predictions with timestamps
    - Data source information and training statistics
    """
    try:
        df = load_recent_prices(hours=min(2000, h * 10))
            
        if df.empty:
            raise HTTPException(status_code=500, detail="No price data available")
        
        if method == "seasonal":
            fc = seasonal_naive_forecast(df, h, season=24)
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
            "data_source": "TimescaleDB Tiger Cloud",
            "training_records": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/recommendations",
         summary="Get Optimal Time Slot Recommendations",
         description="Find the cheapest time windows to run energy-intensive devices",
         tags=["Scheduling"])
def get_recommendations(
    h: int = Query(
        24, 
        ge=2, 
        le=168, 
        description="Forecast horizon in hours to search within (2-168). Larger values find better deals but take more processing time",
        example=48
    ),
    k: int = Query(
        3, 
        ge=1, 
        le=10, 
        description="Number of cheapest time slots to return (1-10). Get multiple options for flexibility",
        example=3
    ),
    run_hours: int = Query(
        2, 
        ge=1, 
        le=8, 
        description="Device runtime duration in hours (1-8). How long your device/process will run continuously",
        example=2
    )
):
    """
    **Smart Energy Scheduling Recommendations**
    
    Analyzes forecasted energy prices to find optimal time windows for running energy-intensive devices.
    
    **Perfect For:**
    - Electric vehicle charging
    - Washing machines and dryers  
    - Industrial processes
    - Data center batch jobs
    - Heat pumps and water heaters
    
    **How It Works:**
    1. Generates price forecast for specified horizon
    2. Finds non-overlapping time windows of specified duration
    3. Returns the cheapest slots with average prices
    
    **Example Use Case:**
    - Device needs 2 hours to run
    - Search next 48 hours
    - Get 3 cheapest options
    - Save money by timing your energy usage
    
    **Returns:**
    - Start/end timestamps for each recommended slot
    - Average price for each time window
    - Total potential savings information
    """
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
            "data_source": "TimescaleDB Tiger Cloud",
            "training_records": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendations generation failed: {str(e)}")

@app.post("/index/build")
def build_index():
    df = load_recent_prices(hours=8760)
    rag_index.build_index(df, out_dir=get_index_dir())
    return {"ok": True, "message": "Index built"}

@app.post("/rag/ask")
def rag_ask(payload: AskPayload):
    ctx = rag_index.retrieve(payload.question, k=payload.k, out_dir=get_index_dir())
    answer = generate_answer(payload.question, ctx)
    return {"answer": answer, "context": ctx}

# Historical Energy Price Endpoints

@app.get("/historical/prices",
         summary="Get Historical Energy Price Data",
         description="Retrieve detailed historical energy prices with hourly breakdowns",
         tags=["Historical Data"])
def get_historical_prices(
    start_date: Optional[str] = Query(
        None, 
        description="Start date in YYYY-MM-DD format. If not specified, returns most recent data",
        example="2025-01-01",
        regex=r"^\d{4}-\d{2}-\d{2}$"
    ),
    end_date: Optional[str] = Query(
        None, 
        description="End date in YYYY-MM-DD format. If not specified, uses current date",
        example="2025-01-31",
        regex=r"^\d{4}-\d{2}-\d{2}$"
    ),
    limit: int = Query(
        30, 
        ge=1, 
        le=365, 
        description="Maximum number of daily records to return (1-365). Used when date range is not specified",
        example=30
    )
):
    """
    **Historical Energy Price Data**
    
    Provides detailed historical energy price data with hourly breakdowns for analysis and backtesting.
    
    **Features:**
    - Complete 24-hour daily breakdowns
    - DST (Daylight Saving Time) handling with hour_3a/3b
    - Flexible date range filtering
    - Prices in EUR per MWh
    
    **Data Structure:**
    - Each record represents one day
    - Hourly prices from hour_01 to hour_24
    - Special DST handling for time transitions
    - Metadata for unusual days (25-hour days)
    
    **Use Cases:**
    - Historical analysis and reporting
    - Backtesting forecasting models
    - Price trend analysis
    - Energy trading strategy development
    """
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

@app.get("/historical/peak-analysis")
def get_peak_hours_analysis_endpoint(days: int = Query(30, ge=1, le=365)):
    """Analyze peak hours patterns from recent historical data."""
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

@app.get("/historical/summary",
         summary="Historical Data Summary",
         description="Get comprehensive statistics and overview of available historical data",
         tags=["Historical Data"])
def get_historical_summary():
    """
    **Historical Data Summary & Statistics**
    
    Provides comprehensive overview of available historical energy price data.
    
    **Includes:**
    - Total number of available daily records
    - Complete date range coverage
    - Price statistics (min, max, average, volatility)
    - Data source information
    
    **Perfect For:**
    - Quick data availability check
    - Understanding dataset scope
    - Planning analysis timeframes
    - Validating data coverage for forecasting
    """
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
            "data_source": "TimescaleDB Tiger Cloud"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get historical summary: {str(e)}")

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
    head = "Based on the retrieved price facts:\n" + "\n".join(context[:5])
    tail = f"\n\nYour question: {question}\n(For a richer answer, set OPENAI_API_KEY to enable the LLM.)"
    return head + tail
