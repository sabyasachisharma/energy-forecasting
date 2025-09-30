# Smart Energy Pricing System

AI-powered electricity price forecasting with intelligent device scheduling.

## Quick Start

### 1. Setup Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.env.example .env
```

### 2. Configure Database
Update your `.env` file with TimescaleDB Tiger Cloud credentials:
```bash
DB_HOST=your-host.tsdb.cloud.timescale.com
DB_PORT=37813
DB_NAME=tsdb
DB_USER=tsdbadmin
DB_PASSWORD=your-password-here
USE_DATABASE=true
```

### 3. Start Services
```bash
# Terminal 1: Start API server
uvicorn app.service:app --reload --host 127.0.0.1 --port 8000

streamlit run streamlit_app.py
```

### **Price Forecasting**
- **SARIMAX**: Advanced time series modeling with seasonal patterns
- **Seasonal Naive**: Fast fallback method for quick predictions



### **Dual Data Sources**
- **Primary**: TimescaleDB Tiger Cloud (time-series optimized)

## API Endpoints

### Core Forecasting
- `GET /forecast?h=24&method=sarimax` - Generate price predictions
- `GET /recommendations?h=24&k=3&run_hours=2` - Get optimal time slots


### Diagnostics
- `GET /health` - System status and data availability
- `GET /historical/summary` - Database statistics


## Dashboard Usage

1. **Generate Forecast**: Click "Generate Forecast" and wait for processing
2. **Get Recommendations**: Click "Get Recommendations" for optimal scheduling

