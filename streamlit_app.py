import streamlit as st
import plotly.express as px
import pandas as pd
import requests, os

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Smart Energy Dashboard", layout="wide")

st.title("Energy Prices - Forecast")

tab1, tab2, tab3 = st.tabs(["Forecast", "Recommendations", "ML Models"])

with tab1:
    h = st.slider("Forecast horizon (hours)", 6, 168, 48, step=6)
    method = st.selectbox("Method", ["sarimax", "seasonal", "xgboost", "lstm", "hybrid", "auto_ml"])
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Checking server status..."):
            try:
                health_check = requests.get(f"{API_BASE}/health", timeout=10)
                if health_check.status_code != 200:
                    st.error("FastAPI server is not responding. Please make sure it's running on port 8000.")
                    st.stop()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI server. Please start the server:")
                st.code("uvicorn app.service:app --reload --host 127.0.0.1 --port 8000")
                st.stop()
            except Exception as e:
                st.error(f"Health check failed: {e}")
                st.stop()
        
        with st.spinner(f"Generating {method} forecast for {h} hours..."):
            try:
                response = requests.get(f"{API_BASE}/forecast", params={"h": h, "method": method}, timeout=30)
                if response.status_code == 200:
                    r = response.json()
                    df = pd.DataFrame(r["forecast"])
                    if not df.empty:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        fig = px.line(df, x="timestamp", y="price", title=f"Forecast ({method}, {h}h)")
                        fig.update_layout(xaxis_title="Time", yaxis_title="Price (€/MWh)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success(f"Forecast generated using {r['training_records']} records from {r['data_source']}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Min Price", f"€{df['price'].min():.2f}")
                        with col2:
                            st.metric("Max Price", f"€{df['price'].max():.2f}")
                        with col3:
                            st.metric("Avg Price", f"€{df['price'].mean():.2f}")
                    else:
                        st.warning("No forecast data available.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. The forecast is taking longer than expected. This can happen with large datasets or complex SARIMAX modeling.")
                st.info("Try using 'seasonal' method for faster results, or reduce the forecast horizon.")
            except requests.exceptions.JSONDecodeError:
                st.error("Invalid response from server. The API might be returning an error.")
            except Exception as e:
                st.error(f"Forecast error: {e}")

with tab2:
    run_hours = st.slider("Run duration (hours)", 1, 8, 2)
    k = st.slider("Number of slots", 1, 10, 3)
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner(f"Finding {k} cheapest {run_hours}-hour slots..."):
            try:
                response = requests.get(f"{API_BASE}/recommendations", params={"h": h, "k": k, "run_hours": run_hours}, timeout=20)
                if response.status_code == 200:
                    r2 = response.json()
                    recs = pd.DataFrame(r2["recommendations"])
                    if not recs.empty:
                        st.success(f"Found optimal time slots from {r2['data_source']}")
                        
                        for i, rec in recs.iterrows():
                            start_time = pd.to_datetime(rec['start']).strftime('%Y-%m-%d %H:%M')
                            end_time = pd.to_datetime(rec['end']).strftime('%H:%M')
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**Slot {i+1}:** {start_time} → {end_time}")
                                with col2:
                                    st.metric("", f"€{rec['avg_price']:.2f}/MWh")
                        
                        st.dataframe(recs, use_container_width=True)
                        st.download_button("Download CSV", recs.to_csv(index=False), "recommendations.csv")
                    else:
                        st.warning("No recommendations found for this horizon.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try reducing the forecast horizon or number of slots.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI server.")
            except requests.exceptions.JSONDecodeError:
                st.error("Invalid response from recommendations API.")
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")

with tab3:
    st.header("ML Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Train Models")
        if st.button("Train Ensemble Models", type="primary"):
            with st.spinner("Training ML models (this may take several minutes)..."):
                try:
                    response = requests.post(f"{API_BASE}/models/train", timeout=300)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Models trained successfully!")
                        st.json(result)
                    else:
                        st.error(f"Training failed: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("Training timed out. Models may still be training in the background.")
                except Exception as e:
                    st.error(f"Training error: {e}")
        
        st.subheader("Compare Methods")
        compare_horizon = st.slider("Comparison horizon (hours)", 6, 48, 24, step=6)
        if st.button("Compare All Methods"):
            with st.spinner("Comparing forecasting methods..."):
                try:
                    response = requests.post(f"{API_BASE}/models/compare", 
                                           params={"horizon": compare_horizon}, timeout=120)
                    if response.status_code == 200:
                        comparison = response.json()
                        st.success("Comparison completed!")
                        
                        # Display results
                        for method, result in comparison["comparison"].items():
                            if result["success"]:
                                st.write(f"**{method.upper()}**: {result['count']} predictions")
                            else:
                                st.write(f"**{method.upper()}**: Failed - {result['error']}")
                        
                        # Create comparison chart
                        chart_data = {}
                        for method, result in comparison["comparison"].items():
                            if result["success"] and result["forecast"]:
                                df_method = pd.DataFrame(result["forecast"])
                                df_method["timestamp"] = pd.to_datetime(df_method["timestamp"])
                                chart_data[method] = df_method.set_index("timestamp")["price"]
                        
                        if chart_data:
                            comparison_df = pd.DataFrame(chart_data)
                            st.line_chart(comparison_df)
                    else:
                        st.error(f"Comparison failed: {response.text}")
                except Exception as e:
                    st.error(f"Comparison error: {e}")
    
    with col2:
        st.subheader("Model Performance")
        if st.button("List All Models"):
            try:
                response = requests.get(f"{API_BASE}/models/list", timeout=30)
                if response.status_code == 200:
                    models = response.json()
                    if models["models"]:
                        df_models = pd.DataFrame(models["models"])
                        st.dataframe(df_models[["name", "type", "rmse", "mape", "mae", "created_at"]])
                    else:
                        st.info("No trained models found. Train some models first!")
                else:
                    st.error(f"Failed to list models: {response.text}")
            except Exception as e:
                st.error(f"Error listing models: {e}")
        
        st.subheader("Best Model")
        metric_choice = st.selectbox("Metric", ["rmse", "mape", "mae"])
        if st.button("Get Best Model"):
            try:
                response = requests.get(f"{API_BASE}/models/best", 
                                      params={"metric": metric_choice}, timeout=30)
                if response.status_code == 200:
                    best = response.json()
                    st.success(f"Best model by {metric_choice}:")
                    st.json(best["best_model"])
                else:
                    st.error(f"Failed to get best model: {response.text}")
            except Exception as e:
                st.error(f"Error getting best model: {e}")