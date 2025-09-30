import streamlit as st
import plotly.express as px
import pandas as pd
import requests, os

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Smart Energy Dashboard", layout="wide")

st.title("Energy Prices - Forecast + Smart Scheduling")

tab1, tab2 = st.tabs(["Forecast", "Recommendations"])

with tab1:
    h = st.slider("Forecast horizon (hours)", 6, 168, 48, step=6)
    method = st.radio("Method", ["sarimax", "seasonal"], horizontal=True)
    
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
                        
                        # Show data source info
                        st.success(f"Forecast generated using {r['training_records']} records from {r['data_source']}")
                        
                        # Show some forecast stats
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
                        
                        # Format the recommendations nicely
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

