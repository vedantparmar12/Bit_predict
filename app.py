import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from prophet.plot import plot_components_plotly

def prepare_data_for_prophet(df):
    df_prophet = df.rename(columns={'date': 'ds', 'sales': 'y'})
    return df_prophet

def train_prophet_model(train_data):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train_data)
    return model

def evaluate_model(true_values, predictions):
    if len(true_values) == 0 or len(predictions) == 0 or len(true_values) != len(predictions):
        return np.nan, np.nan
    
    true_values_np = np.asarray(true_values)
    predictions_np = np.asarray(predictions)
    
    if np.any(np.isinf(true_values_np)) or np.any(np.isinf(predictions_np)):
        return np.nan, np.nan
        
    if np.any(np.isnan(true_values_np)) or np.any(np.isnan(predictions_np)):
         finite_mask = np.isfinite(true_values_np) & np.isfinite(predictions_np)
         if not np.any(finite_mask):
             return np.nan, np.nan
         true_values_np = true_values_np[finite_mask]
         predictions_np = predictions_np[finite_mask]
         if len(true_values_np) == 0:
             return np.nan, np.nan

    mae = mean_absolute_error(true_values_np, predictions_np)
    mape = mean_absolute_percentage_error(true_values_np, predictions_np)
    return mae, mape

def run_analysis(df_for_forecast, split_date, num_days_to_forecast):
    st.header("📈 Data Split & Forecast Period")
    train_set = df_for_forecast[df_for_forecast['date'] <= split_date].copy()
    
    forecast_start_date = split_date + pd.Timedelta(days=1)
    forecast_end_date = split_date + pd.Timedelta(days=num_days_to_forecast)
    forecast_period_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')

    test_set = df_for_forecast[
        (df_for_forecast['date'] >= forecast_start_date) &
        (df_for_forecast['date'] <= forecast_end_date)
    ].copy()

    if train_set.empty:
        st.error("The training set is empty. Adjust the split date or ensure data availability.")
        return
    
    st.write(f"Training data: {len(train_set)} records (up to {split_date.strftime('%Y-%m-%d')})")
    st.write(f"Forecast period: {len(forecast_period_dates)} days (from {forecast_start_date.strftime('%Y-%m-%d')} to {forecast_end_date.strftime('%Y-%m-%d')})")
    if not test_set.empty:
        st.write(f"Actuals available for {len(test_set)} days in the forecast period for evaluation.")
    else:
        st.write("No actuals available in the database for the forecast period for evaluation.")

    prophet_mae, prophet_mape = np.nan, np.nan
    hw_mae, hw_mape = np.nan, np.nan
    prophet_forecast_full_df = pd.DataFrame()
    prophet_predictions_renamed = pd.DataFrame()
    hw_predictions_df = pd.DataFrame()
    
    st.markdown("---")
    st.header("🔵 Prophet Model")
    try:
        with st.spinner("Training Prophet model..."):
            train_prophet_ready = prepare_data_for_prophet(train_set)
            
            if train_prophet_ready.shape[0] < 2:
                 st.warning("Prophet requires at least 2 data points for training. Skipping Prophet model.")
                 raise ValueError("Not enough data for Prophet training.")

            prophet_model = train_prophet_model(train_prophet_ready)

        with st.spinner("Making Prophet predictions..."):
            future_df_prophet = pd.DataFrame({'ds': forecast_period_dates})
            prophet_forecast_full_df = prophet_model.predict(future_df_prophet)
            prophet_predictions_renamed = prophet_forecast_full_df[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'prediction'})

        if not test_set.empty:
            eval_df_prophet = pd.merge(prophet_predictions_renamed, test_set, on='date', how='inner')
            if not eval_df_prophet.empty:
                prophet_mae, prophet_mape = evaluate_model(eval_df_prophet['sales'], eval_df_prophet['prediction'])

        col1, col2 = st.columns(2)
        col1.metric("Prophet MAE", f"{prophet_mae:.2f}" if not np.isnan(prophet_mae) else "N/A")
        col2.metric("Prophet MAPE", f"{prophet_mape:.2%}" if not np.isnan(prophet_mape) else "N/A")

        if not prophet_forecast_full_df.empty:
            st.subheader("Prophet Forecast Components")
            fig_components_plotly = plot_components_plotly(prophet_model, prophet_forecast_full_df)
            st.plotly_chart(fig_components_plotly, use_container_width=True)
    except Exception as e:
        st.error(f"Error in Prophet modeling: {e}")
        prophet_mae, prophet_mape = np.nan, np.nan
        prophet_forecast_full_df = pd.DataFrame()
        prophet_predictions_renamed = pd.DataFrame()

    st.markdown("---")
    st.header("🔴 Holt-Winters Model")
    try:
        with st.spinner("Training Holt-Winters model..."):
            hw_train_series = train_set.set_index('date')['sales']
            seasonal_periods_hw = 365
            
            if len(hw_train_series) < seasonal_periods_hw:
                st.warning(f"Holt-Winters training data ({len(hw_train_series)} points) is less than common seasonal period ({seasonal_periods_hw}). Results may be less reliable.")
            if len(hw_train_series) < 2:
                 st.warning("Holt-Winters requires at least 2 data points for training. Skipping model.")
                 raise ValueError("Not enough data for Holt-Winters training.")

            hw_model = ExponentialSmoothing(
                hw_train_series,
                trend="additive",
                seasonal="additive",
                seasonal_periods=seasonal_periods_hw,
                initialization_method='estimated'
            )
            hw_fit = hw_model.fit(optimized=True)

        with st.spinner("Making Holt-Winters predictions..."):
            hw_predictions_series = hw_fit.predict(start=forecast_start_date, end=forecast_end_date)
            hw_predictions_df = hw_predictions_series.reset_index()
            hw_predictions_df.columns = ['date', 'prediction']
            hw_predictions_df['date'] = pd.to_datetime(hw_predictions_df['date'])


        if not test_set.empty:
            eval_df_hw = pd.merge(hw_predictions_df, test_set, on='date', how='inner')
            if not eval_df_hw.empty:
                hw_mae, hw_mape = evaluate_model(eval_df_hw['sales'], eval_df_hw['prediction'])
        
        col1, col2 = st.columns(2)
        col1.metric("Holt-Winters MAE", f"{hw_mae:.2f}" if not np.isnan(hw_mae) else "N/A")
        col2.metric("Holt-Winters MAPE", f"{hw_mape:.2%}" if not np.isnan(hw_mape) else "N/A")
    except Exception as e:
        st.error(f"Error in Holt-Winters modeling: {e}")
        hw_mae, hw_mape = np.nan, np.nan
        hw_predictions_df = pd.DataFrame()

    st.markdown("---")
    st.header("📊 Model Comparison & Plots")
    if not np.isnan(prophet_mae) and not np.isnan(hw_mae) and prophet_mae is not None and hw_mae is not None:
        st.subheader("Performance Metrics Summary (Prophet vs Holt-Winters)")
        improvement_metrics = {}
        if hw_mae != 0 and not np.isnan(hw_mae):
            mae_improvement = (hw_mae - prophet_mae) / abs(hw_mae) * 100
            improvement_metrics["MAE Improvement"] = f"{mae_improvement:.2f}%"
        else:
            improvement_metrics["MAE Improvement"] = "N/A"

        if hw_mape != 0 and not np.isnan(hw_mape):
            mape_improvement = (hw_mape - prophet_mape) / abs(hw_mape) * 100
            improvement_metrics["MAPE Improvement"] = f"{mape_improvement:.2f}%"
        else:
            improvement_metrics["MAPE Improvement"] = "N/A"
        
        st.table(pd.DataFrame.from_dict(improvement_metrics, orient='index', columns=['Value']))

    st.subheader("Sales Forecast Comparison (Actual vs. Predictions)")
    fig_compare = go.Figure()
    
    if not test_set.empty:
        fig_compare.add_trace(go.Scatter(x=test_set['date'], y=test_set['sales'], mode='lines+markers', name='Actual Sales', line=dict(color='black')))

    if not prophet_predictions_renamed.empty and not prophet_forecast_full_df.empty:
        fig_compare.add_trace(go.Scatter(x=prophet_predictions_renamed['date'], y=prophet_predictions_renamed['prediction'], mode='lines', name='Prophet Forecast', line=dict(color='blue')))
        fig_compare.add_trace(go.Scatter(
            x=prophet_forecast_full_df['ds'], y=prophet_forecast_full_df['yhat_upper'],
            mode='lines', line=dict(width=0), name='Prophet Upper CI', showlegend=False, hoverinfo='skip'
        ))
        fig_compare.add_trace(go.Scatter(
            x=prophet_forecast_full_df['ds'], y=prophet_forecast_full_df['yhat_lower'],
            mode='lines', line=dict(width=0), name='Prophet Lower CI', fillcolor='rgba(0,0,255,0.1)',
            fill='tonexty', showlegend=False, hoverinfo='skip'
        ))

    if not hw_predictions_df.empty:
        fig_compare.add_trace(go.Scatter(x=hw_predictions_df['date'], y=hw_predictions_df['prediction'], mode='lines', name='Holt-Winters Forecast', line=dict(color='red')))

    fig_compare.update_layout(
        xaxis_title='Date',
        yaxis_title='Sales',
        legend_title="Legend",
        hovermode="x unified"
    )
    st.plotly_chart(fig_compare, use_container_width=True)

def create_rich_sample_data():
    st.toast("Generating rich sample data...", icon="⏳")
    stores = [f"Store {i+1}" for i in range(2)] 
    items_per_store = {
        "Store 1": [f"S1 Item {j+1}" for j in range(2)],
        "Store 2": [f"S2 Item {j+1}" for j in range(2)]
    }
    start_date = '2019-01-01'
    end_date = '2022-12-31'
    date_rng = pd.date_range(start=start_date, end=end_date, freq='D')

    if date_rng.empty:
        return pd.DataFrame(columns=['date', 'store', 'item', 'sales'])

    all_data = []
    for store_idx, store_name in enumerate(stores):
        for item_idx, item_name in enumerate(items_per_store[store_name]):
            base_sales = 70 + (store_idx * 15) + (item_idx * 8)
            trend_factor = 0.03 * (store_idx + 1)
            
            current_sales_values = base_sales + np.arange(len(date_rng)) * trend_factor
            current_sales_values += (12 + store_idx*3) * np.sin(np.arange(len(date_rng)) / (30.44/2) * np.pi)
            current_sales_values += (18 + item_idx*4) * np.sin(np.arange(len(date_rng)) / (365.25/2) * np.pi)
            current_sales_values += np.random.normal(0, 6 + store_idx, size=len(date_rng))
            
            sales_values_np = np.array(current_sales_values).clip(min=1)

            store_item_df = pd.DataFrame({
                'date': date_rng,
                'store': store_name,
                'item': item_name,
                'sales': sales_values_np
            })
            all_data.append(store_item_df)
    
    if not all_data:
        return pd.DataFrame(columns=['date', 'store', 'item', 'sales'])
        
    final_df = pd.concat(all_data, ignore_index=True)
    st.toast("Sample data generated!", icon="✅")
    return final_df

def main():
    st.set_page_config(layout="wide", page_title="Sales Forecasting Tool", page_icon="📈")
    
    st.markdown("<h1 style='text-align: center; color: #007bff;'>📈 Sales Forecasting Dashboard 📊</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p align="center">
    Upload daily sales data or use sample data to compare Prophet and Holt-Winters forecasting models.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.header("⚙️ Configuration")

    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV sales data",
        type=["csv"],
        help="CSV needs 'date' & 'sales' columns."
    )
    
    load_sample_button = st.sidebar.button("Load Sample Data", help="Generates sample data with stores and items.")

    if 'active_data' not in st.session_state:
        st.session_state.active_data = None
        st.session_state.data_source_name = None
        st.session_state.is_sample_data = False
        st.session_state.selected_store = None
        st.session_state.selected_item = None


    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.session_state.data_source_name = uploaded_file.name
            
            if 'date' not in df_input.columns or 'sales' not in df_input.columns:
                st.sidebar.error("Error: CSV must contain 'date' and 'sales' columns.")
                st.session_state.active_data = None
                return

            df_input['date'] = pd.to_datetime(df_input['date'])
            df_input['sales'] = pd.to_numeric(df_input['sales'], errors='coerce')
            df_input.dropna(subset=['date', 'sales'], inplace=True)

            if df_input.duplicated(subset=['date']).any():
                st.sidebar.info("Aggregating sales by summing for duplicate dates.")
                df_processed_agg = df_input.groupby('date')['sales'].sum().reset_index()
            else:
                df_processed_agg = df_input[['date', 'sales']].copy()
            
            df_processed_agg = df_processed_agg.sort_values(by='date').reset_index(drop=True)
            
            if len(df_processed_agg) < 10:
                st.sidebar.warning("Warning: Very few data points after processing.")
            
            st.session_state.active_data = df_processed_agg
            st.session_state.is_sample_data = False
            st.sidebar.success(f"File '{st.session_state.data_source_name}' loaded!")

        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")
            st.session_state.active_data = None
            return
    
    if load_sample_button:
        st.session_state.active_data = create_rich_sample_data()
        st.session_state.data_source_name = "Sample Store/Item Data"
        st.session_state.is_sample_data = True
        st.session_state.selected_store = None 
        st.session_state.selected_item = None


    if st.session_state.active_data is not None:
        data_for_forecast = None
        
        if st.session_state.is_sample_data:
            st.subheader(f"📋 Sample Data Configuration: {st.session_state.data_source_name}")
            
            all_stores = sorted(st.session_state.active_data['store'].unique())
            if not st.session_state.selected_store and all_stores:
                st.session_state.selected_store = all_stores[0]

            selected_store_val = st.sidebar.selectbox(
                "Select Store", 
                all_stores, 
                index=all_stores.index(st.session_state.selected_store) if st.session_state.selected_store in all_stores else 0
            )
            st.session_state.selected_store = selected_store_val
            
            items_in_store = sorted(st.session_state.active_data[st.session_state.active_data['store'] == selected_store_val]['item'].unique())
            if not st.session_state.selected_item and items_in_store:
                 st.session_state.selected_item = items_in_store[0]
            elif st.session_state.selected_item not in items_in_store and items_in_store: # if previous item not in new store
                 st.session_state.selected_item = items_in_store[0]


            selected_item_val = st.sidebar.selectbox(
                "Select Item", 
                items_in_store, 
                index=items_in_store.index(st.session_state.selected_item) if st.session_state.selected_item in items_in_store else 0
            )
            st.session_state.selected_item = selected_item_val

            data_for_forecast = st.session_state.active_data[
                (st.session_state.active_data['store'] == selected_store_val) &
                (st.session_state.active_data['item'] == selected_item_val)
            ][['date', 'sales']].copy()
            data_for_forecast = data_for_forecast.sort_values(by='date').reset_index(drop=True)
            st.dataframe(data_for_forecast.head(), use_container_width=True)
        else:
            st.subheader(f"📋 Data Preview: {st.session_state.data_source_name}")
            data_for_forecast = st.session_state.active_data
            st.dataframe(data_for_forecast.head(), use_container_width=True)

        min_data_date = data_for_forecast['date'].min()
        max_data_date = data_for_forecast['date'].max()

        if (max_data_date - min_data_date).days < 14:
             st.warning("Dataset for forecast is small (less than 14 days). Results may be unreliable.")
             return

        default_split_idx = int(len(data_for_forecast) * 0.8)
        default_split_date_val = data_for_forecast['date'].iloc[default_split_idx] if default_split_idx > 0 and default_split_idx < len(data_for_forecast) else min_data_date + (max_data_date - min_data_date) * 0.5
        
        min_split_allowable = min_data_date + pd.Timedelta(days=7) 
        max_split_allowable = max_data_date - pd.Timedelta(days=1) 

        if min_split_allowable >= max_split_allowable:
            st.sidebar.error("Not enough data range in the selected series for a valid train/test split.")
            return

        split_date_input = st.sidebar.date_input(
            "Select Split Date",
            value=default_split_date_val,
            min_value=min_split_allowable,
            max_value=max_split_allowable,
            help="Data up to this date is for training."
        )
        split_date_selected = pd.to_datetime(split_date_input)
        
        num_days_to_forecast = st.sidebar.number_input("Number of days to forecast", min_value=7, max_value=365, value=30, help="How many days after the split date to predict.")

        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Run Forecast Analysis", type="primary", use_container_width=True):
            if data_for_forecast is not None and not data_for_forecast.empty:
                with st.spinner("⏳ Running forecast analysis... Please wait."):
                    run_analysis(data_for_forecast, split_date_selected, num_days_to_forecast)
                st.success("✅ Analysis Complete!")
            else:
                st.error("No data available to run analysis. Please select valid sample data filters or upload a file.")
        
    else:
        st.info("👋 Welcome! Upload a CSV or click 'Load Sample Data' in the sidebar to begin.")
        st.markdown("""
        #### How to get started:
        1.  Use the **sidebar** to configure your analysis.
        2.  **Upload a CSV file**: Click "Browse files".
            * It must contain 'date' and 'sales' columns.
        3.  Or, **Load Sample Data**: Click the "Load Sample Data" button.
            * Then, select a Store and Item from the dropdowns that appear.
        4.  Adjust the **Split Date** and **Number of days to forecast**.
        5.  Click **"Run Forecast Analysis"**! 🚀
        """)

if __name__ == "__main__":
    main()
