import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go

# Function to load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('your_data.csv')  # Replace with your actual data file name
    df['date'] = pd.to_datetime(df['date'])
    return df

# Prepare data for Prophet
def prepare_data_for_prophet(df):
    return df.rename(columns={'date': 'ds', 'sales': 'y'})

# Train Prophet model
def train_prophet_model(train_data):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train_data)
    return model

# Make predictions using Prophet
def make_prophet_predictions(model, future_dates):
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'prediction'})

# Evaluate model
def evaluate_model(true_values, predictions):
    mae = mean_absolute_error(true_values, predictions)
    mape = mean_absolute_percentage_error(true_values, predictions)
    return mae, mape

# Streamlit app
def main():
    st.title('Sales Forecasting with Prophet')

    # Load data
    df = load_data()

    # Sidebar for user input
    st.sidebar.header('User Input')
    selected_store = st.sidebar.selectbox('Select Store', sorted(df['store'].unique()))
    selected_item = st.sidebar.selectbox('Select Item', sorted(df['item'].unique()))
    split_date = st.sidebar.date_input('Select Split Date', value=pd.to_datetime('2017-11-30'))
    forecast_days = st.sidebar.number_input('Number of days to forecast', min_value=1, max_value=365, value=30)

    # Filter data based on user selection
    df_filtered = df[(df['store'] == selected_store) & (df['item'] == selected_item)]

    # Prepare data
    train_set = df_filtered[df_filtered['date'] <= pd.to_datetime(split_date)]
    test_set = df_filtered[df_filtered['date'] > pd.to_datetime(split_date)]

    # Group by date and calculate mean sales
    train_set_daily = train_set.groupby('date')['sales'].mean().reset_index()
    test_set_daily = test_set.groupby('date')['sales'].mean().reset_index()

    # Prepare data for Prophet
    train_prophet = prepare_data_for_prophet(train_set_daily)
    test_prophet = prepare_data_for_prophet(test_set_daily)

    # Train Prophet model
    with st.spinner('Training Prophet model...'):
        prophet_model = train_prophet_model(train_prophet)

    # Make predictions
    last_date = df_filtered['date'].max()
    future_dates = pd.date_range(start=last_date, periods=forecast_days)
    prophet_predictions = make_prophet_predictions(prophet_model, future_dates)

    # Evaluate Prophet model
    prophet_mae, prophet_mape = evaluate_model(test_prophet['y'], prophet_predictions['prediction'][:len(test_prophet)])

    # Display results
    st.header('Forecast Results')
    st.write(f"Prophet Model - MAE: {prophet_mae:.2f}, MAPE: {prophet_mape:.2%}")

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_set_daily['date'], y=train_set_daily['sales'], name='Training Data'))
    fig.add_trace(go.Scatter(x=test_set_daily['date'], y=test_set_daily['sales'], name='Test Data'))
    fig.add_trace(go.Scatter(x=prophet_predictions['date'], y=prophet_predictions['prediction'], name='Prophet Forecast'))
    fig.update_layout(title='Sales Forecast', xaxis_title='Date', yaxis_title='Sales')
    st.plotly_chart(fig)

    # Display forecast data
    st.subheader('Forecast Data')
    st.dataframe(prophet_predictions)

if __name__ == '__main__':
    main()