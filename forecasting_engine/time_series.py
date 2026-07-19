import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

def decompose_series(series, period=7, model="additive"):
    return seasonal_decompose(series, period=period, model=model, extrapolate_trend="freq")

def fit_arima(train_series, order=None):
    if order is not None:
        model = ARIMA(train_series, order=order).fit()
        return model, order
    best_aic, best_order, best_model = np.inf, None, None
    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 4):
                try:
                    m = ARIMA(train_series, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic, best_order, best_model = m.aic, (p, d, q), m
                except Exception:
                    continue
    return best_model, best_order

def forecast_arima(model, steps):
    forecast = model.get_forecast(steps=steps)
    return forecast.predicted_mean, forecast.conf_int()

def fit_prophet(train_df):
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet is not installed. Run: pip install prophet")
    model = Prophet()
    model.fit(train_df)
    return model

def forecast_prophet(model, periods, freq="D"):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    return model.predict(future)

def evaluate_forecast(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return {"RMSE": rmse, "MAE": mae}

def render():
    import streamlit as st
    import matplotlib.pyplot as plt
    st.subheader("Sales Demand Forecasting")
    st.caption("ARIMA & Prophet forecasting with trend / seasonality decomposition")
    data_source = st.radio("Dataset Source", ["Built-in Demo Data", "Upload CSV"], key="forecast_source")
    if data_source == "Built-in Demo Data":
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        trend = np.linspace(100, 180, len(dates))
        seasonality = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
        noise = np.random.normal(0, 4, len(dates))
        values = trend + seasonality + noise
        df = pd.DataFrame({"date": dates, "sales": values})
    else:
        uploaded = st.file_uploader("Upload CSV with 'date' and 'sales' columns", type=["csv"], key="forecast_upload")
        if uploaded is None:
            st.info("Upload a CSV to continue, or switch to the built-in demo data.")
            return
        df = pd.read_csv(uploaded, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    series = pd.Series(df["sales"].values, index=pd.DatetimeIndex(df["date"]))
    horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
    test_size = st.slider("Hold-out size for evaluation (days)", 7, 60, 20)
    if st.button("Run Forecast"):
        train, test = series.iloc[:-test_size], series.iloc[-test_size:]
        st.subheader("Trend & Seasonality Decomposition")
        decomposition = decompose_series(series, period=7)
        fig1, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(decomposition.trend, color="navy"); axes[0].set_title("Trend")
        axes[1].plot(decomposition.seasonal, color="teal"); axes[1].set_title("Seasonality")
        axes[2].plot(decomposition.resid, color="gray"); axes[2].set_title("Residuals")
        plt.tight_layout()
        st.pyplot(fig1)
        st.subheader("ARIMA vs Prophet Forecast")
        with st.spinner("Fitting ARIMA..."):
            arima_model, order = fit_arima(train.values)
            arima_pred, _ = forecast_arima(arima_model, steps=test_size)
        st.caption(f"Best ARIMA order found: {order}")
        prophet_pred = None
        if PROPHET_AVAILABLE:
            with st.spinner("Fitting Prophet..."):
                prophet_train = pd.DataFrame({"ds": train.index, "y": train.values})
                prophet_model = fit_prophet(prophet_train)
                prophet_result = forecast_prophet(prophet_model, periods=test_size, freq="D")
                prophet_pred = prophet_result["yhat"].values[-test_size:]
        else:
            st.warning("Prophet is not installed — showing ARIMA only. Run: pip install prophet")
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series.index, series.values, label="Actual", color="black", linewidth=1)
        ax.plot(test.index, arima_pred, label="ARIMA Forecast", color="steelblue")
        if prophet_pred is not None:
            ax.plot(test.index, prophet_pred, label="Prophet Forecast", color="firebrick")
        ax.axvline(train.index[-1], color="gray", linestyle="--", linewidth=1)
        ax.legend()
        ax.set_title("Sales Forecast: ARIMA vs Prophet")
        st.pyplot(fig2)
        st.subheader("Evaluation Metrics")
        arima_metrics = evaluate_forecast(test.values, arima_pred)
        cols = st.columns(2)
        cols[0].metric("ARIMA RMSE", f"{arima_metrics['RMSE']:.2f}")
        cols[0].metric("ARIMA MAE", f"{arima_metrics['MAE']:.2f}")
        if prophet_pred is not None:
            prophet_metrics = evaluate_forecast(test.values, prophet_pred)
            cols[1].metric("Prophet RMSE", f"{prophet_metrics['RMSE']:.2f}")
            cols[1].metric("Prophet MAE", f"{prophet_metrics['MAE']:.2f}")
        with st.expander("Forecast beyond hold-out data"):
            full_arima_model, _ = fit_arima(series.values, order=order)
            future_pred, future_ci = forecast_arima(full_arima_model, steps=horizon)
            st.line_chart(pd.Series(future_pred, name="ARIMA future forecast"))