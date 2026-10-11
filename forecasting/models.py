"""
forecasting/models.py — Forecasting models for TechSight.

Implements moving average and linear regression baselines
for predicting future patent/research activity.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import FORECAST_HORIZON, MOVING_AVG_WINDOW, TEST_YEARS

logger = logging.getLogger(__name__)


def moving_average_forecast(series, window=MOVING_AVG_WINDOW,
                            horizon=FORECAST_HORIZON):
    """
    Simple moving average forecast.

    Uses the last `window` values to predict the next `horizon` values.
    Each prediction uses the previous predictions in a rolling fashion.

    Args:
        series: Array-like of historical values (chronological)
        window: Number of periods for the moving average
        horizon: Number of future periods to forecast

    Returns:
        Array of predicted values
    """
    values = list(series)
    predictions = []

    for _ in range(horizon):
        recent = values[-window:]
        pred = np.mean(recent)
        predictions.append(max(0, pred))
        values.append(pred)

    return np.array(predictions)


def linear_regression_forecast(years, values, future_years):
    """
    Linear regression forecast.

    Fits a simple linear trend and extrapolates to future years.

    Args:
        years: Array of historical year values
        values: Array of historical count values
        future_years: Array of years to predict

    Returns:
        Array of predicted values, fitted model
    """
    X_train = np.array(years).reshape(-1, 1)
    y_train = np.array(values)

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_future = np.array(future_years).reshape(-1, 1)
    predictions = model.predict(X_future)

    # Ensure non-negative
    predictions = np.maximum(predictions, 0)

    return predictions, model


def evaluate_model(actual, predicted):
    """
    Compute evaluation metrics for forecast accuracy.

    Returns:
        Dict with MAE, RMSE, and MAPE
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE (avoid division by zero)
    mask = actual != 0
    if mask.any():
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = 0.0

    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}


def generate_forecasts(indicators_df):
    """
    Generate forecasts for all technologies using both models.

    Args:
        indicators_df: DataFrame with year, technology_name, total_count

    Returns:
        DataFrame of forecasts with confidence bounds
    """
    logger.info("Generating forecasts...")
    all_forecasts = []
    all_metrics = []

    for tech_name in indicators_df["technology_name"].unique():
        tech_data = indicators_df[
            indicators_df["technology_name"] == tech_name
        ].sort_values("year")

        years = tech_data["year"].values
        future_years = np.arange(years[-1] + 1, years[-1] + 1 + FORECAST_HORIZON)

        for series_col in ["patent_count", "paper_count"]:
            values = tech_data[series_col].values.astype(float)

            # --- Train/test split for evaluation ---
            if len(values) > TEST_YEARS + 2:
                train_vals = values[:-TEST_YEARS]
                test_vals = values[-TEST_YEARS:]
                train_years = years[:-TEST_YEARS]
                test_years_arr = years[-TEST_YEARS:]

                # Evaluate moving average
                ma_test_preds = moving_average_forecast(
                    train_vals, window=MOVING_AVG_WINDOW, horizon=TEST_YEARS
                )
                ma_metrics = evaluate_model(test_vals, ma_test_preds)
                ma_metrics["model"] = "moving_average"
                ma_metrics["technology"] = tech_name
                ma_metrics["series"] = series_col
                all_metrics.append(ma_metrics)

                # Evaluate linear regression
                lr_test_preds, _ = linear_regression_forecast(
                    train_years, train_vals, test_years_arr
                )
                lr_metrics = evaluate_model(test_vals, lr_test_preds)
                lr_metrics["model"] = "linear_regression"
                lr_metrics["technology"] = tech_name
                lr_metrics["series"] = series_col
                all_metrics.append(lr_metrics)

            # --- Full forecasts ---
            # Moving average
            ma_preds = moving_average_forecast(values)
            for i, year in enumerate(future_years):
                all_forecasts.append({
                    "technology_name": tech_name,
                    "year": int(year),
                    "series_type": series_col,
                    "model": "moving_average",
                    "predicted_value": round(ma_preds[i]),
                    "lower_bound": round(ma_preds[i] * 0.75),
                    "upper_bound": round(ma_preds[i] * 1.25),
                    "is_future": True,
                })

            # Linear regression
            lr_preds, lr_model = linear_regression_forecast(
                years, values, future_years
            )
            residual_std = np.std(values - lr_model.predict(
                years.reshape(-1, 1)
            ))
            for i, year in enumerate(future_years):
                margin = residual_std * 1.96 * (i + 1) / FORECAST_HORIZON
                all_forecasts.append({
                    "technology_name": tech_name,
                    "year": int(year),
                    "series_type": series_col,
                    "model": "linear_regression",
                    "predicted_value": round(max(0, lr_preds[i])),
                    "lower_bound": round(max(0, lr_preds[i] - margin)),
                    "upper_bound": round(lr_preds[i] + margin),
                    "is_future": True,
                })

    forecasts_df = pd.DataFrame(all_forecasts)
    metrics_df = pd.DataFrame(all_metrics)

    logger.info("Generated %d forecast points", len(forecasts_df))
    return forecasts_df, metrics_df
