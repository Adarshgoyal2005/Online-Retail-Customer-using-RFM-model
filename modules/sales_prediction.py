import numpy as np
from sklearn.linear_model import LinearRegression


def predict_sales(df, forecast_months=3):
    """
    Aggregate monthly sales and predict future revenue using Linear Regression.

    Returns a dict:
        historical  – list of {month, sales} dicts
        forecast    – list of {month, sales} dicts
        r2_score    – model fit quality
    """
    df = df.copy()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    monthly = (
        df.groupby("YearMonth")["TotalAmount"].sum().reset_index()
    )
    monthly = monthly.sort_values("YearMonth")
    monthly["MonthIndex"] = range(len(monthly))

    X = monthly[["MonthIndex"]].values
    y = monthly["TotalAmount"].values

    model = LinearRegression()
    model.fit(X, y)

    from sklearn.metrics import r2_score as r2
    r2_val = round(r2(y, model.predict(X)), 4)

    # Future indices
    last_idx = monthly["MonthIndex"].max()
    future_indices = np.array(
        [[last_idx + i + 1] for i in range(forecast_months)]
    )
    future_sales = model.predict(future_indices)

    # Build last period for labels
    last_period = monthly["YearMonth"].max()
    forecast_periods = [
        str(last_period + i + 1) for i in range(forecast_months)
    ]

    historical = [
        {"month": str(row.YearMonth), "sales": round(float(row.TotalAmount), 2)}
        for row in monthly.itertuples()
    ]
    forecast = [
        {"month": forecast_periods[i], "sales": round(float(future_sales[i]), 2)}
        for i in range(forecast_months)
    ]

    return {
        "historical": historical,
        "forecast": forecast,
        "r2_score": r2_val,
        "next_month_sales": round(float(future_sales[0]), 2),
    }
