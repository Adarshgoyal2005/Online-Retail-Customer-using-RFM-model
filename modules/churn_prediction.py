import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


CHURN_RECENCY_THRESHOLD = 180  # days


def predict_churn(rfm_df):
    """
    Label customers as churned (Recency > threshold) and train two classifiers.

    Returns a dict with:
        churn_count        – number of churned customers
        active_count       – number of active customers
        churn_pct          – percentage churned
        lr_accuracy        – Logistic Regression accuracy (%)
        dt_accuracy        – Decision Tree accuracy (%)
        threshold_days     – recency threshold used
    """
    df = rfm_df.copy()
    df["Churn"] = (df["Recency"] > CHURN_RECENCY_THRESHOLD).astype(int)

    X = df[["Recency", "Frequency", "Monetary"]].values
    y = df["Churn"].values

    # Need at least 2 classes to train
    if len(np.unique(y)) < 2:
        return {
            "churn_count": int(y.sum()),
            "active_count": int((y == 0).sum()),
            "churn_pct": round(y.mean() * 100, 1),
            "lr_accuracy": "N/A",
            "dt_accuracy": "N/A",
            "threshold_days": CHURN_RECENCY_THRESHOLD,
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_acc = round(accuracy_score(y_test, lr.predict(X_test_s)) * 100, 1)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = round(accuracy_score(y_test, dt.predict(X_test)) * 100, 1)

    total = len(y)
    churn_count = int(y.sum())

    return {
        "churn_count": churn_count,
        "active_count": total - churn_count,
        "churn_pct": round(churn_count / total * 100, 1),
        "lr_accuracy": lr_acc,
        "dt_accuracy": dt_acc,
        "threshold_days": CHURN_RECENCY_THRESHOLD,
    }
