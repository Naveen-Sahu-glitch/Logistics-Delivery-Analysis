
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

# Create output folder
OUTPUT_DIR = Path("outputs/week4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the logistics dataset
file_path = Path("data/olist_orders_dataset.csv")

if not file_path.exists():
    raise FileNotFoundError(
        "Dataset not found. Check that data/olist_orders_dataset.csv exists."
    )

orders = pd.read_csv(file_path)

# Convert timestamps to datetime
date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column], errors="coerce"
    )

# Keep delivered orders with valid purchase and delivery dates
data = orders[
    (orders["order_status"] == "delivered")
    & orders["order_purchase_timestamp"].notna()
    & orders["order_delivered_customer_date"].notna()
].copy()

# Target: actual delivery duration in days
data["delivery_time_days"] = (
    data["order_delivered_customer_date"]
    - data["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Keep reasonable positive delivery durations
data = data[
    (data["delivery_time_days"] > 0)
    & (data["delivery_time_days"] <= 100)
].copy()

# Use features available at order purchase time.
# Do NOT use actual delivery date or estimated/actual outcomes as predictors.
data["purchase_month"] = data[
    "order_purchase_timestamp"
].dt.month

data["purchase_day_of_week"] = data[
    "order_purchase_timestamp"
].dt.dayofweek

data["purchase_hour"] = data[
    "order_purchase_timestamp"
].dt.hour

data["purchase_day"] = data[
    "order_purchase_timestamp"
].dt.day

features = [
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "purchase_day",
]

X = data[features]
y = data["delivery_time_days"]

# Split chronologically to test on later orders
data = data.sort_values("order_purchase_timestamp")
X = data[features]
y = data["delivery_time_days"]

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Define two regression models
models = {
    "Linear Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression()
    ),
    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=50,
            random_state=42
        )
    ),
}

# Train and evaluate
results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    mae = mean_absolute_error(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)

    results.append({
        "Model": name,
        "MAE_days": mae,
        "RMSE_days": rmse,
        "R2_score": r2,
    })

    predictions[name] = predicted

    print(f"\n{name}")
    print(f"MAE: {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R2 score: {r2:.4f}")

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "model_evaluation.csv", index=False
)

# Save sample actual vs predicted results from the better MAE model
best_model_name = results_df.sort_values(
    "MAE_days"
).iloc[0]["Model"]

comparison = pd.DataFrame({
    "Actual_delivery_days": y_test.to_numpy(),
    "Predicted_delivery_days": predictions[best_model_name],
})

comparison.to_csv(
    OUTPUT_DIR / "actual_vs_predicted.csv", index=False
)

# Plot model MAE comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["MAE_days"])
plt.ylabel("Mean Absolute Error (days)")
plt.title("Week 4: Model Performance Comparison")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "model_comparison.png", dpi=150
)
plt.close()

# Write practical recommendations for further investigation
recommendations = [
    "Review orders predicted to have longer delivery times.",
    "Compare delivery performance by region when reliable location data is available.",
    "Plan staffing and dispatch capacity around historical demand peaks.",
    "Evaluate routes using distance, traffic, and vehicle-capacity data if available.",
    "Compare freight cost and delivery time before changing shipping policies.",
    "Re-train and validate the model with additional operational features.",
]

(OUTPUT_DIR / "optimization_recommendations.txt").write_text(
    "\n".join(recommendations), encoding="utf-8"
)

print("\nWeek 4 model training and evaluation completed!")
print(f"Best model by MAE: {best_model_name}")
print(f"Results saved in: {OUTPUT_DIR.resolve()}")
print("\nNote: This baseline uses purchase-time features only.")
print("It may have limited predictive power; do not treat it as a")
print("production-ready delivery forecast without further validation.")
