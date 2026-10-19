
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "olist_orders_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

print("Original rows:", len(df))
print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate order IDs
df = df.drop_duplicates(subset="order_id")

# Convert dates safely
date_cols = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Keep delivered orders with required dates
cleaned = df[
    (df["order_status"] == "delivered")
    & df["order_purchase_timestamp"].notna()
    & df["order_delivered_customer_date"].notna()
    & df["order_estimated_delivery_date"].notna()
].copy()

# Calculate delivery duration
cleaned["delivery_time_days"] = (
    cleaned["order_delivered_customer_date"]
    - cleaned["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Flag impossible negative durations
cleaned["invalid_duration"] = (
    cleaned["delivery_time_days"] < 0
)

# IQR outlier detection
q1 = cleaned["delivery_time_days"].quantile(0.25)
q3 = cleaned["delivery_time_days"].quantile(0.75)
iqr = q3 - q1

lower = max(0, q1 - 1.5 * iqr)
upper = q3 + 1.5 * iqr

cleaned["delivery_time_outlier"] = (
    (cleaned["delivery_time_days"] < lower)
    | (cleaned["delivery_time_days"] > upper)
)

# Min-max normalization
minimum = cleaned["delivery_time_days"].min()
maximum = cleaned["delivery_time_days"].max()

if maximum > minimum:
    cleaned["delivery_time_normalized"] = (
        (cleaned["delivery_time_days"] - minimum)
        / (maximum - minimum)
    )
else:
    cleaned["delivery_time_normalized"] = 0.0

# Save cleaned data
cleaned.to_csv(
    OUTPUT_DIR / "week2_cleaned_orders.csv",
    index=False
)

# Save data-quality summary
summary = pd.DataFrame({
    "Metric": [
        "Original rows",
        "Rows after duplicate removal",
        "Eligible delivered orders",
        "Missing purchase timestamps",
        "Missing actual delivery timestamps",
        "Missing estimated delivery timestamps",
        "Invalid negative durations",
        "Delivery-time outliers flagged"
    ],
    "Value": [
        # Original row count is captured before filtering below
        len(pd.read_csv(DATA_FILE)),
        len(df),
        len(cleaned),
        df["order_purchase_timestamp"].isna().sum(),
        df["order_delivered_customer_date"].isna().sum(),
        df["order_estimated_delivery_date"].isna().sum(),
        cleaned["invalid_duration"].sum(),
        cleaned["delivery_time_outlier"].sum()
    ]
})

summary.to_csv(
    OUTPUT_DIR / "week2_data_quality_summary.csv",
    index=False
)

print("\nPreprocessing completed.")
print("Cleaned data and quality summary saved in outputs.")
