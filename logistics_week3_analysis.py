
import os
import pandas as pd
import matplotlib.pyplot as plt

# Create output folder
os.makedirs("outputs/week3", exist_ok=True)

# Load datasets
orders = pd.read_csv("data/olist_orders_dataset.csv")
items = pd.read_csv("data/olist_order_items_dataset.csv")

# Convert date columns
date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Calculate delivery time and delay
orders["delivery_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_purchase_timestamp"]
).dt.total_seconds() / 86400

orders["delay_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

# Select delivered orders with valid delivery times
delivered = orders[
    (orders["order_status"] == "delivered")
    & (orders["delivery_days"] >= 0)
].copy()

delivered["delivery_result"] = (
    delivered["delay_days"].apply(
        lambda x: "Late" if x > 0 else "On Time"
    )
)

# Aggregate item value and freight cost by order
item_summary = items.groupby("order_id").agg(
    order_value=("price", "sum"),
    freight_cost=("freight_value", "sum")
).reset_index()

analysis = delivered.merge(
    item_summary, on="order_id", how="left"
)

# -----------------------------
# 1. Descriptive statistics
# -----------------------------
print("\n--- DELIVERY TIME STATISTICS ---")
print(analysis["delivery_days"].describe())

print("\nMedian delivery time:",
      analysis["delivery_days"].median())

print("\nOn-time vs late deliveries:")
print(analysis["delivery_result"].value_counts())

print("\nAverage order value:",
      analysis["order_value"].mean())

print("\nAverage freight cost:",
      analysis["freight_cost"].mean())

# Save summary statistics
analysis["delivery_days"].describe().to_csv(
    "outputs/week3/delivery_statistics.csv"
)

# -----------------------------
# 2. Delivery time histogram
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(analysis["delivery_days"].dropna(),
         bins=30, edgecolor="black")
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("outputs/week3/delivery_time_distribution.png")
plt.close()

# -----------------------------
# 3. On-time vs late deliveries
# -----------------------------
counts = analysis["delivery_result"].value_counts()

plt.figure(figsize=(7, 5))
counts.plot(kind="bar", edgecolor="black")
plt.title("On-Time vs Late Deliveries")
plt.xlabel("Delivery Result")
plt.ylabel("Number of Orders")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/week3/on_time_vs_late.png")
plt.close()

# -----------------------------
# 4. Monthly delivery trend
# -----------------------------
analysis["purchase_month"] = (
    analysis["order_purchase_timestamp"].dt.to_period("M")
)

monthly = analysis.groupby("purchase_month")[
    "delivery_days"
].mean()

plt.figure(figsize=(10, 5))
monthly.index = monthly.index.astype(str)
monthly.plot(marker="o")
plt.title("Average Delivery Time by Month")
plt.xlabel("Purchase Month")
plt.ylabel("Average Delivery Time (Days)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/week3/monthly_delivery_trend.png")
plt.close()

# -----------------------------
# 5. Delivery time box plot
# -----------------------------
plt.figure(figsize=(7, 5))
plt.boxplot(analysis["delivery_days"].dropna())
plt.title("Delivery Time: Outlier Analysis")
plt.ylabel("Delivery Time (Days)")
plt.tight_layout()
plt.savefig("outputs/week3/delivery_time_boxplot.png")
plt.close()

# -----------------------------
# 6. Correlation heatmap
# -----------------------------
correlation_data = analysis[
    ["delivery_days", "delay_days",
     "order_value", "freight_cost"]
].dropna()

corr = correlation_data.corr(numeric_only=True)

plt.figure(figsize=(7, 5))
plt.imshow(corr, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns,
           rotation=35, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha="center", va="center")

plt.title("Correlation Between Logistics Variables")
plt.tight_layout()
plt.savefig("outputs/week3/correlation_heatmap.png")
plt.close()

# Save processed analysis dataset
analysis.to_csv(
    "outputs/week3/logistics_eda_results.csv",
    index=False
)

print("\nWeek 3 analysis completed successfully!")
print("Charts and results saved in outputs/week3/")
