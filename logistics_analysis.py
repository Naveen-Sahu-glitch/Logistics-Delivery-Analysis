
from pathlib import Path
import pandas as pd

DATA_FILE = Path("data/olist_orders_dataset.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Please put olist_orders_dataset.csv inside the data folder."
    )

orders = pd.read_csv(DATA_FILE)

date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in date_columns:
    orders[col] = pd.to_datetime(
        orders[col], errors="coerce"
    )

eligible = orders[
    (orders["order_status"] == "delivered")
    & orders["order_purchase_timestamp"].notna()
    & orders["order_delivered_customer_date"].notna()
    & orders["order_estimated_delivery_date"].notna()
].copy()

eligible["delivery_time_days"] = (
    eligible["order_delivered_customer_date"]
    - eligible["order_purchase_timestamp"]
).dt.total_seconds() / 86400

eligible["delay_days"] = (
    eligible["order_delivered_customer_date"]
    - eligible["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

eligible["is_late"] = eligible["delay_days"] > 0

if eligible.empty:
    raise ValueError("No eligible delivered orders found.")

kpis = {
    "Eligible delivered orders": len(eligible),
    "On-time delivery rate (%)": (
        (~eligible["is_late"]).mean() * 100
    ),
    "Late delivery rate (%)": (
        eligible["is_late"].mean() * 100
    ),
    "Average delivery time (days)": (
        eligible["delivery_time_days"].mean()
    ),
    "Average delay among late orders (days)": (
        eligible.loc[
            eligible["is_late"], "delay_days"
        ].mean()
    ),
}

print("\nLOGISTICS DELIVERY KPIs")
for name, value in kpis.items():
    print(f"{name}: {value:.2f}" if pd.notna(value) else f"{name}: N/A")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

pd.DataFrame(
    list(kpis.items()),
    columns=["KPI", "Value"]
).to_csv(output_dir / "delivery_kpis.csv", index=False)

print("\nKPI report saved in outputs/delivery_kpis.csv")
    
