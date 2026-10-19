
# Logistics Delivery Performance and Delay Analysis

## Project Overview
This project analyses historical e-commerce delivery data using Python.
It measures delivery performance and explores patterns related to late deliveries.

## Objectives
- Calculate on-time and late delivery rates.
- Measure average delivery time.
- Calculate average delay among late orders.
- Analyse monthly delivery performance.
- Examine freight costs when data is available.
- Support evidence-based logistics decisions.

## Dataset
Brazilian E-Commerce Public Dataset by Olist.

Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

This is historical Brazilian data, not live Indian logistics data.

## Technologies
- Python
- Pandas
- Matplotlib
- Git and GitHub

## Key Performance Indicators
1. On-time delivery rate
2. Late delivery rate
3. Average delivery time
4. Average delay days
5. Average freight cost per eligible order

## Installation
Install dependencies:

pip install -r requirements.txt

Download the Olist dataset and place the required CSV files inside
the data/ folder.

## Run the Project
python logistics_analysis.py

## Output
The script generates:
- delivery_kpis.csv
- monthly_delivery_kpis.csv
- eligible_orders_with_delivery_features.csv
- data_quality_summary.txt
- delivery_performance.png

## Methodology
1. Data collection
2. Data inspection
3. Data cleaning
4. KPI calculation
5. Monthly analysis
6. Data visualisation
7. Interpretation and recommendations

## Limitations
The dataset is historical and Brazil-specific.
Actual results must be calculated before making conclusions.

## References
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
https://pandas.pydata.org/docs/
https://matplotlib.org/stable/
