"""Part A: run the seven SQL queries, cross-check them in pandas, and profile order value.

Run with ``python -m src.analytics``. Results go to ``outputs/metrics/part_a_analytics.json``.
"""
from __future__ import annotations

import logging
import re
import sqlite3

import numpy as np
import pandas as pd

from src.config import CLEAN, DB_PATH, SQL_FILE, save_metrics

logger = logging.getLogger("copilot.analytics")


def load_queries() -> dict[str, str]:
    """Split ``sql/queries.sql`` into ``{"Q1": "<sql>", ...}`` on the ``-- Qn:`` markers."""
    text = SQL_FILE.read_text(encoding="utf-8")
    parts = re.split(r"^-- (Q\d+):.*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def run_queries() -> dict[str, pd.DataFrame]:
    """Execute every query against the SQLite database with ``pd.read_sql_query``."""
    with sqlite3.connect(DB_PATH) as con:
        return {name: pd.read_sql_query(sql, con) for name, sql in load_queries().items()}


def summarise_queries(results: dict[str, pd.DataFrame]) -> dict:
    """Reduce each query result to the figures the report needs."""
    q2, q5, q6, q7 = results["Q2"], results["Q5"], results["Q6"], results["Q7"]
    largest = q2.loc[q2["revenue"].idxmax()]
    top = q6[q6["row_number_in_city"] == 1]
    return {
        "Q1_customers": len(results["Q1"]),
        "Q2_rows": len(q2),
        "Q2_total_inr": round(float(q2["revenue"].sum()), 2),
        "Q2_largest_cell": {"city": largest["city"], "category": largest["category"], "revenue": float(largest["revenue"])},
        "Q3_products": len(results["Q3"]),
        "Q4_customers": len(results["Q4"]),
        "Q5_customers": len(q5),
        "Q5_average_inr": float(q5["avg_spend"].iloc[0]),
        "Q6_top_per_city": {r.city: {"customer_id": r.customer_id, "total_spend": float(r.total_spend)}
                            for r in top.itertuples()},
        "Q6_rows": len(q6),
        "Q6_rank_differs_from_row_number": int((q6["rank_in_city"] != q6["row_number_in_city"]).sum()),
        "Q6_dense_rank_differs_from_rank": int((q6["rank_in_city"] != q6["dense_rank_in_city"]).sum()),
        "Q7_rows": len(q7),
        "Q7_last_cumulative_inr": float(q7["cumulative_revenue"].iloc[-1]),
    }


def _completed_orders() -> pd.DataFrame:
    orders = pd.read_csv(CLEAN / "orders.csv")
    return orders[orders["status"].eq("completed")]


def pandas_cross_check(sql_total: float) -> dict:
    """Rebuild Q2 with a pivot table and aggregate per city with named aggregation."""
    completed = _completed_orders()
    customers = pd.read_csv(CLEAN / "customers.csv")[["customer_id", "city"]]
    products = pd.read_csv(CLEAN / "products.csv")[["product_id", "category"]]
    joined = completed.merge(customers, on="customer_id").merge(products, on="product_id")
    pivot = pd.pivot_table(joined, values="order_amount", index="city", columns="category",
                           aggfunc="sum", margins=True, margins_name="Total")
    grand_total = round(float(pivot.loc["Total", "Total"]), 2)
    per_city = joined.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                          orders=("order_amount", "count")).round(2)
    return {
        "pivot_grand_total_inr": grand_total,
        "sql_total_inr": sql_total,
        "totals_match": abs(grand_total - sql_total) < 0.01,
        "per_city": per_city.reset_index().to_dict("records"),
    }


def exploratory_statistics() -> dict:
    """Distribution, IQR outlier fences and correlations of completed orders."""
    completed = _completed_orders()
    amount = completed["order_amount"]
    q1, q3 = float(amount.quantile(0.25)), float(amount.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    corr = completed[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    off_diagonal = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    strongest = off_diagonal.abs().idxmax()
    return {
        "mean": round(float(amount.mean()), 2), "median": round(float(amount.median()), 2),
        "skewness": round(float(amount.skew()), 2),
        "q1": round(q1, 2), "q3": round(q3, 2), "iqr": round(iqr, 2),
        "lower_fence": round(lower, 2), "upper_fence": round(upper, 2),
        "outliers_above": int((amount > upper).sum()), "outliers_below": int((amount < lower).sum()),
        "correlation": corr.round(3).to_dict(),
        "strongest_pair": list(strongest), "strongest_value": round(float(off_diagonal[strongest]), 3),
    }


def run() -> dict:
    """Run the SQL, the pandas cross-check and the exploratory statistics."""
    results = run_queries()
    summary = summarise_queries(results)
    payload = {"sql": summary, "cross_check": pandas_cross_check(summary["Q2_total_inr"]),
               "exploratory": exploratory_statistics()}
    save_metrics("part_a_analytics", payload)
    logger.info("Part A analytics finished: %d queries", len(results))
    return payload


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging()
    run()
