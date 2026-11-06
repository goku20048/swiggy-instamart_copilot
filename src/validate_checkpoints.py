"""Part A validation: assert the published checkpoints against the stored clean data.

Run with ``python -m src.validate_checkpoints``. The values are compared with figures freshly
computed from ``data/clean`` and ``data/instamart.db``; a failed assertion stops the pipeline.
Prints exactly ``ALL CHECKPOINTS PASSED`` when every check holds.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from src.config import CLEAN, DB_PATH


def validate() -> None:
    """Raise ``AssertionError`` on the first checkpoint that does not hold."""
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    completed = orders[orders["status"].eq("completed")]
    cancelled = orders[orders["status"].eq("cancelled")]

    # Task 3: customers.
    assert customers["city"].value_counts().to_dict() == {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
    assert customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
    assert customers["age"].notna().all()
    assert customers["age"].between(18, 60).all()

    # Task 4: orders.
    assert len(orders) == 12309
    assert orders["order_id"].is_unique
    assert len(completed) == 11110
    assert len(cancelled) == 1199
    assert not orders["status"].eq("pending").any()
    assert orders["order_amount"].notna().all()
    assert completed["delivery_minutes"].notna().all()
    assert cancelled["delivery_minutes"].isna().sum() == 1199
    assert round(float(completed["order_amount"].sum()), 2) == 2998668.80

    # Task 5: the database holds the same data as the CSV files.
    with sqlite3.connect(DB_PATH) as con:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"customers", "products", "orders"} <= tables
        assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309
        assert con.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 1200
        assert con.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 40
        sample = con.execute("SELECT order_ts FROM orders LIMIT 1").fetchone()[0]
        assert pd.to_datetime(sample, format="%Y-%m-%d %H:%M:%S") is not pd.NaT
        revenue = con.execute("SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status='completed'").fetchone()[0]
        assert revenue == 2998668.80


if __name__ == "__main__":
    validate()
    print("ALL CHECKPOINTS PASSED")
