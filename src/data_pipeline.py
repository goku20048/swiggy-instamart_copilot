"""Part A: audit the raw exports, clean them with explicit rules, and load SQLite.

Run with ``python -m src.data_pipeline``. Every rule records how many rows or values it
changed; those counts go to ``outputs/metrics/part_a_pipeline.json`` and the README.
"""
from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from src.config import CLEAN, DB_PATH, RAW, save_metrics

logger = logging.getLogger("copilot.data")

GENDER_MAP = {"male": "male", "m": "male", "female": "female", "f": "female"}
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _read_text(name: str) -> pd.DataFrame:
    """Read a raw export with every cell as literal text, so blanks and 'N/A' stay visible."""
    return pd.read_csv(RAW / f"{name}.csv", dtype=str, keep_default_na=False)


def profile_table(name: str) -> dict:
    """Shape, inferred dtypes and per-column missing counts, using pandas' default parser."""
    df = pd.read_csv(RAW / f"{name}.csv")
    return {
        "shape": list(df.shape),
        "columns": {
            col: {"dtype": str(df[col].dtype), "missing": int(df[col].isna().sum()),
                  "missing_pct": round(float(df[col].isna().mean() * 100), 2)}
            for col in df.columns
        },
    }


def audit() -> dict:
    """Profile the three raw tables and compute the data-quality figures."""
    orders = _read_text("orders")
    customers = _read_text("customers")
    delivery_text = orders["delivery_minutes"].str.strip()
    age_blank = customers["age"].str.strip().eq("")
    age_numeric = pd.to_numeric(customers["age"].where(~age_blank), errors="coerce")
    gender_blank = customers["gender"].str.strip().eq("")
    figures = {
        "orders_rows": len(orders),
        "orders_exact_duplicates": int(orders.duplicated().sum()),
        "orders_pending": int(orders["status"].eq("pending").sum()),
        "orders_distinct_order_id": int(orders["order_id"].nunique()),
        "orders_non_numeric_amount": int(pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum()),
        "orders_negative_delivery": int(pd.to_numeric(delivery_text, errors="coerce").lt(0).sum()),
        "orders_blank_delivery": int(delivery_text.eq("").sum()),
        "customers_blank_age": int(age_blank.sum()),
        "customers_age_outside_10_100": int((age_numeric.notna() & ~age_numeric.between(10, 100)).sum()),
        "customers_raw_city_spellings": int(customers["city"].nunique()),
        "customers_raw_gender_spellings": int(customers.loc[~gender_blank, "gender"].nunique()),
        "customers_blank_gender": int(gender_blank.sum()),
    }
    return {"profiles": {n: profile_table(n) for n in ("customers", "products", "orders")}, "figures": figures}


def clean_customers(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardise city and gender, repair impossible ages, parse the signup date."""
    df = raw.copy()
    log: dict[str, int] = {}
    tidy_city = df["city"].str.strip().str.title()
    log["city_values_changed"] = int((tidy_city != df["city"]).sum())
    df["city"] = tidy_city

    tidy_gender = df["gender"].str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
    log["gender_values_changed"] = int((tidy_gender != df["gender"]).sum())
    log["gender_blank_to_unknown"] = int(tidy_gender.eq("unknown").sum())
    df["gender"] = tidy_gender

    age = pd.to_numeric(df["age"], errors="coerce")
    impossible = age.notna() & ~age.between(10, 100)
    log["ages_set_missing_out_of_range"] = int(impossible.sum())
    age = age.mask(impossible)
    log["ages_filled_with_city_median"] = int(age.isna().sum())
    df["age"] = age.fillna(age.groupby(df["city"]).transform("median"))

    df["signup_date"] = pd.to_datetime(df["signup_date"], format="%Y-%m-%d")
    return df, log


def clean_orders(raw: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the five order rules, in the documented order."""
    df = raw.copy()
    log: dict[str, int] = {"rows_in": len(df)}

    # Rule 1: timestamps become datetimes.
    df["order_ts"] = pd.to_datetime(df["order_ts"], format=TIMESTAMP_FORMAT)
    log["timestamps_parsed"] = int(df["order_ts"].notna().sum())

    # Rule 2: remove exact duplicate rows.
    before = len(df)
    df = df.drop_duplicates()
    log["exact_duplicates_removed"] = before - len(df)

    # Rule 3: one record per order, keeping the latest timestamp.
    before = len(df)
    df = df.sort_values(["order_id", "order_ts"], kind="mergesort").drop_duplicates("order_id", keep="last")
    log["superseded_records_removed"] = before - len(df)

    # Rule 4: unparseable amounts are recomputed from the catalogue price.
    for col in ("quantity", "discount_pct"):
        df[col] = pd.to_numeric(df[col])
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    missing_amount = df["order_amount"].isna()
    log["amounts_recomputed_from_price"] = int(missing_amount.sum())
    price = df["product_id"].map(products.set_index("product_id")["price_inr"])
    recomputed = (price * df["quantity"] * (1 - df["discount_pct"] / 100)).round(2)
    df["order_amount"] = df["order_amount"].where(~missing_amount, recomputed)

    # Rule 5: negative durations become missing; only completed orders are imputed.
    df["delivery_minutes"] = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    negative = df["delivery_minutes"].lt(0)
    log["negative_delivery_set_missing"] = int(negative.sum())
    df["delivery_minutes"] = df["delivery_minutes"].mask(negative)
    completed = df["status"].eq("completed")
    category = df["product_id"].map(products.set_index("product_id")["category"])
    category_median = df["delivery_minutes"].where(completed).groupby(category).transform("median")
    to_fill = completed & df["delivery_minutes"].isna()
    log["completed_delivery_imputed"] = int(to_fill.sum())
    df.loc[to_fill, "delivery_minutes"] = category_median[to_fill]
    log["cancelled_delivery_left_missing"] = int((df["status"].eq("cancelled") & df["delivery_minutes"].isna()).sum())

    log["rows_out"] = len(df)
    return df.sort_values("order_id", kind="mergesort").reset_index(drop=True), log


def store(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> None:
    """Write clean CSVs and SQLite tables; dates and timestamps are stored as text."""
    CLEAN.mkdir(parents=True, exist_ok=True)
    customers_out = customers.assign(signup_date=customers["signup_date"].dt.strftime("%Y-%m-%d"))
    orders_out = orders.assign(order_ts=orders["order_ts"].dt.strftime(TIMESTAMP_FORMAT))
    tables = {"customers": customers_out, "products": products, "orders": orders_out}
    for name, frame in tables.items():
        frame.to_csv(CLEAN / f"{name}.csv", index=False)
    with sqlite3.connect(DB_PATH) as con:
        for name, frame in tables.items():
            frame.to_sql(name, con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)")


def run() -> dict:
    """Audit, clean and store. Returns the audit figures and the per-rule change counts."""
    audit_result = audit()
    products = pd.read_csv(RAW / "products.csv")
    customers, customer_log = clean_customers(_read_text("customers"))
    orders, order_log = clean_orders(_read_text("orders"), products)
    store(customers, products, orders)
    result = {
        "audit": audit_result,
        "customer_rules": customer_log,
        "order_rules": order_log,
        "after_cleaning": {
            "city_counts": customers["city"].value_counts().to_dict(),
            "gender_counts": customers["gender"].value_counts().to_dict(),
            "age_min": float(customers["age"].min()), "age_max": float(customers["age"].max()),
            "age_missing": int(customers["age"].isna().sum()),
            "non_integer_ages": int((customers["age"] % 1 != 0).sum()),
            "status_counts": orders["status"].value_counts().to_dict(),
            "completed_revenue_inr": round(float(orders.loc[orders["status"].eq("completed"), "order_amount"].sum()), 2),
            "delivery_missing_by_status": orders.groupby("status")["delivery_minutes"].apply(
                lambda s: int(s.isna().sum())).to_dict(),
        },
    }
    save_metrics("part_a_pipeline", result)
    logger.info("Part A pipeline finished: %d orders, %d customers", len(orders), len(customers))
    return result


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging()
    run()
