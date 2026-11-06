"""Part A: cleaning rules on small frames, and every published checkpoint on the real data."""
import sqlite3
import unittest

import pandas as pd

from src import analytics, data_pipeline
from src.config import CLEAN, DB_PATH
from src.validate_checkpoints import validate


class CleaningRules(unittest.TestCase):
    def test_customers(self):
        raw = pd.DataFrame({"customer_id": ["C1", "C2", "C3", "C4"], "city": [" pune ", "PUNE", "Pune", "delhi"],
                            "signup_date": ["2024-01-01"] * 4, "age": ["20", "999", "", "40"], "gender": ["M", "f", "", "FEMALE"]})
        clean, log = data_pipeline.clean_customers(raw)
        self.assertEqual(clean["city"].tolist(), ["Pune", "Pune", "Pune", "Delhi"])
        self.assertEqual(clean["gender"].tolist(), ["male", "female", "unknown", "female"])
        self.assertEqual(clean["age"].tolist(), [20.0, 20.0, 20.0, 40.0])  # city median replaces 999 and blank
        self.assertEqual(log["ages_set_missing_out_of_range"], 1)
        self.assertEqual(str(clean["signup_date"].dtype)[:8], "datetime")

    def _orders(self):
        return pd.DataFrame({
            "order_id": ["A", "A", "B", "C", "C"], "customer_id": ["C1"] * 5, "product_id": ["P1"] * 5,
            "quantity": ["2"] * 5,
            "order_ts": ["2025-01-01 10:00:00", "2025-01-01 10:05:00", "2025-01-02 10:00:00",
                         "2025-01-03 10:00:00", "2025-01-03 10:00:00"],
            "status": ["pending", "completed", "completed", "cancelled", "cancelled"],
            "discount_pct": ["50"] * 5, "order_amount": ["N/A", "N/A", "10", "5", "5"],
            "delivery_minutes": ["", "-10", "30", "", ""]})

    def test_orders(self):
        products = pd.DataFrame({"product_id": ["P1"], "category": ["Dairy"], "price_inr": [100]})
        clean, log = data_pipeline.clean_orders(self._orders(), products).__iter__()
        self.assertEqual(clean["order_id"].tolist(), ["A", "B", "C"])
        by_id = clean.set_index("order_id")
        self.assertEqual(by_id.loc["A", "status"], "completed")          # latest record wins
        self.assertEqual(by_id.loc["A", "order_amount"], 100.0)          # 100 x 2 x 0.5
        self.assertEqual(by_id.loc["A", "delivery_minutes"], 30.0)       # negative -> category median
        self.assertTrue(pd.isna(by_id.loc["C", "delivery_minutes"]))     # cancelled is never imputed
        self.assertEqual(log["exact_duplicates_removed"], 1)
        self.assertEqual(log["superseded_records_removed"], 1)


class AuditFigures(unittest.TestCase):
    def test_all_reference_figures(self):
        self.assertEqual(data_pipeline.audit()["figures"], {
            "orders_rows": 13084, "orders_exact_duplicates": 119, "orders_pending": 656,
            "orders_distinct_order_id": 12309, "orders_non_numeric_amount": 120, "orders_negative_delivery": 106,
            "orders_blank_delivery": 2541, "customers_blank_age": 61, "customers_age_outside_10_100": 6,
            "customers_raw_city_spellings": 24, "customers_raw_gender_spellings": 10, "customers_blank_gender": 30})


class StoredData(unittest.TestCase):
    def test_validation_passes(self):
        validate()

    def test_timestamp_text_format(self):
        with sqlite3.connect(DB_PATH) as con:
            values = pd.read_sql_query("SELECT order_ts FROM orders", con)["order_ts"]
        self.assertTrue(values.str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all())

    def test_csv_and_database_agree(self):
        csv_orders = pd.read_csv(CLEAN / "orders.csv")
        with sqlite3.connect(DB_PATH) as con:
            db_orders = pd.read_sql_query("SELECT * FROM orders", con)
        self.assertEqual(len(csv_orders), len(db_orders))
        self.assertAlmostEqual(csv_orders["order_amount"].sum(), db_orders["order_amount"].sum(), places=2)


class SqlQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = analytics.run_queries()
        cls.summary = analytics.summarise_queries(cls.results)

    def test_seven_queries_exist(self):
        self.assertEqual(sorted(self.results), [f"Q{i}" for i in range(1, 8)])

    def test_counts(self):
        s = self.summary
        self.assertEqual((s["Q1_customers"], s["Q2_rows"], s["Q3_products"], s["Q4_customers"], s["Q5_customers"], s["Q7_rows"]),
                         (30, 30, 11, 25, 550, 183))

    def test_money(self):
        s = self.summary
        self.assertEqual(s["Q2_total_inr"], 2998668.80)
        self.assertEqual(s["Q5_average_inr"], 2552.06)
        self.assertEqual(s["Q7_last_cumulative_inr"], 2998668.80)
        self.assertEqual(s["Q2_largest_cell"], {"city": "Bengaluru", "category": "Electronics", "revenue": 233909.45})

    def test_window_functions(self):
        s = self.summary
        self.assertEqual(s["Q6_rank_differs_from_row_number"], 8)
        self.assertEqual({c: (v["customer_id"], v["total_spend"]) for c, v in s["Q6_top_per_city"].items()}, {
            "Bengaluru": ("C0082", 12856.20), "Chennai": ("C0930", 10151.40), "Delhi": ("C0291", 13742.60),
            "Hyderabad": ("C0743", 12730.05), "Mumbai": ("C0693", 14766.55), "Pune": ("C0426", 10377.10)})


class PandasCrossCheck(unittest.TestCase):
    def test_pivot_matches_sql(self):
        check = analytics.pandas_cross_check(2998668.80)
        self.assertTrue(check["totals_match"])
        city = {row["city"]: row for row in check["per_city"]}
        self.assertEqual((city["Bengaluru"]["total"], city["Bengaluru"]["mean"], city["Bengaluru"]["orders"]), (523643.40, 276.47, 1894))
        self.assertEqual((city["Mumbai"]["total"], city["Mumbai"]["mean"], city["Mumbai"]["orders"]), (479058.50, 297.00, 1613))

    def test_exploratory_statistics(self):
        e = analytics.exploratory_statistics()
        self.assertEqual((e["mean"], e["median"], e["q1"], e["q3"], e["upper_fence"]), (269.91, 135.0, 64.8, 297.0, 645.3))
        self.assertEqual((e["outliers_above"], e["outliers_below"]), (897, 0))
        self.assertAlmostEqual(e["skewness"], 4.61, places=2)
        self.assertEqual(set(e["strongest_pair"]), {"order_amount", "discount_pct"})
        self.assertAlmostEqual(e["strongest_value"], -0.095, places=3)


if __name__ == "__main__":
    unittest.main()
