"""Part B: RFM, segments, leakage controls and persisted artefacts."""
import contextlib
import io
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import customer_models as cm
from src.config import METRICS, OUTPUTS


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.customers, cls.orders, cls.products = cm.load_clean()
        cls.metrics = json.loads((METRICS / "part_b.json").read_text(encoding="utf-8"))
        cls.segments = pd.read_csv(OUTPUTS / "customer_segments.csv")


class Rfm(Base):
    def test_reference_statistics(self):
        rfm = cm.build_rfm(self.orders)
        self.assertEqual(len(rfm), 1175)
        self.assertEqual((rfm["recency"].median(), rfm["recency"].min(), rfm["recency"].max()), (19, 0, 180))
        self.assertAlmostEqual(rfm["frequency"].mean(), 9.46, places=2)
        self.assertEqual(rfm["frequency"].max(), 33)
        self.assertAlmostEqual(rfm["monetary"].mean(), 2552.06, places=2)
        self.assertAlmostEqual(rfm["monetary"].max(), 14766.55, places=2)


class Segments(Base):
    REFERENCE = {2: (1820.8, 0.465), 3: (1268.2, 0.445), 4: (743.1, 0.526), 5: (617.4, 0.481),
                 6: (503.2, 0.482), 7: (408.5, 0.416), 8: (366.5, 0.412)}

    def test_kmeans_table_within_tolerance(self):
        table = self.metrics["segments"]["kmeans"]
        self.assertEqual([row["k"] for row in table], list(range(2, 9)))
        for row in table:
            inertia, silhouette = self.REFERENCE[row["k"]]
            self.assertLessEqual(abs(row["inertia"] - inertia) / inertia, 0.05)
            self.assertLessEqual(abs(row["silhouette"] - silhouette), 0.05)

    def test_profile_sizes(self):
        self.assertEqual(sorted(self.segments["segment_id"].value_counts().tolist()), [138, 182, 419, 436])

    def test_premium_flag_is_exactly_the_highest_monetary_segment(self):
        means = self.segments.groupby("segment_id")["monetary"].mean()
        flagged = self.segments.loc[self.segments["is_premium"], "segment_id"].unique().tolist()
        self.assertEqual(flagged, [int(means.idxmax())])

    def test_output_columns(self):
        self.assertEqual(self.segments.columns.tolist(),
                         ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"])

    def test_pca(self):
        first, second = self.metrics["segments"]["pca_explained_variance_ratio"]
        self.assertLessEqual(abs(first - 0.634), 0.02)
        self.assertLessEqual(abs(second - 0.218), 0.02)


class LapseData(Base):
    def test_cohort_and_split(self):
        X, y = cm.build_lapse_dataset(self.customers, self.orders, self.products)
        self.assertEqual((len(X), int(y.sum())), (1156, 443))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        self.assertEqual((len(X_train), len(X_test)), (924, 232))
        self.assertAlmostEqual(float((y_test == 0).mean()), 0.616, places=3)

    def test_no_missing_features(self):
        X, _ = cm.build_lapse_dataset(self.customers, self.orders, self.products)
        self.assertEqual(int(X.isna().sum().sum()), 0)

    def test_september_orders_cannot_change_any_feature(self):
        """Corrupting every order on or after the cutoff must leave the features untouched."""
        X, _ = cm.build_lapse_dataset(self.customers, self.orders, self.products)
        tampered = self.orders.copy()
        later = tampered["order_ts"] >= pd.Timestamp("2025-09-01")
        tampered.loc[later, "order_amount"] = 999999.0
        tampered.loc[later, "discount_pct"] = 99
        tampered.loc[later, "status"] = "cancelled"
        X_after, _ = cm.build_lapse_dataset(self.customers, tampered, self.products)
        pd.testing.assert_frame_equal(X, X_after)

    def test_segment_labels_are_not_features(self):
        for name in cm.NUMERIC_FEATURES + cm.CATEGORICAL_FEATURES:
            self.assertNotIn("segment", name)
            self.assertNotIn("premium", name)

    def test_label_definition(self):
        _, y = cm.build_lapse_dataset(self.customers, self.orders, self.products)
        september = self.orders[(self.orders["order_ts"] >= "2025-09-01") & self.orders["status"].eq("completed")]
        active = set(september["customer_id"])
        self.assertTrue(all((customer not in active) == bool(value) for customer, value in y.items()))


class Preprocessing(Base):
    def test_scaler_sees_training_rows_only(self):
        X, y = cm.build_lapse_dataset(self.customers, self.orders, self.products)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        prep = cm.make_preprocessor().fit(X_train, y_train)
        position = cm.NUMERIC_FEATURES.index("frequency")
        self.assertAlmostEqual(prep.named_transformers_["num"].mean_[position], X_train["frequency"].mean())
        self.assertNotAlmostEqual(prep.named_transformers_["num"].mean_[position], X["frequency"].mean())

    def test_frequency_means_are_printed_side_by_side(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cm.print_frequency_means(8.11474, 8.02851)
        self.assertEqual(output.getvalue().strip(),
                         "frequency mean: learned by scaler on training rows = 8.1147, full cohort = 8.0285")

    def test_encoder_drops_one_level_per_variable(self):
        X, y = cm.build_lapse_dataset(self.customers, self.orders, self.products)
        encoder = cm.make_preprocessor().fit(X, y).named_transformers_["cat"]
        expected = sum(X[c].nunique() - 1 for c in cm.CATEGORICAL_FEATURES)
        self.assertEqual(len(encoder.get_feature_names_out()), expected)

    def test_oversampling_keeps_rows_and_balances_classes(self):
        X = pd.DataFrame({"a": range(10)})
        y = pd.Series([1, 1, 1] + [0] * 7)
        X_new, y_new = cm.oversample_training(X, y)
        self.assertEqual((int(y_new.eq(0).sum()), int(y_new.eq(1).sum())), (7, 7))
        self.assertTrue(set(X.index) <= set(X_new.index))
        self.assertEqual(set(X_new.index[10:]) - {0, 1, 2}, set())


class Models(Base):
    def test_grid_sizes(self):
        self.assertEqual(self.metrics["lapse"]["random_forest"]["fits"], 45)
        self.assertEqual(self.metrics["lapse"]["logistic_regression"]["fits"], 25)

    def test_selected_model_clears_the_bar(self):
        lapse = self.metrics["lapse"]
        self.assertGreaterEqual(lapse[lapse["selected_model"]]["test"]["f1"], 0.55)

    def test_balance_experiment(self):
        balance = self.metrics["lapse"]["class_balance"]
        self.assertEqual(set(balance["balanced_class_counts"].values()), {570})
        self.assertGreater(balance["balanced"]["recall"], balance["unbalanced"]["recall"])
        self.assertLessEqual(balance["balanced"]["precision"], balance["unbalanced"]["precision"])

    def test_saved_model_reproduces_predictions(self):
        X, y = cm.build_lapse_dataset(self.customers, self.orders, self.products)
        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        predictions = joblib.load(OUTPUTS / "lapse_model.joblib").predict(X_test)
        self.assertTrue(self.metrics["lapse"]["reloaded_predictions_identical"])
        self.assertEqual(len(predictions), 232)
        self.assertTrue(np.isin(predictions, [0, 1]).all())

    def test_retention_list(self):
        retention = pd.read_csv(OUTPUTS / "retention_list.csv")
        self.assertEqual(retention.columns.tolist(), ["customer_id", "lapse_probability", "actually_lapsed"])
        self.assertLessEqual(len(retention), 10)
        premium = set(self.segments.loc[self.segments["is_premium"], "customer_id"])
        self.assertTrue(set(retention["customer_id"]) <= premium)
        self.assertTrue(retention["lapse_probability"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
