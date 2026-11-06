"""Part B: RFM segmentation and a leakage-aware lapse model.

Run with ``python -m src.customer_models``. Reads ``data/clean``; writes
``outputs/customer_segments.csv``, ``outputs/lapse_model.joblib``,
``outputs/retention_list.csv`` and ``outputs/metrics/part_b.json``.

Leakage controls: lapse features use only orders dated before the cutoff; segment labels are
never features; the scaler and encoder live inside a Pipeline, so they are fitted on training
rows only, and within each cross-validation fold on that fold's rows.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

from src.config import CLEAN, FINAL_K, LAPSE_CUTOFF, LAPSE_WINDOW_END, OUTPUTS, RANDOM_SEED, SNAPSHOT, save_metrics

logger = logging.getLogger("copilot.models")

RFM_COLUMNS = ["recency", "frequency", "monetary"]
NUMERIC_FEATURES = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate",
                    "avg_delivery_minutes", "age", "tenure_days"]
CATEGORICAL_FEATURES = ["favourite_category", "city", "gender"]
SEGMENT_ACTIONS = {
    "Premium Spenders": "Assign priority support and early access to electronics launches; protect them with a personal win-back offer the moment they go quiet.",
    "Loyal Regulars": "Offer a subscription or basket-builder bundle on staples to raise order value without discounting every order.",
    "Active Occasionals": "Send category-led reminders and a second-order nudge to move them towards a weekly habit.",
    "Dormant Occasionals": "Run a low-cost, time-limited reactivation coupon and stop contact if there is no response.",
}


# --------------------------------------------------------------------------- loading
def load_clean() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the clean customer, order and product tables with parsed dates."""
    customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
    products = pd.read_csv(CLEAN / "products.csv")
    return customers, orders, products


# --------------------------------------------------------------------------- segmentation
def build_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    """Recency, frequency and monetary value per customer, as of the snapshot moment."""
    completed = orders[orders["status"].eq("completed")]
    rfm = completed.groupby("customer_id").agg(last_order=("order_ts", "max"), frequency=("order_id", "nunique"),
                                               monetary=("order_amount", "sum")).reset_index()
    rfm["recency"] = (pd.Timestamp(SNAPSHOT) - rfm["last_order"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm[["customer_id", *RFM_COLUMNS]]


def kmeans_diagnostics(scaled: np.ndarray) -> list[dict]:
    """Inertia and silhouette for K = 2..8."""
    rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(scaled)
        rows.append({"k": k, "inertia": round(float(model.inertia_), 1),
                     "silhouette": round(float(silhouette_score(scaled, model.labels_)), 3)})
    return rows


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Name clusters from their measured profile. Cluster numbers are labels with no ordering."""
    remaining = profile.copy()
    premium = remaining["monetary"].idxmax()
    remaining = remaining.drop(index=premium)
    names = {int(premium): "Premium Spenders"}
    regulars = remaining["frequency"].idxmax()
    remaining = remaining.drop(index=regulars)
    names[int(regulars)] = "Loyal Regulars"
    ordered = remaining.sort_values("recency").index.tolist()
    labels = ["Active Occasionals", "Dormant Occasionals"] + [f"Segment {i}" for i in range(len(ordered))]
    for segment_id, label in zip(ordered, labels):
        names[int(segment_id)] = label
    return names


def segment_customers(orders: pd.DataFrame, k: int = FINAL_K) -> tuple[pd.DataFrame, dict]:
    """Fit the final K-means model, profile and name its clusters, and run PCA."""
    rfm = build_rfm(orders)
    scaled = StandardScaler().fit_transform(rfm[RFM_COLUMNS])
    diagnostics = kmeans_diagnostics(scaled)
    rfm["segment_id"] = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit_predict(scaled)
    profile = rfm.groupby("segment_id")[RFM_COLUMNS].mean().round(2)
    profile["size"] = rfm.groupby("segment_id").size()
    names = name_segments(profile)
    rfm["segment_name"] = rfm["segment_id"].map(names)
    premium_id = int(profile["monetary"].idxmax())
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)
    OUTPUTS.mkdir(exist_ok=True)
    rfm[["customer_id", *RFM_COLUMNS, "segment_id", "segment_name", "is_premium"]].to_csv(
        OUTPUTS / "customer_segments.csv", index=False)
    pca = PCA(n_components=2).fit(scaled)
    ratios = [round(float(r), 3) for r in pca.explained_variance_ratio_]
    profile_rows = profile.assign(segment_name=profile.index.map(names),
                                  share_pct=(profile["size"] / len(rfm) * 100).round(1),
                                  revenue_share_pct=(rfm.groupby("segment_id")["monetary"].sum()
                                                     / rfm["monetary"].sum() * 100).round(1),
                                  action=profile.index.map(names).map(SEGMENT_ACTIONS)).reset_index()
    summary = {
        "rfm": {"customers": len(rfm), "recency_median": float(rfm["recency"].median()),
                "recency_min": int(rfm["recency"].min()), "recency_max": int(rfm["recency"].max()),
                "frequency_mean": round(float(rfm["frequency"].mean()), 2), "frequency_max": int(rfm["frequency"].max()),
                "monetary_mean": round(float(rfm["monetary"].mean()), 2), "monetary_max": round(float(rfm["monetary"].max()), 2)},
        "kmeans": diagnostics, "chosen_k": k, "profile": profile_rows.to_dict("records"),
        "premium_segment_id": premium_id,
        "pca_explained_variance_ratio": ratios, "pca_total": round(sum(ratios), 3),
        "pca_loadings_pc1": dict(zip(RFM_COLUMNS, pca.components_[0].round(3).tolist())),
        "pca_loadings_pc2": dict(zip(RFM_COLUMNS, pca.components_[1].round(3).tolist())),
    }
    return rfm, summary


# --------------------------------------------------------------------------- lapse dataset
def build_lapse_dataset(customers: pd.DataFrame, orders: pd.DataFrame,
                        products: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features from orders before the cutoff, and the September label. Indexed by customer."""
    cutoff, window_end = pd.Timestamp(LAPSE_CUTOFF), pd.Timestamp(LAPSE_WINDOW_END)
    history = orders[orders["order_ts"] < cutoff]
    if not (history["order_ts"] < cutoff).all():
        raise AssertionError("feature window contains an order on or after the cutoff")
    completed = history[history["status"].eq("completed")]

    features = completed.groupby("customer_id").agg(
        last_order=("order_ts", "max"), frequency=("order_id", "nunique"), monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"), avg_delivery_minutes=("delivery_minutes", "mean"))
    features["recency"] = (cutoff - features["last_order"]).dt.days
    features = features.drop(columns="last_order")

    final = history[history["status"].isin(["completed", "cancelled"])]
    features["cancellation_rate"] = final["status"].eq("cancelled").groupby(final["customer_id"]).mean()

    with_category = completed.merge(products[["product_id", "category"]], on="product_id")
    counts = with_category.groupby(["customer_id", "category"]).size().rename("n").reset_index()
    favourite = (counts.sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
                 .drop_duplicates("customer_id").set_index("customer_id")["category"])
    features["favourite_category"] = favourite

    people = customers.set_index("customer_id")
    features = features.join(people[["city", "age", "gender"]])
    features["tenure_days"] = (cutoff - people["signup_date"].reindex(features.index)).dt.days

    september = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] < window_end)
                       & orders["status"].eq("completed")]
    label = (~features.index.isin(september["customer_id"])).astype(int)
    y = pd.Series(label, index=features.index, name="lapsed")
    X = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    if X.isna().any().any():
        raise AssertionError(f"missing feature values: {X.isna().sum()[lambda s: s > 0].to_dict()}")
    return X, y


def make_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode nominal columns with k-1 columns per variable."""
    return ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES)])


def print_frequency_means(learned_on_train: float, full_cohort: float) -> None:
    """Show the mean the scaler learned next to the full-cohort mean; they differ by design."""
    print(f"frequency mean: learned by scaler on training rows = {learned_on_train:.4f}, "
          f"full cohort = {full_cohort:.4f}")


def evaluate_lapsed(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Lapsed-class metrics on the given rows, plus threshold-free ranking quality."""
    predicted = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    return {
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, predicted).ravel().tolist(),
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 3),
        "recall": round(float(recall_score(y, predicted, zero_division=0)), 3),
        "f1": round(float(f1_score(y, predicted, zero_division=0)), 3),
        "accuracy": round(float(accuracy_score(y, predicted)), 3),
        "roc_auc": round(float(roc_auc_score(y, probability)), 3),
        "pr_auc": round(float(average_precision_score(y, probability)), 3),
    }


def tune(estimator, grid: dict, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Grid-search a preprocessing + model Pipeline with 5-fold F1 on the training rows only."""
    pipeline = Pipeline([("prep", make_preprocessor()), ("model", estimator)])
    search = GridSearchCV(pipeline, {f"model__{key}": values for key, values in grid.items()}, cv=5, scoring="f1")
    return search.fit(X_train, y_train)


def oversample_training(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Keep every training row and add duplicates of lapsed rows until the classes are equal."""
    lapsed_rows = y_train[y_train.eq(1)].index
    extra_needed = int(y_train.eq(0).sum() - y_train.eq(1).sum())
    extra = resample(lapsed_rows, replace=True, n_samples=extra_needed, random_state=RANDOM_SEED)
    return pd.concat([X_train, X_train.loc[extra]]), pd.concat([y_train, y_train.loc[extra]])


def clean_feature_names(pipeline: Pipeline) -> list[str]:
    return [n.split("__", 1)[1] for n in pipeline.named_steps["prep"].get_feature_names_out()]


def segment_lapse_analysis(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, model: Pipeline,
                           rfm: pd.DataFrame) -> list[dict]:
    """Observed lapse rate per segment, and the model's mean risk on test rows of that segment.

    Descriptive only: segments come from whole-window RFM, which includes September.
    """
    names = rfm.set_index("customer_id")["segment_name"]
    frame = X.assign(segment=names.reindex(X.index), lapsed=y)
    test_risk = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    rows = []
    for segment, group in frame.groupby("segment"):
        in_test = test_risk.reindex(group.index).dropna()
        favourite = group["favourite_category"].value_counts(normalize=True)
        rows.append({"segment_name": segment, "cohort_customers": len(group),
                     "lapse_rate_pct": round(float(group["lapsed"].mean() * 100), 1),
                     "lapsed_customers": int(group["lapsed"].sum()),
                     "top_favourite_category": favourite.index[0],
                     "top_favourite_category_pct": round(float(favourite.iloc[0] * 100), 1),
                     "mean_cancellation_rate_pct": round(float(group["cancellation_rate"].mean() * 100), 1),
                     "test_customers": len(in_test),
                     "mean_predicted_risk_test": round(float(in_test.mean()), 3) if len(in_test) else None})
    return rows


# --------------------------------------------------------------------------- lapse modelling
def build_lapse_model(customers: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame,
                      rfm: pd.DataFrame) -> dict:
    """Tune, evaluate, interpret and persist the lapse model, and write the retention list."""
    X, y = build_lapse_dataset(customers, orders, products)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    baseline = np.zeros(len(y_test), dtype=int)
    baseline_metrics = {"accuracy": round(float(accuracy_score(y_test, baseline)), 3),
                        "f1": float(f1_score(y_test, baseline, zero_division=0)),
                        "precision": float(precision_score(y_test, baseline, zero_division=0)),
                        "recall": float(recall_score(y_test, baseline, zero_division=0))}

    # Tuning touches training rows only; the test set is scored after both searches finish.
    logistic = tune(LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10, 100]}, X_train, y_train)
    forest = tune(RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
                  {"max_depth": [3, 5, 8], "min_samples_leaf": [5, 10, 20]}, X_train, y_train)
    logistic_best, forest_best = logistic.best_estimator_, forest.best_estimator_
    logistic_test, forest_test = evaluate_lapsed(logistic_best, X_test, y_test), evaluate_lapsed(forest_best, X_test, y_test)

    fitted_scaler = forest_best.named_steps["prep"].named_transformers_["num"]
    frequency_position = NUMERIC_FEATURES.index("frequency")
    print_frequency_means(float(fitted_scaler.mean_[frequency_position]), float(X["frequency"].mean()))

    # Class-balance experiment: resample training rows only, score on the unchanged test set.
    X_balanced, y_balanced = oversample_training(X_train, y_train)
    balanced = Pipeline([("prep", make_preprocessor()),
                         ("model", LogisticRegression(max_iter=1000, C=logistic.best_params_["model__C"]))])
    balanced.fit(X_balanced, y_balanced)
    balanced_test = evaluate_lapsed(balanced, X_test, y_test)

    # Interpretation.
    importance = pd.Series(forest_best.named_steps["model"].feature_importances_, index=clean_feature_names(forest_best))
    coefficients = pd.Series(logistic_best.named_steps["model"].coef_[0], index=clean_feature_names(logistic_best))

    # Selection uses cross-validated F1, so the test set does not pick the winner.
    selected_name = "random_forest" if forest.best_score_ >= logistic.best_score_ else "logistic_regression"
    selected = forest_best if selected_name == "random_forest" else logistic_best

    # Persistence round trip.
    path = OUTPUTS / "lapse_model.joblib"
    joblib.dump(selected, path)
    reloaded = joblib.load(path)
    predictions_identical = bool(np.array_equal(selected.predict(X_test), reloaded.predict(X_test)))
    if not predictions_identical:
        raise AssertionError("reloaded model does not reproduce the test predictions")

    # Retention list (illustrative: Premium is a post-hoc whole-window label).
    premium_ids = set(rfm.loc[rfm["is_premium"], "customer_id"])
    risk = pd.DataFrame({"customer_id": X_test.index,
                         "lapse_probability": selected.predict_proba(X_test)[:, 1].round(4),
                         "actually_lapsed": y_test.to_numpy()})
    premium_test = risk[risk["customer_id"].isin(premium_ids)]
    retention = premium_test.sort_values(["lapse_probability", "customer_id"], ascending=[False, True]).head(10)
    retention.to_csv(OUTPUTS / "retention_list.csv", index=False)

    # Error analysis on the selected model.
    predicted = selected.predict(X_test)
    errors = X_test.assign(lapsed=y_test, predicted=predicted)
    error_profile = {
        "false_negative_mean_recency": round(float(errors.query("lapsed==1 and predicted==0")["recency"].mean()), 1),
        "false_negative_mean_frequency": round(float(errors.query("lapsed==1 and predicted==0")["frequency"].mean()), 1),
        "true_positive_mean_recency": round(float(errors.query("lapsed==1 and predicted==1")["recency"].mean()), 1),
        "true_positive_mean_frequency": round(float(errors.query("lapsed==1 and predicted==1")["frequency"].mean()), 1),
    }

    return {
        "cohort": {"customers": len(X), "lapsed": int(y.sum()), "lapsed_pct": round(float(y.mean() * 100), 2),
                   "train_rows": len(X_train), "test_rows": len(X_test),
                   "train_lapsed": int(y_train.sum()), "test_lapsed": int(y_test.sum()),
                   "latest_feature_order": str(orders[orders["order_ts"] < pd.Timestamp(LAPSE_CUTOFF)]["order_ts"].max())},
        "baseline_always_not_lapsed": baseline_metrics,
        "preprocessing": {"frequency_mean_learned_by_scaler_on_train": round(float(fitted_scaler.mean_[frequency_position]), 4),
                          "frequency_mean_full_cohort": round(float(X["frequency"].mean()), 4),
                          "missing_feature_values": int(X.isna().sum().sum()),
                          "encoded_feature_count": len(clean_feature_names(forest_best))},
        "logistic_regression": {"best_params": logistic.best_params_, "best_cv_f1": round(float(logistic.best_score_), 3),
                                "fits": len(logistic.cv_results_["params"]) * 5, "test": logistic_test},
        "random_forest": {"best_params": forest.best_params_, "best_cv_f1": round(float(forest.best_score_), 3),
                          "fits": len(forest.cv_results_["params"]) * 5, "test": forest_test},
        "selected_model": selected_name,
        "class_balance": {"balanced_class_counts": y_balanced.value_counts().to_dict(),
                          "unbalanced": logistic_test, "balanced": balanced_test},
        "top_rf_features": {k: round(float(v), 3) for k, v in importance.sort_values(ascending=False).head(5).items()},
        "logistic_coefficients": {k: round(float(v), 3) for k, v in coefficients.sort_values(key=np.abs, ascending=False).items()},
        "reloaded_predictions_identical": predictions_identical,
        "retention_list": {"premium_customers_in_test": len(premium_test),
                           "premium_test_truly_lapsed": int(premium_test["actually_lapsed"].sum()),
                           "listed": len(retention),
                           "truly_lapsed": int(retention["actually_lapsed"].sum()),
                           "rows": retention.to_dict("records")},
        "error_analysis": error_profile,
        "segment_lapse": segment_lapse_analysis(X, y, X_test, selected, rfm),
    }


def run() -> dict:
    """Run segmentation and lapse modelling and save all metrics."""
    customers, orders, products = load_clean()
    rfm, segment_summary = segment_customers(orders)
    lapse_summary = build_lapse_model(customers, orders, products, rfm)
    payload = {"segments": segment_summary, "lapse": lapse_summary}
    save_metrics("part_b", payload)
    logger.info("Part B finished: selected=%s", lapse_summary["selected_model"])
    return payload


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging()
    run()
