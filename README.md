# Overview

**Instamart Customer Intelligence & Support Copilot** is an internal tool for a quick-commerce Customer Experience team. It turns messy order exports into a trusted database, segments customers and predicts who will lapse, triages support tickets safely, answers policy questions from documents, and decides refund requests with a bounded agent. All data is synthetic and fictional; the brand name only frames the scenario. All monetary values are in Indian Rupees (INR).

**Problem.** Leadership does not trust the raw exports, the retention team cannot see who is about to stop ordering, and the support desk is overloaded with free-text tickets. Each Part answers one of these, and each consumes what the earlier Parts produced.

```text
generate_data.py -> data/raw, data/policies
  Part A  src/data_pipeline.py, src/analytics.py  -> data/clean, data/instamart.db
  Part B  src/customer_models.py                  -> outputs/customer_segments.csv, lapse model
  Part C  src/triage.py        (reads segments)   -> outputs/triage_results.csv
  Part D  src/policy_rag.py    (reads policies)   -> retrieval and grounded answers
          src/refund_agent.py  (reads the database, reuses retrieval) -> tools, ledger, ReAct loop, resume
mock_llm.py: the single gateway for every model call in Parts C and D
```

**Stack.** Python 3.11 or newer, SQLite, pandas, numpy, scikit-learn and joblib. Tests use the standard-library `unittest`. No charts or images are committed; findings are reported in numbers and words.

**Repository map.**

| Path | Purpose |
|---|---|
| `generate_data.py` | Supplied deterministic data generator, unmodified |
| `mock_llm.py` | Supplied offline language-model stand-in, unmodified |
| `run_project.py` | Runs Parts A to D in order, or one Part with `--part` |
| `src/config.py` | Paths, seeds, logging and metrics helpers |
| `src/data_pipeline.py`, `src/analytics.py`, `src/validate_checkpoints.py` | Part A |
| `src/customer_models.py` | Part B |
| `src/triage.py` | Part C |
| `src/policy_rag.py`, `src/refund_agent.py` | Part D |
| `sql/queries.sql` | The seven analytical queries |
| `notebooks/` | Walk-through that calls the same modules |
| `tests/` | Unit, integration and result checks |
| `outputs/metrics/` | Every figure reported below, as computed JSON |

**Language-model layer.** Every model call goes through the supplied `mock_llm.py`, which is deterministic and rule-based. It is not a real language model and it needs no key, account or network. No hosted model is used.

**Supplied files.** `generate_data.py` and `mock_llm.py` are the brief's versions, unmodified, and a test checks their SHA-256 hashes. The generator writes every raw file, including `data/raw/agent_scenarios.csv`, and the five policy documents.

# Reproduce

Run from the repository root. Python 3.11 or newer is required. Commands are for Windows PowerShell; step 2 shows the Linux and macOS form.

1. `python -m venv .venv`
2. `.venv\Scripts\Activate.ps1` (Linux or macOS: `source .venv/bin/activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python run_project.py`
6. `python -m src.validate_checkpoints`
7. `python -m unittest discover -s tests -t .`

Step 4 ends with `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. Step 6 prints `ALL CHECKPOINTS PASSED`. Step 7 expects the outputs of step 5.

If PowerShell blocks step 2, run `Set-ExecutionPolicy -Scope Process RemoteSigned` first. To rerun one Part: `python run_project.py --part c`. Seeds are fixed wherever randomness appears, so reruns give the same results apart from ledger timestamps.

No environment variables, keys or network access are needed.

# Part A

## Raw data audit

Shapes: `customers.csv` 1,200 x 5, `products.csv` 40 x 4, `orders.csv` 13,084 x 9.

| Table | Column | Data type | Missing | Missing % |
|---|---|---|---:|---:|
| customers | `customer_id` | text | 0 | 0.00 |
| customers | `city` | text | 0 | 0.00 |
| customers | `signup_date` | text | 0 | 0.00 |
| customers | `age` | float | 61 | 5.08 |
| customers | `gender` | text | 30 | 2.50 |
| products | `product_id` | text | 0 | 0.00 |
| products | `product_name` | text | 0 | 0.00 |
| products | `category` | text | 0 | 0.00 |
| products | `price_inr` | integer | 0 | 0.00 |
| orders | `order_id` | text | 0 | 0.00 |
| orders | `customer_id` | text | 0 | 0.00 |
| orders | `product_id` | text | 0 | 0.00 |
| orders | `quantity` | integer | 0 | 0.00 |
| orders | `order_ts` | text | 0 | 0.00 |
| orders | `status` | text | 0 | 0.00 |
| orders | `discount_pct` | integer | 0 | 0.00 |
| orders | `order_amount` | text | 53 | 0.41 |
| orders | `delivery_minutes` | float | 2,541 | 19.42 |

The 53 missing amounts are the `N/A` entries that pandas parses as missing by default. Another 67 entries contain the word `error`, which is why the column stays text and why the non-numeric count below is 120.

| Audit figure | Result | Reference |
|---|---:|---:|
| Raw order rows | 13,084 | 13,084 |
| Exact duplicate rows | 119 | 119 |
| Rows with status `pending` | 656 | 656 |
| Distinct `order_id` | 12,309 | 12,309 |
| Non-numeric `order_amount` | 120 | 120 |
| Negative `delivery_minutes` | 106 | 106 |
| Blank `delivery_minutes` | 2,541 | 2,541 |
| Blank `age` | 61 | 61 |
| `age` outside 10 to 100 | 6 | 6 |
| Raw city spellings | 24 | 24 |
| Raw non-blank gender spellings | 10 | 10 |
| Blank `gender` | 30 | 30 |

## Cleaning rules

Rule order matters and is fixed in `src/data_pipeline.py`.

| Table | Rule | Rows or values changed |
|---|---|---:|
| customers | Trim spaces and apply title case to `city`, leaving six cities | 476 |
| customers | Map gender spellings to male, female or unknown (blank becomes unknown) | 950, including 30 blanks |
| customers | Treat ages outside 10 to 100 as missing | 6 |
| customers | Fill missing ages with the median of the customer's city (`groupby` and `transform`) | 67 |
| customers | Convert `signup_date` to a datetime | 1,200 |
| orders | 1. Convert `order_ts` to a datetime | 13,084 |
| orders | 2. Remove exact duplicate rows | 119 removed |
| orders | 3. Keep the record with the latest `order_ts` for each `order_id` | 656 removed |
| orders | 4. Convert `order_amount` to numeric; refill unparseable values as round(price x quantity x (1 - discount / 100), 2) | 112 |
| orders | 5a. Set negative `delivery_minutes` to missing | 105 |
| orders | 5b. Fill missing delivery time on completed orders with the product category median among completed orders (`groupby` and `transform`) | 774 |
| orders | 5c. Leave delivery time missing on cancelled orders, because they were never delivered | 1,199 left missing |

The counts for rules 4 and 5a are below the audit figures (120 and 106) because some raw bad values sat on duplicate rows removed by rules 2 and 3. City medians can end in .5, so 21 customers have a non-integer age; the ages are not rounded, because the rule asks for the median.

**After cleaning (all match the references).** Cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Genders: female 610, male 560, unknown 30. Ages: 0 missing, all between 18 and 60. Orders: 12,309 rows, one per `order_id`; 11,110 completed and 1,199 cancelled; no pending rows; 0 missing amounts; 0 missing delivery times on completed orders and 1,199 on cancelled orders. Completed revenue: INR 2,998,668.80.

`src/validate_checkpoints.py` asserts 19 checkpoints against the stored CSV files and database, then prints `ALL CHECKPOINTS PASSED`.

## SQL analysis

Queries are in `sql/queries.sql` and run through `pd.read_sql_query`.

| Query | Pattern | Result | Reference |
|---|---|---|---|
| Q1 | WHERE then HAVING | 30 customers | 30 |
| Q2 | Three-table join | 30 rows, total INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 | same |
| Q3 | Scalar subquery | 11 products | 11 |
| Q4 | LEFT JOIN | 25 customers | 25 |
| Q5 | Two CTEs | 550 customers; average INR 2,552.06 | same |
| Q6 | Ranking windows | See below; 8 rows have RANK() different from ROW_NUMBER() | 8 |
| Q7 | Running total | 183 days; last cumulative value INR 2,998,668.80 | same |

Q6 top customer per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10).

RANK() gives customers with equal spend the same rank while ROW_NUMBER() always counts upward, so each of the 8 rows that shares its total spend with a higher-listed customer in the same city gets a rank below its row number.

## pandas cross-check

`pd.pivot_table(..., margins=True)` gives a grand total of INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Completed orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

## Exploratory analysis of completed `order_amount`

| Mean | Median | Skewness | Q1 | Q3 | Lower fence | Upper fence | Outliers above | Outliers below |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 269.91 | 135.00 | 4.61 | 64.80 | 297.00 | -283.50 | 645.30 | 897 | 0 |

<!-- eda-start -->
**What is this data about?** Six months of synthetic quick-commerce orders (April to September 2025) across six cities, with catalogue, discount, status and delivery details.

**What do its distributions and statistics look like?** Order value is strongly right-skewed: the mean (INR 269.91) is about double the median (INR 135.00), and skewness is 4.61. Most baskets are small, and a few are large.

**Are there outliers or anomalies?** 897 orders exceed the INR 645.30 upper fence and none fall below the lower fence. These are mostly genuine electronics purchases, so they were kept. The true anomalies (negative durations, unparseable amounts, impossible ages) were handled during cleaning.

**What relationships exist between variables?** All correlations are weak. The largest in absolute value is `order_amount` with `discount_pct` at -0.095; `order_amount` with `quantity` is +0.087, slightly smaller. Delivery time is essentially unrelated to the others. Correlation of this size does not establish causation, and neither pair should drive a pricing decision alone.
<!-- eda-end -->

# Part B

## RFM table

Snapshot 2025-10-01 00:00:00, completed orders only: 1,175 customers. Recency median 19, minimum 0, maximum 180 days. Frequency mean 9.46, maximum 33. Monetary mean INR 2,552.06, maximum INR 14,766.55. All match the references.

## Choosing K

| K | Inertia | Change from previous K | Silhouette |
|---:|---:|---:|---:|
| 2 | 1,820.8 | | 0.465 |
| 3 | 1,268.2 | -30.4% | 0.445 |
| 4 | 743.1 | -41.4% | 0.526 |
| 5 | 617.4 | -16.9% | 0.480 |
| 6 | 503.2 | -18.5% | 0.482 |
| 7 | 408.5 | -18.8% | 0.416 |
| 8 | 366.5 | -10.3% | 0.412 |

**Chosen K = 4.** The elbow is at K=4, the largest drop in inertia, with smaller gains afterwards. K=4 also has the best silhouette, and its four groups map to four distinct campaigns the retention team can run.

Inertia is the sum of squared distances to the nearest centroid, so more centroids can only lower it, and K=8 will always beat K=2 for that reason alone. Taken to the limit, one cluster per customer gives zero inertia and no business use. That is why K has to be chosen from where the improvement flattens, from silhouette and from interpretability, not from the lowest inertia.

## Segments

Cluster numbers are only labels and have no ordering.

| ID | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Share of customers | Share of revenue | Premium |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | Active Occasionals | 436 | 28.86 | 3.77 | 1,057.62 | 37.1% | 15.4% | no |
| 1 | Premium Spenders | 138 | 24.12 | 6.77 | 7,129.70 | 11.7% | 32.8% | **yes** |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | 35.7% | 47.6% | no |
| 3 | Dormant Occasionals | 182 | 109.04 | 2.20 | 685.99 | 15.5% | 4.2% | no |

PCA on the scaled RFM matrix: explained variance ratios 0.634 and 0.218, total 0.852. The first component is mainly high frequency and spend against high recency (loadings -0.582 recency, 0.613 frequency, 0.534 monetary). The second is mainly monetary value (0.817).

**Segment briefing.** Lapse rate is the September lapse rate of each segment's members in the modelling cohort. It is descriptive only, because segments are built from whole-window RFM that includes September. For the same reason, the Dormant rate of 100% is largely true by construction: a customer whose last order is more than 30 days before the snapshot cannot have ordered in September.

| Segment | Characteristics and behaviour | Business value | Lapse risk | Influencing factors | Recommended action | Expected benefit (spend protected, estimate) |
|---|---|---|---|---|---|---|
| Premium Spenders | Few, costly orders; 97.1% favour Electronics | 11.7% of customers bring 32.8% of revenue | 31.9% (44 of 138) | Long gaps between purchases are normal, so low frequency hides risk | Priority support and early access to electronics launches; a personal win-back offer once they go quiet | About INR 7,130 per customer kept; about INR 313,700 across the 44 who lapsed |
| Loyal Regulars | Order about every 9 days; 38.4% favour Dairy | Largest revenue share, 47.6% | 4.8% (20 of 419) | High frequency is the strongest protective signal | Subscription or basket-builder bundles on staples, without blanket discounts | About INR 3,410 per customer kept; about INR 68,200 across the 20 who lapsed |
| Active Occasionals | About 4 small orders in six months, last one about a month ago | 37.1% of customers, 15.4% of revenue | 47.2% (197 of 417) | Low frequency; the habit is not yet formed | Category reminders and a second-order nudge towards a weekly habit | About INR 1,058 per customer kept; about INR 208,400 across the 197 who lapsed |
| Dormant Occasionals | About 2 orders, last one about 109 days ago | 4.2% of revenue | 100% (182 of 182), by construction | High recency and low frequency | A low-cost, time-limited reactivation coupon; stop contact if there is no response | About INR 686 per customer reactivated |

The benefit column is the segment's mean six-month spend, multiplied by the number who lapsed. It sizes the exposure and is not a measured campaign uplift, since no campaign has been tested.

## Lapse problem

Cutoff 2025-09-01. A customer is `lapsed` when they have no completed order from 2025-09-01 to 2025-09-30.

| Item | Result | Reference |
|---|---:|---:|
| Cohort customers | 1,156 | 1,156 |
| Lapsed | 443 (38.32%) | 443 (38.32%) |
| Training / test rows | 924 / 232 | 924 / 232 |
| Baseline accuracy, always "not lapsed" | 61.6% | 61.6% |
| Baseline lapsed-class F1 | 0.000 | 0 |

**Features:** recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, age and tenure (scaled); favourite category, city and gender (one-hot, k-1 columns).

**Leakage controls.** No encoder, scaler or resampler saw test rows: both transforms live inside a scikit-learn Pipeline fitted on training rows only, and within each cross-validation fold on that fold's rows only. No data from 2025-09-01 onwards entered the features; the latest order used is dated 2025-08-31 22:35:45, and a test corrupts every later order and confirms the features do not change. The segment labels from the K-means step were not used as features. The code also asserts that no feature column contains a missing value (0 found).

The pipeline prints both values when it runs: the scaler learned a `frequency` mean of 8.1147 against a full-cohort mean of 8.0285; they differ because the scaler is fitted on the 924 training rows only, a slightly different sample that excludes the test customers.

## Tuning and evaluation

The test set was used only after both searches finished. Metrics are for the lapsed class.

| Model | Best parameters | Fits | Best CV F1 | TN / FP / FN / TP | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | always "not lapsed" | | | 143 / 0 / 89 / 0 | 0.000 | 0.000 | 0.000 | | |
| Logistic regression | C=100 | 25 | 0.651 | 111 / 32 / 31 / 58 | 0.644 | 0.652 | 0.648 | 0.801 | 0.679 |
| Random forest | max_depth=8, min_samples_leaf=10 | 45 | 0.687 | 111 / 32 / 32 / 57 | 0.640 | 0.640 | 0.640 | 0.782 | 0.615 |

The forest's parameters match the reference, and its test F1 is at the reference level. Logistic regression chose C=100 and reached 0.648, against a reference of C=1 and about 0.62; the brief allows this difference because feature construction can vary.

**Selected model: random forest.** It was chosen on cross-validated F1 (0.687 against 0.651), so the test set did not influence the choice. On the test set the two are effectively tied, differing by one customer out of 89 lapsers. Missing a lapsing customer loses their future orders, worth INR 2,552 of six-month spend on average, while wrongly targeting a loyal one costs only a coupon and some margin. Recall therefore matters more than precision here. At recall 0.640 the forest still misses 32 lapsers, so the balanced model below is the better choice if the team prefers reach over offer cost.

**Error analysis.** The customers the forest misses look healthy before the cutoff (mean recency 18.5 days, mean frequency 6.7), while those it catches already look weak (49.8 days, 2.3 orders). The model detects customers already fading and is weak on sudden drop-offs.

## Class-balance experiment

The training set was over-sampled to 570 rows per class by adding duplicates of lapsed training rows (`random_state=42`). The best logistic regression was refitted and scored on the original, unchanged test set.

| Logistic regression | TN / FP / FN / TP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Un-balanced | 111 / 32 / 31 / 58 | 0.644 | 0.652 | 0.648 |
| Balanced | 90 / 53 / 13 / 76 | 0.589 | 0.854 | 0.697 |

Recall moved most, rising by 0.202, while precision fell by 0.055. For the retention team this means reaching 18 more lapsing customers at the cost of 21 more offers to customers who would have stayed. Over-sampling did not add information, it only moved the decision boundary.

## Interpretation

| Random forest feature | Importance | Business reading |
|---|---:|---|
| `monetary` | 0.278 | Spend so far strongly separates customer types |
| `frequency` | 0.277 | Ordering habit is as informative as spend |
| `recency` | 0.111 | A long gap since the last order signals fading |
| `avg_delivery_minutes` | 0.062 | Slower experience carries some signal, far weaker than behaviour |
| `avg_discount_pct` | 0.053 | Discount exposure adds little |

| Logistic regression feature | Coefficient | Business reading |
|---|---:|---|
| `frequency` | -1.698 (negative) | More orders mean less lapse risk: build the habit |
| `favourite_category_Electronics` | -1.137 (negative) | Compared with Dairy, the dropped base level, and with other features held fixed, electronics buyers are less likely to count as lapsed |
| `monetary` | +0.285 (positive) | At equal frequency, higher spend goes with more risk; spend and frequency are related, so read this with care |

**Persistence.** The selected Pipeline is saved to `outputs/lapse_model.joblib`. The reloaded model reproduced the original test predictions exactly, and the code asserts this.

**Retention list.** `outputs/retention_list.csv` holds the top 10 of the 40 Premium customers in the test set, ranked by predicted lapse probability. **1 of the 10 truly lapsed**, although 12 of all 40 did (30%), so within this segment the model's ranking did no better than chance on this small sample. A likely reason, not tested here, is that high spend reads as a sign of health to the forest. It should not be used to rank within this segment without further work. `outputs/retention_list.csv` is illustrative and not a leakage-free evaluation, because the Premium flag is a post-hoc whole-window label that includes the September window defining the lapse label.

# Part C

## Guardrails and redaction

Input guardrails block 4 empty tickets (`skipped_empty`) and 4 long tickets (`flagged_long`, 734 estimated tokens each, limit 400) before any model call, leaving 156 eligible tickets.

| Substitution | Count | Reference |
|---|---:|---:|
| Email to `[EMAIL]` | 55 | 55 |
| Phone to `[PHONE]` | 91 | 91 |
| Order id to `[ORDER_ID]` | 132 | 132 |
| Total | 278 | 278 |
| Matches left after re-applying the patterns | 0 | 0 |

The user message is exactly the redacted text. The pipeline asserts that no pattern matches before each call, and a test spies on every message sent to confirm it.

## Resilience and validation

Each call retries up to 3 times, waiting `0.05 x 2**n x uniform(0.9, 1.15)` seconds before retry n. A full run produces 19 `TransientLLMError`s, and all 19 recover on the first retry. Waits in the final run range from 0.0453 to 0.0562 seconds, inside the allowed 0.045 to 0.0575. The final run's log, one line per retry, is in `outputs/triage_retry_log.txt`. If retries ever ran out, the ticket would go to a human with status `flagged_llm_error`; this did not happen in any run.

| Status (156 eligible tickets) | Count | Reference |
|---|---:|---:|
| `classified_json` | 135 | 135 |
| `classified_keyword` | 12 | 12 |
| `flagged_invalid_output` | 9 | 9 |

## Zero-shot versus few-shot

| Run | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 147 | 126 | 85.7% | 8,308 | 56.52 |
| Few-shot | 147 | 147 | 100.0% | 20,944 | 142.48 |

Zero-shot confusion table (rows are the true category, columns the predicted one):

| True \ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

The few-shot prompt costs 85.96 more tokens per classified ticket (the prompt itself is 106 estimated tokens against 25), or 12,636 extra tokens per run. That buys 21 corrected tickets, about 602 tokens per correction. The extra cost is worth paying here because 11 of those corrections are fraud tickets that would otherwise sit in a billing queue instead of reaching the security team within 2 hours. Zero-shot would be reasonable only for low-risk bulk sorting, where a wrong queue is cheap to fix.

## Budget, routing and output

**Budget demonstration:** with `TOKEN_BUDGET = 6000`, zero-shot, 43 tickets end `budget_exhausted` and go to human review (`outputs/triage_budget_demo.csv`). The budget is checked before each call, so spending can pass it by at most one call; this run spent 6,012 tokens.

**Final run** (`outputs/triage_results.csv`, 164 rows, few-shot, `TOKEN_BUDGET = 100000`):

| Route | Priority | Tickets |
|---|---|---:|
| `support_queue` | NORMAL | 102 |
| `security_team` | HIGH | 34 |
| `premium_desk` | HIGH | 11 |
| `human_review` | NORMAL | 17 |

**Rule-validated versus human.** 147 tickets were rule-validated: 135 passed the JSON allow-list check, and 12 passed the single-category keyword check. 17 went to humans: 4 empty, 4 long and 9 invalid outputs. Nothing was guessed. Validation only proves a reply is well-formed and on the allowed list, not that it is correct; the accuracy above comes from comparing with `true_category`, which would not exist in production.

# Part D

## Policy assistant

The five documents are split into 40-word chunks with a 10-word overlap, giving 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. For `refund policy late delivery` the top three chunks all come from `refund_policy.txt`.

TF measures how often a word appears in a chunk, so a chunk that repeats a query word scores higher. IDF measures how rare the word is across all chunks, so distinctive words such as `wallet` count for more than common ones such as `order`. One limitation is that TF-IDF matches exact words only: "money back" and "refund" mean the same thing but share no vocabulary, so such a query can miss the right chunk.

| Evaluation | Expected doc in top 3 (of 12) | Answer contains fact (of 12) | Fallback on unanswerable (of 2) |
|---|---:|---:|---:|
| 40 words, overlap 10 (reference 12, 12, 2) | 12 | 12 | 2 |
| 40 words, overlap 0 | 12 | 10 | 2 |

Zero overlap breaks two questions, because their answers are split across a chunk boundary. The documents are still found, but the sentence holding the fact is cut in two:

- *What delivery time makes a completed order eligible for a full refund?* The answer becomes "receives a full refund of the order amount."; the words "more than 45 minutes" are left in the previous chunk.
- *How soon is money returned when a payment fails but is deducted?* The sentence stops at "returned automatically", and "within 48 hours" falls in the next chunk.

## Refund agent

`src/refund_agent.py` holds the five tools with `TOOLS_SPEC` and `TOOL_MAP`. The refund rules are plain Python, so the model cannot change them. Requests are checked against the specification before they run. The two money tools then check the refund themselves, whatever the model asks: the order must exist and be completed, delivery must exceed 45 minutes, and the amount must equal the order amount. `issue_refund` accepts only refunds up to INR 1000 and `escalate_to_human` only larger ones. A refused request returns an error and writes nothing. A second request for an order already in the ledger writes nothing either, and returns the recorded refund with the run that recorded it. These are the only cases in which the tools do not append a line and return just `refund_inr`. The same file holds `run_agent` and `resume_agent`.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1,012 |
| S2 | REFUND | 135.00 | 5 | 1,012 |
| S3 | ESCALATE | 1999.00 | 5 | 999 |
| S4 | DENIED | 0.00 | 4 | 697 |
| S5 | DENIED | 0.00 | 3 | 406 |
| S6 | NOT_FOUND | 0.00 | 2 | 118 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,138 |

All seven match the expected outcome and refund amount. Run logs are in `outputs/agent_runs.jsonl`, one line per scenario.

## Stopping safely

| Demonstration | Status | Detail |
|---|---|---|
| S1 with `token_limit=200` | FALLBACK_TOKEN_LIMIT | Stopped after 3 model calls and 423 tokens; the requested tool was not run |
| S1 with `time_limit_seconds=0` | FALLBACK_TIMEOUT | Stopped before the first model call |
| S7 with default limits | FALLBACK_MAX_ITERATIONS | All six model calls finished within the token limit |

Every fallback returns the steps completed so far and a human-handoff summary in `final_answer`. These runs are logged in `outputs/agent_demos.jsonl`.

## Crash and resume

S1 raised a simulated crash right after the second tool execution, once its checkpoint was saved. `resume_agent` then continued from `state/demo_S1_crash.json`.

- Before the crash: get_order, search_policy
- After the resume: calculate_refund, issue_refund
- Tools executed across both, in order: get_order, search_policy, calculate_refund, issue_refund

That is four executions with none repeated, and the final status is REFUND with INR 128.25. The demonstration runs after the S1 scenario on the same order, so its `issue_refund` finds the refund already recorded by `run_S1` and returns it without writing a second line. `outputs/refund_ledger.jsonl` therefore holds one line per refunded order: S1, S2 and S3.

## ReAct versus plan-and-execute

<!-- react-start -->
The ReAct agent asks the model before every step, so calls follow the work: 5 for S1, 3 for S5 and 2 for S6. Tokens grow faster than calls because each call resends the whole history, including retrieved policy text: S1 used 1,012 tokens and S6 only 118.

A plan-and-execute design asks once for the whole step list and replans only on error. For S1 that could mean about 2 model calls (plan, then final answer), a clear saving, with the tools run by plain code.

The difference shows with unexpected observations. In S6 the order does not exist, and ReAct stops at once after 2 calls. A planner must treat this as an error and replan, or carry conditional branches in its plan. A cancelled order (S5) is not an error at all, so a fixed plan would go on to calculate a refund unless a guard caught it; ReAct reads the status and denies after 3 calls.

For this refund workflow I would choose ReAct. Each decision depends on the previous observation, runs are short and bounded at 6 calls, and a wrong payout costs far more than a few hundred tokens. Checkpointing after each step also gives a natural recovery point. The S7 loop shows ReAct's main risk, repeating itself without progress, and the limits contain that. A very high-volume, stable workflow could justify moving to a planner later.
<!-- react-end -->

# Limitations and Next Steps

**Limitations**

- The Premium flag is a post-hoc whole-window label that includes the September window defining the lapse label, so `outputs/retention_list.csv` is illustrative and not a leakage-free evaluation. The segment lapse rates are descriptive for the same reason.
- The retention list performs poorly: 1 of 10 listed customers truly lapsed, against 12 of 40 across the Premium test customers. The model should not rank within that segment yet.
- The lapse model uses one six-month window and one cutoff, so it has no validation across time or seasons. Test metrics rest on 232 customers, and the two tuned models differ by one customer.
- All language-model results come from the deterministic, rule-based `mock_llm.py`, so the 100% few-shot accuracy reflects its keyword rules and how the data was generated. It does not predict how a real model would perform.
- Redaction covers only the three stated patterns. Names, addresses and differently formatted phone numbers would pass through.
- TF-IDF matches exact words, with no semantic search and no similarity threshold.
- The ledger is a local audit file and no money moves. Refunds are recorded at most once per order, so a crash between a tool's execution and its checkpoint can repeat the call on resume but not the payout. The duplicate check reads the ledger file, so it is safe for one process at a time, not for concurrent writers.
- All refund requests are treated as inside the 7-day window, as specified. `calculate_refund` keeps the brief's two-argument signature, so it cannot see the order status; the status rule is enforced in the two money tools.

**Next steps**

- Build the Premium flag from pre-cutoff data only, and validate the lapse model over several cutoffs.
- Tune the decision threshold against real offer costs, and test retention offers with a controlled experiment to measure actual uplift.
- Move the ledger into the SQLite database with a unique key on the order, so duplicate protection also holds for concurrent writers.
- Evaluate a hosted model and semantic retrieval on labelled tickets and questions, with broader personal-data detection.
- Add monitoring for data contracts, model drift and token spend.
