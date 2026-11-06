-- Instamart Customer Intelligence: analytical queries against data/instamart.db.
-- Unless stated otherwise, only completed orders count as revenue or spend. Amounts are in INR.
-- Each query starts with a line of the form "-- Qn: title"; src/analytics.py splits on that marker.

-- Q1: WHERE then HAVING - customers whose completed orders above INR 500 total more than INR 8000
SELECT customer_id,
       COUNT(*)                    AS qualifying_orders,
       ROUND(SUM(order_amount), 2) AS total_above_500
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_above_500 DESC;

-- Q2: three-table join - revenue by city and product category
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: scalar subquery - products priced above the average catalogue price
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN - customers who have no completed order
SELECT c.customer_id, c.city, c.signup_date
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: common table expressions - customers whose total spend is above the average total
WITH customer_totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_total AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_totals
)
SELECT t.customer_id,
       ROUND(t.total_spend, 2) AS total_spend,
       ROUND(a.avg_spend, 2)   AS avg_spend
FROM customer_totals t
CROSS JOIN average_total a
WHERE t.total_spend > a.avg_spend
ORDER BY t.total_spend DESC;

-- Q6: ranking window functions - customer spend ranked within each city
WITH customer_totals AS (
    SELECT c.customer_id, c.city, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id,
       city,
       total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rank_in_city,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_in_city,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_number_in_city
FROM customer_totals
ORDER BY city, row_number_in_city;

-- Q7: running total - daily completed revenue with cumulative revenue
WITH daily AS (
    SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT day,
       ROUND(revenue, 2)                            AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2)   AS cumulative_revenue
FROM daily
ORDER BY day;
