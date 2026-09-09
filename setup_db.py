# setup_db.py — RisingWave streaming setup (production only)
# Only needed when running with Docker + Kafka + RisingWave
# For development, SQLite is used automatically by data_generator.py

import psycopg2
import time

RISINGWAVE_HOST = "localhost"
RISINGWAVE_PORT = 4566
RISINGWAVE_DB   = "dev"
RISINGWAVE_USER = "root"
KAFKA_BROKER    = "kafka:29092"
KAFKA_TOPIC     = "user_behavior_topic"

def setup_streaming():
    print("Connecting to RisingWave...")
    conn = psycopg2.connect(
        host=RISINGWAVE_HOST, port=RISINGWAVE_PORT,
        database=RISINGWAVE_DB, user=RISINGWAVE_USER
    )
    conn.autocommit = True
    cur = conn.cursor()

    print("Creating Kafka source...")
    cur.execute(f"""
        CREATE SOURCE IF NOT EXISTS user_behavior_source (
            user_id     VARCHAR,
            event_type  VARCHAR,
            product_id  VARCHAR,
            price       DOUBLE PRECISION,
            event_time  TIMESTAMP
        )
        WITH (
            connector = 'kafka',
            topic = '{KAFKA_TOPIC}',
            properties.bootstrap.server = '{KAFKA_BROKER}',
            scan.startup.mode = 'earliest'
        )
        FORMAT PLAIN ENCODE JSON;
    """)

    print("Creating behavioral_scores view...")
    cur.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS behavioral_scores AS
        SELECT
            user_id,
            COUNT(CASE WHEN event_type = 'view' THEN 1 END)     AS views_10min,
            COUNT(CASE WHEN event_type = 'cart' THEN 1 END)     AS carts_10min,
            COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchases_10min,
            SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS spend_10min
        FROM user_behavior_source
        WHERE event_time >= NOW() - INTERVAL '10 minutes'
        GROUP BY user_id;
    """)

    print("Creating historical_rfm view...")
    cur.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS historical_rfm AS
        SELECT
            user_id,
            SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS total_spent,
            COUNT(CASE WHEN event_type = 'purchase' THEN 1 END)           AS purchase_count,
            MAX(CASE WHEN event_type = 'purchase' THEN event_time END)    AS last_purchase_time
        FROM user_behavior_source
        GROUP BY user_id;
    """)

    print("Creating live_customer_segments view...")
    cur.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS live_customer_segments AS
        SELECT
            COALESCE(h.user_id, b.user_id) AS user_id,
            COALESCE(h.total_spent, 0)      AS total_spent,
            COALESCE(h.purchase_count, 0)   AS purchase_count,
            COALESCE(b.views_10min, 0)      AS views_10min,
            COALESCE(b.carts_10min, 0)      AS carts_10min,
            h.last_purchase_time
        FROM historical_rfm h
        FULL OUTER JOIN behavioral_scores b ON h.user_id = b.user_id;
    """)

    print("✅ RisingWave streaming setup complete!")
    cur.close()
    conn.close()

if __name__ == "__main__":
    setup_streaming()