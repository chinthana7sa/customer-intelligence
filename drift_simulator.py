# drift_simulator.py  — Tier 2 new file
# ─────────────────────────────────────────────────────────────
# What this does:
#   Runs alongside data_generator.py as a separate process.
#   Every 30 seconds it reads the current customer pool from
#   the database and applies realistic behavioural drift:
#
#   1. VIP COOLING   — high-spend users gradually reduce visit
#                      frequency, simulating real churn onset
#   2. DORMANT REACTIVATION — some quiet users suddenly return,
#                      simulating win-back campaigns working
#   3. BROWSER CONVERSION — heavy browsers occasionally make
#                      their first purchase (HIGH_POTENTIAL → LOYAL)
#   4. SEASONAL SPIKE — random burst of activity every few minutes
#                      simulating a flash sale or promotion
#   5. CHURN ACCELERATION — churner-persona users go fully silent
#                      making their AT_RISK label earn itself
#
#   All drift events are written to SQLite as real user events,
#   so the DP-GCN model sees them during the next retrain cycle.
#   A drift_log table records every drift event for analysis.
# ─────────────────────────────────────────────────────────────

import sqlite3
import time
import random
import logging
from datetime import datetime, timedelta
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DRIFT - %(levelname)s - %(message)s'
)

# ── Config ────────────────────────────────────────────────────
DB_FILE          = "customer_data.db"
DRIFT_INTERVAL   = 30      # seconds between drift cycles
COOLING_RATE     = 0.08    # 8% of VIPs start cooling per cycle
REACTIVATION_RATE = 0.05   # 5% of dormant users reactivate per cycle
CONVERSION_RATE  = 0.03    # 3% of browsers convert per cycle
CHURN_RATE       = 0.10    # 10% of churners go fully silent per cycle
SPIKE_PROB       = 0.15    # 15% chance of a promotional spike per cycle


# ── Product catalogue (mirrors data_generator.py) ─────────────
CATEGORIES = {
    "Electronics": [f"ELEC_{i:03d}" for i in range(1, 31)],
    "Clothing":    [f"CLTH_{i:03d}" for i in range(1, 31)],
    "Books":       [f"BOOK_{i:03d}" for i in range(1, 31)],
    "Food":        [f"FOOD_{i:03d}" for i in range(1, 31)],
    "Sports":      [f"SPRT_{i:03d}" for i in range(1, 31)],
}

PRICE_RANGES = {
    "Electronics": (80.0,  1200.0),
    "Clothing":    (25.0,  350.0),
    "Books":       (10.0,  80.0),
    "Food":        (8.0,   120.0),
    "Sports":      (40.0,  600.0),
}


def setup_drift_log(conn):
    """Create drift_log table for recording all drift events."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drift_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            drift_time   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            drift_type   TEXT NOT NULL,
            user_id      TEXT NOT NULL,
            detail       TEXT
        )
    """)
    conn.commit()


def log_drift(conn, drift_type: str, user_id: str, detail: str = ""):
    """Record a drift event to drift_log table."""
    conn.execute("""
        INSERT INTO drift_log (drift_type, user_id, detail)
        VALUES (?, ?, ?)
    """, (drift_type, user_id, detail))


def get_customer_profiles(conn) -> pd.DataFrame:
    """
    Read current customer profiles from user_events.
    Returns one row per user with spend, activity, and recency.
    """
    df = pd.read_sql_query("""
        SELECT
            user_id,
            SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS total_spent,
            COUNT(CASE WHEN event_type = 'purchase' THEN 1 END)          AS purchase_count,
            COUNT(CASE WHEN event_type = 'view'     THEN 1 END)          AS view_count,
            COUNT(CASE WHEN event_type = 'cart'     THEN 1 END)          AS cart_count,
            MAX(event_time)                                               AS last_event,
            COUNT(*)                                                      AS total_events
        FROM user_events
        GROUP BY user_id
    """, conn)

    if df.empty:
        return df

    df['last_event']    = pd.to_datetime(df['last_event'], format='mixed')
    df['hours_inactive'] = (
        datetime.now() - df['last_event']
    ).dt.total_seconds() / 3600

    return df


def write_drift_events(conn, user_id, events):
    now = datetime.now().isoformat()
    for event_type, product_id, price in events:
        conn.execute("""
            INSERT INTO user_events (user_id, event_type, product_id, price, event_time)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, event_type, product_id, round(price, 2), now))
    for attempt in range(5):
        try:
            conn.commit()
            break
        except sqlite3.OperationalError:
            time.sleep(1)


def apply_vip_cooling(conn, profiles: pd.DataFrame) -> int:
    """
    VIP COOLING — top spenders start reducing activity.
    Simulates real churn onset: fewer views, no purchases.
    These users transition toward DORMANT_VIP over time.
    """
    if profiles.empty:
        return 0

    spend_p80 = profiles['total_spent'].quantile(0.80)
    vips      = profiles[profiles['total_spent'] >= spend_p80]
    cooling   = vips.sample(frac=COOLING_RATE, random_state=random.randint(0, 9999))

    count = 0
    for _, user in cooling.iterrows():
        # Generate 1-2 half-hearted views only — no cart, no purchase
        cat      = random.choice(list(CATEGORIES.keys()))
        products = CATEGORIES[cat]
        events   = [("view", random.choice(products), 0.0)
                    for _ in range(random.randint(1, 2))]
        write_drift_events(conn, user['user_id'], events)
        log_drift(conn, "VIP_COOLING", user['user_id'],
                  f"spend={user['total_spent']:.2f}")
        count += 1

    if count:
        logging.info(f"❄️  VIP Cooling: {count} high-value users reducing activity")
    return count


def apply_dormant_reactivation(conn, profiles: pd.DataFrame) -> int:
    """
    DORMANT REACTIVATION — inactive users suddenly return.
    Simulates win-back email campaigns, seasonal promotions.
    These users generate a burst of activity and sometimes purchase.
    """
    if profiles.empty:
        return 0

    # Dormant = no activity in last 2+ hours
    dormant   = profiles[profiles['hours_inactive'] >= 2.0]
    reactivated = dormant.sample(
        n=min(len(dormant), max(1, int(len(dormant) * REACTIVATION_RATE))),
        random_state=random.randint(0, 9999)
    )

    count = 0
    for _, user in reactivated.iterrows():
        cat      = random.choice(list(CATEGORIES.keys()))
        products = CATEGORIES[cat]
        price_range = PRICE_RANGES[cat]

        # Burst of 3-6 events including cart and maybe a purchase
        events = []
        for _ in range(random.randint(2, 4)):
            events.append(("view", random.choice(products), 0.0))
        events.append(("cart", random.choice(products),
                        round(random.uniform(*price_range) * 0.7, 2)))
        if random.random() < 0.40:   # 40% chance they actually buy
            events.append(("purchase", random.choice(products),
                            round(random.uniform(*price_range), 2)))

        write_drift_events(conn, user['user_id'], events)
        log_drift(conn, "REACTIVATION", user['user_id'],
                  f"inactive_hours={user['hours_inactive']:.1f}")
        count += 1

    if count:
        logging.info(f"♻️  Reactivation: {count} dormant users returned")
    return count


def apply_browser_conversion(conn, profiles: pd.DataFrame) -> int:
    """
    BROWSER CONVERSION — heavy viewers make their first purchase.
    Simulates a HIGH_POTENTIAL customer finally converting.
    Critical for making HIGH_POTENTIAL a meaningful segment.
    """
    if profiles.empty:
        return 0

    # Browsers: many views, zero or minimal purchases
    browsers = profiles[
        (profiles['view_count'] >= 5) &
        (profiles['purchase_count'] == 0)
    ]
    converting = browsers.sample(
        n=min(len(browsers), max(1, int(len(browsers) * CONVERSION_RATE))),
        random_state=random.randint(0, 9999)
    )

    count = 0
    for _, user in converting.iterrows():
        # A decisive session: view → cart → purchase
        cat         = random.choice(list(CATEGORIES.keys()))
        products    = CATEGORIES[cat]
        price_range = PRICE_RANGES[cat]
        price       = round(random.uniform(*price_range), 2)

        events = [
            ("view",     random.choice(products), 0.0),
            ("cart",     random.choice(products), round(price * 0.9, 2)),
            ("purchase", random.choice(products), price),
        ]
        write_drift_events(conn, user['user_id'], events)
        log_drift(conn, "CONVERSION", user['user_id'],
                  f"views={user['view_count']} first_purchase={price:.2f}")
        count += 1

    if count:
        logging.info(f"🎯 Conversion: {count} browsers made their first purchase")
    return count


def apply_churn_acceleration(conn, profiles: pd.DataFrame) -> int:
    """
    CHURN ACCELERATION — at-risk users go fully silent.
    Low-spend, low-activity users stop appearing entirely.
    Makes AT_RISK a genuinely earned segment — not just a default.
    These users will appear with increasing hours_inactive,
    which the GCN picks up via the days_since_last_purchase feature.
    """
    # We don't write any events for churners — their silence IS the signal.
    # But we log who is churning so the drift_log shows the pattern.
    if profiles.empty:
        return 0

    # Churner profile: low spend, low activity, already inactive
    spend_p25   = profiles['total_spent'].quantile(0.25)
    at_risk     = profiles[
        (profiles['total_spent'] <= spend_p25) &
        (profiles['hours_inactive'] >= 1.0)
    ]
    churning    = at_risk.sample(
        n=min(len(at_risk), max(1, int(len(at_risk) * CHURN_RATE))),
        random_state=random.randint(0, 9999)
    )

    count = 0
    for _, user in churning.iterrows():
        log_drift(conn, "CHURN_SILENT", user['user_id'],
                  f"spend={user['total_spent']:.2f} "
                  f"inactive={user['hours_inactive']:.1f}h")
        count += 1

    conn.commit()
    if count:
        logging.info(f"💀 Churn: {count} at-risk users going silent")
    return count


def apply_promotional_spike(conn, profiles: pd.DataFrame) -> int:
    """
    PROMOTIONAL SPIKE — random burst of activity across many users.
    Simulates a flash sale, email campaign, or seasonal event.
    Creates interesting temporal patterns for the GCN to detect.
    """
    if profiles.empty or random.random() > SPIKE_PROB:
        return 0

    # Pick 5-15% of random users for the spike
    spike_users = profiles.sample(
        frac=random.uniform(0.05, 0.15),
        random_state=random.randint(0, 9999)
    )

    # Pick a random spike category (flash sale on one category)
    spike_cat   = random.choice(list(CATEGORIES.keys()))
    products    = CATEGORIES[spike_cat]
    price_range = PRICE_RANGES[spike_cat]
    discount    = random.uniform(0.6, 0.85)   # 15-40% discount

    count = 0
    for _, user in spike_users.iterrows():
        events = [("view", random.choice(products), 0.0)]
        if random.random() < 0.6:
            events.append(("cart", random.choice(products),
                            round(random.uniform(*price_range) * discount, 2)))
        if random.random() < 0.30:
            events.append(("purchase", random.choice(products),
                            round(random.uniform(*price_range) * discount, 2)))
        write_drift_events(conn, user['user_id'], events)
        count += 1

    conn.commit()
    logging.info(
        f"📣 Promo spike: {count} users responded to "
        f"{spike_cat} flash sale ({int((1-discount)*100)}% off)"
    )
    log_drift(conn, "PROMO_SPIKE", "ALL",
              f"cat={spike_cat} users={count} discount={discount:.0%}")
    conn.commit()
    return count


def print_drift_summary(conn):
    """Print a summary of all drift events so far."""
    try:
        df = pd.read_sql_query("""
            SELECT drift_type, COUNT(*) as count
            FROM drift_log
            GROUP BY drift_type
            ORDER BY count DESC
        """, conn)
        if not df.empty:
            logging.info("📊 Drift summary:")
            for _, row in df.iterrows():
                logging.info(f"   {row['drift_type']:20s}: {row['count']}")
    except Exception:
        pass


def main():
    logging.info("🌊 Starting Drift Simulator")
    logging.info(f"   DB file        : {DB_FILE}")
    logging.info(f"   Cycle interval : {DRIFT_INTERVAL}s")
    logging.info(f"   VIP cooling    : {COOLING_RATE*100:.0f}% per cycle")
    logging.info(f"   Reactivation   : {REACTIVATION_RATE*100:.0f}% of dormant")
    logging.info(f"   Conversion     : {CONVERSION_RATE*100:.0f}% of browsers")
    logging.info(f"   Promo spike    : {SPIKE_PROB*100:.0f}% chance per cycle")
    logging.info("   Press Ctrl+C to stop\n")

    cycle = 0

    try:
        while True:
            cycle += 1
            logging.info(f"── Drift cycle {cycle} ──────────────────────────")

            conn = sqlite3.connect(DB_FILE, timeout=30)
            setup_drift_log(conn)

            # Read current customer profiles
            profiles = get_customer_profiles(conn)

            if profiles.empty or len(profiles) < 10:
                logging.info("   ⏳ Waiting for more data (need 10+ customers)...")
                conn.close()
                time.sleep(DRIFT_INTERVAL)
                continue

            logging.info(f"   Customers in DB: {len(profiles)}")

            # Apply all drift types
            n_cool    = apply_vip_cooling(conn, profiles)
            n_react   = apply_dormant_reactivation(conn, profiles)
            n_convert = apply_browser_conversion(conn, profiles)
            n_churn   = apply_churn_acceleration(conn, profiles)
            n_spike   = apply_promotional_spike(conn, profiles)

            total = n_cool + n_react + n_convert + n_churn + n_spike
            logging.info(f"   Total drift events this cycle: {total}")

            # Print summary every 10 cycles
            if cycle % 10 == 0:
                print_drift_summary(conn)

            conn.close()
            time.sleep(DRIFT_INTERVAL)

    except KeyboardInterrupt:
        logging.info(f"\n🛑 Drift simulator stopped after {cycle} cycles")
    except Exception as e:
        logging.error(f"❌ Error in drift cycle {cycle}: {e}")
        raise


if __name__ == "__main__":
    main()
