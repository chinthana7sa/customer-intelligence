# data_generator.py  — Tier 1 complete replacement
# ─────────────────────────────────────────────────────────────
# What this fixes vs the original:
#   CRITICAL — persistent 300-user pool (every event now belongs
#              to a returning customer, not a new stranger UUID)
#   NEW      — 5 product categories with realistic price ranges
#   NEW      — session simulation (users browse 3-8 items before
#              buying, mimicking real shopping behaviour)
#   NEW      — time-of-day purchase weights (peak evening, quiet night)
#   NEW      — behavioural drift (VIPs slowly go cold, dormant
#              users randomly reactivate — makes AT_RISK real)
#   NEW      — customer personas (VIP, Regular, Browser, Churner)
#              so the GCN actually has meaningful patterns to learn
# ─────────────────────────────────────────────────────────────

import sqlite3
import time
import uuid
import random
import logging
import os
from datetime import datetime, timedelta
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ── Config ────────────────────────────────────────────────────
DB_FILE        = "customer_data.db"
USER_POOL_SIZE = 300     # persistent users — THE critical fix
MIN_DELAY      = 0.05     # seconds between events
MAX_DELAY      = 0.2

# ── Product catalogue — 5 categories with price bands ─────────
CATEGORIES = {
    "Electronics": {
        "products": [f"ELEC_{i:03d}" for i in range(1, 31)],
        "view_price":     (0.0,   0.0),
        "cart_price":     (50.0,  800.0),
        "purchase_price": (80.0,  1200.0),
    },
    "Clothing": {
        "products": [f"CLTH_{i:03d}" for i in range(1, 31)],
        "view_price":     (0.0,   0.0),
        "cart_price":     (20.0,  200.0),
        "purchase_price": (25.0,  350.0),
    },
    "Books": {
        "products": [f"BOOK_{i:03d}" for i in range(1, 31)],
        "view_price":     (0.0,   0.0),
        "cart_price":     (8.0,   60.0),
        "purchase_price": (10.0,  80.0),
    },
    "Food": {
        "products": [f"FOOD_{i:03d}" for i in range(1, 31)],
        "view_price":     (0.0,   0.0),
        "cart_price":     (5.0,   80.0),
        "purchase_price": (8.0,   120.0),
    },
    "Sports": {
        "products": [f"SPRT_{i:03d}" for i in range(1, 31)],
        "view_price":     (0.0,   0.0),
        "cart_price":     (30.0,  400.0),
        "purchase_price": (40.0,  600.0),
    },
}

# ── Customer personas — defines behaviour pattern per user ────
# Each user is assigned one persona at pool creation.
# Persona drives event weights and session length.
PERSONAS = {
    "vip": {
        # High spenders, buy often, visit frequently
        "event_weights":    [0.50, 0.25, 0.25],   # view/cart/purchase
        "session_length":   (4, 10),
        "inter_session":    (600, 3600),           # seconds between sessions
        "purchase_prob":    0.40,                  # prob of buying in a session
        "preferred_cats":   ["Electronics", "Clothing", "Sports"],
        "drift_to_dormant": 0.001,                 # tiny chance per event of going cold
    },
    "regular": {
        # Moderate spenders, consistent behaviour
        "event_weights":    [0.65, 0.20, 0.15],
        "session_length":   (3, 7),
        "inter_session":    (3600, 14400),
        "purchase_prob":    0.20,
        "preferred_cats":   ["Clothing", "Books", "Food"],
        "drift_to_dormant": 0.002,
    },
    "browser": {
        # Lots of views, rarely buys — high potential if activated
        "event_weights":    [0.85, 0.12, 0.03],
        "session_length":   (5, 12),
        "inter_session":    (1800, 7200),
        "purchase_prob":    0.05,
        "preferred_cats":   ["Electronics", "Clothing", "Sports"],
        "drift_to_dormant": 0.003,
    },
    "churner": {
        # Was active, now barely visits — AT_RISK segment
        "event_weights":    [0.90, 0.08, 0.02],
        "session_length":   (1, 3),
        "inter_session":    (14400, 86400),
        "purchase_prob":    0.02,
        "preferred_cats":   ["Books", "Food"],
        "drift_to_dormant": 0.010,
    },
}

# Persona distribution across the user pool
PERSONA_WEIGHTS = {
    "vip":     0.10,   # 10%  — 30 VIP users
    "regular": 0.40,   # 40%  — 120 regular users
    "browser": 0.35,   # 35%  — 105 browser users
    "churner": 0.15,   # 15%  — 45 at-risk users
}


class CustomerPool:
    """
    Maintains 300 persistent customers, each with:
    - A stable UUID (never changes)
    - A persona (drives event pattern)
    - A behavioural state (active / dormant)
    - A preferred product category
    """

    def __init__(self, size: int = USER_POOL_SIZE):
        self.size    = size
        self.users   = self._create_pool()
        logging.info(
            f"✅ Customer pool created: {size} users | "
            f"VIP={sum(1 for u in self.users if u['persona']=='vip')} | "
            f"Regular={sum(1 for u in self.users if u['persona']=='regular')} | "
            f"Browser={sum(1 for u in self.users if u['persona']=='browser')} | "
            f"Churner={sum(1 for u in self.users if u['persona']=='churner')}"
        )

    def _create_pool(self):
        personas = list(PERSONA_WEIGHTS.keys())
        weights  = list(PERSONA_WEIGHTS.values())
        users    = []

        for _ in range(self.size):
            persona = random.choices(personas, weights=weights, k=1)[0]
            pref_cats = PERSONAS[persona]["preferred_cats"]
            users.append({
                "user_id":      str(uuid.uuid4()),   # stable forever
                "persona":      persona,
                "state":        "active",            # active | dormant
                "pref_cat":     random.choice(pref_cats),
                "dormant_since": None,
            })

        return users

    def pick_active_user(self):
        """Return a random active user; occasionally reactivate a dormant one."""
        active  = [u for u in self.users if u["state"] == "active"]
        dormant = [u for u in self.users if u["state"] == "dormant"]

        # 3% chance: reactivate a dormant user (simulates win-back)
        if dormant and random.random() < 0.03:
            user = random.choice(dormant)
            user["state"]        = "active"
            user["dormant_since"] = None
            logging.info(f"♻️  User {user['user_id'][:8]} reactivated ({user['persona']})")
            return user

        if not active:
            # safety: if everyone dormant, reactivate random
            user = random.choice(self.users)
            user["state"] = "active"
            return user

        return random.choice(active)

    def maybe_drift_to_dormant(self, user):
        """Small per-event probability that this user goes cold."""
        p = PERSONAS[user["persona"]]["drift_to_dormant"]
        if random.random() < p:
            user["state"]        = "dormant"
            user["dormant_since"] = datetime.now()
            logging.info(
                f"💤 User {user['user_id'][:8]} went dormant ({user['persona']})"
            )


def get_time_of_day_purchase_multiplier() -> float:
    """
    Returns a multiplier (0.2 – 1.5) based on current hour.
    Peak buying: 19:00–22:00 (evening)
    Quiet:       02:00–05:00 (night)
    """
    hour = datetime.now().hour
    if   2  <= hour < 5:   return 0.2   # dead of night
    elif 5  <= hour < 9:   return 0.5   # early morning
    elif 9  <= hour < 12:  return 0.8   # morning
    elif 12 <= hour < 14:  return 1.0   # lunch
    elif 14 <= hour < 17:  return 0.7   # afternoon dip
    elif 17 <= hour < 19:  return 1.1   # after-work
    elif 19 <= hour < 22:  return 1.5   # peak evening
    else:                  return 0.6   # late night


def generate_session(user: dict, cur, conn):
    """
    Simulates one shopping session for a user:
    1. Views 3–8 products (their preferred category + 1-2 random)
    2. Adds 0–3 items to cart
    3. Possibly completes a purchase
    All events written to SQLite in sequence with small delays.
    """
    persona    = PERSONAS[user["persona"]]
    session_len = random.randint(*persona["session_length"])
    category   = user["pref_cat"]
    cat_data   = CATEGORIES[category]

    # Occasionally browse outside preferred category
    if random.random() < 0.25:
        category = random.choice(list(CATEGORIES.keys()))
        cat_data = CATEGORIES[category]

    tod_mult   = get_time_of_day_purchase_multiplier()
    events_in_session = 0

    for step in range(session_len):
        product_id = random.choice(cat_data["products"])

        # Determine event type for this step
        weights = persona["event_weights"].copy()
        # Adjust purchase weight by time-of-day
        weights[2] = weights[2] * tod_mult
        # Normalise
        total = sum(weights)
        weights = [w / total for w in weights]

        event_type = random.choices(
            ["view", "cart", "purchase"],
            weights=weights,
            k=1
        )[0]

        # Price
        price_range = cat_data[f"{event_type}_price"]
        price = round(random.uniform(*price_range), 2)

        # Write event
        cur.execute("""
            INSERT INTO user_events
                (user_id, event_type, product_id, price, event_time)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user["user_id"],
            event_type,
            product_id,
            price,
            datetime.now()
        ))
        conn.commit()
        events_in_session += 1

        # Small intra-session delay (browsing feels real)
        time.sleep(random.uniform(0.01, 0.05))

    return events_in_session


def setup_database():
    """Create the events table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT        NOT NULL,
            event_type  TEXT        NOT NULL,
            product_id  TEXT,
            price       REAL        DEFAULT 0.0,
            event_time  TIMESTAMP   NOT NULL
        )
    """)
    # Index on user_id makes GCN feature extraction much faster
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_id
        ON user_events (user_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_time
        ON user_events (event_time)
    """)
    conn.commit()
    conn.close()
    logging.info("✅ Database ready with indexes")


def main():
    logging.info("🚀 Starting Customer Intelligence Data Generator")
    logging.info(f"   User pool : {USER_POOL_SIZE} persistent customers")
    logging.info(f"   DB file   : {DB_FILE}")
    logging.info("   Press Ctrl+C to stop\n")

    setup_database()
    pool = CustomerPool(USER_POOL_SIZE)

    conn        = sqlite3.connect(DB_FILE)
    cur         = conn.cursor()
    total_events = 0
    session_count = 0

    try:
        while True:
            # Pick an active user
            user = pool.pick_active_user()

            # Run one shopping session for them
            n = generate_session(user, cur, conn)
            total_events  += n
            session_count += 1

            # Maybe this user drifts to dormant after their session
            pool.maybe_drift_to_dormant(user)

            # Log progress every 10 sessions
            if session_count % 10 == 0:
                active_count  = sum(1 for u in pool.users if u["state"] == "active")
                dormant_count = sum(1 for u in pool.users if u["state"] == "dormant")
                logging.info(
                    f"📊 Sessions: {session_count} | "
                    f"Events: {total_events} | "
                    f"Active users: {active_count} | "
                    f"Dormant: {dormant_count}"
                )

            #Inter-session pause (simulates users coming back later)
            #persona_cfg = PERSONAS[user["persona"]]
            #wait = random.uniform(*persona_cfg["inter_session"])
            #Scale down for simulation speed (divide by 60 → minutes become seconds)
            #wait = wait / 60.0
            #wait = max(MIN_DELAY, min(wait, MAX_DELAY))
            #time.sleep(wait)
            
            # Inter-session pause — fast mode for simulation
            time.sleep(random.uniform(0.1, 0.3))
    except KeyboardInterrupt:
        logging.info(
            f"\n🛑 Stopped. "
            f"Sessions: {session_count} | "
            f"Total events: {total_events}"
        )
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
