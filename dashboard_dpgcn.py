# dashboard_dpgcn.py — Complete DP-GCN Dashboard
# All fixes applied — clean version

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import random
import threading
import logging
import numpy as np
import streamlit.components.v1 as components

from graph_visualizer import get_graph_html
from explainability import explain_top_customers, render_explanation_html
from action_engine import (generate_actions, get_campaign_summary,
                            to_csv_bytes, render_action_summary_html)
from temporal_gcn import get_temporal_summary
from dpgcn_model import get_dpgcn

st.set_page_config(page_title="DP-GCN Customer Intelligence", layout="wide", page_icon="🚀")

DB_FILE = "customer_data.db"


def get_connection():
    return sqlite3.connect(DB_FILE, timeout=30)


def train_model():
    conn = get_connection()
    df_events = pd.read_sql_query(
        "SELECT * FROM user_events ORDER BY event_time DESC LIMIT 100000",
        conn
    )
    conn.close()
    if len(df_events) <= 50:
        return False, f"Need more than 50 events — currently have {len(df_events)}."
    n_unique = df_events["user_id"].nunique()
    if n_unique < 10:
        return False, f"Need at least 10 unique customers — currently have {n_unique}."
    with st.spinner("🧠 Training DP-GCN on customer relationships..."):
        success = st.session_state.dpgcn.train(df_events, epochs=20)
    if not success:
        return False, "Training did not complete — check the terminal logs for details."
    return True, None


# ── Schema helpers ───────────────────────────────────────────────
def reset_database():
    """Wipe user_events and recreate the schema fresh."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS user_events")
    cur.execute("""
        CREATE TABLE user_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT        NOT NULL,
            event_type  TEXT        NOT NULL,
            product_id  TEXT,
            price       REAL        DEFAULT 0.0,
            event_time  TIMESTAMP   NOT NULL
        )
    """)
    cur.execute("CREATE INDEX idx_user_id ON user_events (user_id)")
    cur.execute("CREATE INDEX idx_event_time ON user_events (event_time)")
    conn.commit()
    conn.close()


# ── Dataset ingestion & schema-adaptation layer ──────────────────
#
#   Uploaded Dataset
#         │
#         ▼
#   Detect dataset schema  (column normalization + heuristics)
#         │
#   ┌─────┼──────────┬───────────┐
#   ▼     ▼          ▼           ▼
# Retail Churn     Event      Generic
#   │     │          │           │
#   └─────┴──────────┴───────────┘
#                │
#                ▼
#       Common event format (user_id, event_type, product_id, price, event_time)
#                │
#                ▼
#     Feature engineering → Temporal graph → DP-GCN   (unchanged downstream)
#
# This lets heterogeneous customer datasets (retail transaction logs like
# UCI Online Retail II, generic clickstream/event logs, and churn-label
# datasets) all be mapped into one unified temporal customer-event
# representation before anything touches the model.
import re


def _norm(s: str) -> str:
    """Normalize a column name for robust matching: 'Customer ID',
    'customer_id', 'customerId' all collapse to 'customerid'."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# Canonical field -> normalized alias tokens that map onto it.
FIELD_ALIASES = {
    "user_id":     ["customerid", "userid", "clientid", "custid", "buyerid",
                     "shopperid", "memberid", "accountid"],
    "event_time":  ["invoicedate", "timestamp", "eventtime", "transactiondate",
                     "purchasedate", "datetime", "date", "orderdate",
                     "eventdate", "createdat", "activitydate"],
    "product_id":  ["stockcode", "productid", "product", "itemid", "sku",
                     "item", "productname"],
    "price":       ["unitprice", "price", "amount", "value", "revenue",
                     "total", "totalprice", "ordervalue", "spend", "sales",
                     "monthlycharges", "totalcharges"],
    "quantity":    ["quantity", "qty", "units", "unitssold"],
    "event_type":  ["eventtype", "event", "action", "type", "activity",
                     "interaction"],
    "churn_label": ["churn", "exited", "attrition", "ischurn", "churned"],
    "tenure":      ["tenure", "tenuremonths", "monthsactive", "customertenure"],
    "invoice_no":  ["invoiceno", "invoiceid", "orderid"],
    "country":     ["country", "region"],
    "description": ["description", "productdescription", "itemdescription"],
}

EVENT_TYPE_ALIASES = {
    "view":     ["view", "viewed", "browse", "browsed", "visit", "impression", "click"],
    "cart":     ["cart", "add_to_cart", "added_to_cart", "basket", "add-to-cart"],
    "purchase": ["purchase", "purchased", "buy", "bought", "order", "sale",
                 "checkout", "transaction", "sold"],
}

DATASET_TYPE_LABELS = {
    "retail":  "Retail / transaction dataset (e.g. UCI Online Retail)",
    "churn":   "Customer churn dataset",
    "event":   "Customer event / clickstream dataset",
    "generic": "Generic dataset (treated as a purchase log)",
}


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a .csv, .xlsx, or .xls file.")


def _build_column_map(df: pd.DataFrame) -> dict:
    """Normalize every uploaded column name and match it against
    FIELD_ALIASES to find which real column (if any) covers each field."""
    norm_lookup = {_norm(c): c for c in df.columns}
    colmap = {}
    for field, aliases in FIELD_ALIASES.items():
        colmap[field] = None
        for alias in aliases:
            if alias in norm_lookup:
                colmap[field] = norm_lookup[alias]
                break
    return colmap


def detect_dataset_schema(colmap: dict) -> str:
    """Classify the uploaded dataset before mapping it, so a
    retail/churn/event dataset each get an adapter suited to its shape
    rather than one hard-coded column mapping for everything."""
    has_event_type = colmap.get("event_type") is not None
    has_churn = colmap.get("churn_label") is not None
    has_tenure = colmap.get("tenure") is not None
    has_invoice = colmap.get("invoice_no") is not None
    has_qty = colmap.get("quantity") is not None
    has_retail_hint = colmap.get("country") is not None or colmap.get("description") is not None

    if has_churn or has_tenure:
        return "churn"
    if has_event_type:
        return "event"
    if has_invoice or (has_qty and (colmap.get("product_id") or has_retail_hint)):
        return "retail"
    return "generic"


def _normalize_event_value(v: str) -> str:
    for canon, aliases in EVENT_TYPE_ALIASES.items():
        if v == canon or v in aliases:
            return canon
    return "purchase"


def adapt_event_dataset(df: pd.DataFrame, colmap: dict, warnings: list) -> pd.DataFrame:
    """Adapter for per-interaction event/clickstream logs (view/cart/purchase)."""
    out = pd.DataFrame()

    if colmap.get("user_id") is None:
        raise ValueError(
            "Couldn't find a customer/user ID column. Please include a "
            "column such as 'user_id', 'Customer ID', or 'customerId'."
        )
    out["user_id"] = df[colmap["user_id"]].astype(str).str.strip()

    if colmap.get("event_type") is not None:
        raw = df[colmap["event_type"]].astype(str).str.lower().str.strip()
        out["event_type"] = raw.apply(_normalize_event_value)
    else:
        warnings.append("No event-type column found — every row was treated as a 'purchase'.")
        out["event_type"] = "purchase"

    if colmap.get("product_id") is not None:
        out["product_id"] = df[colmap["product_id"]].astype(str)
    else:
        warnings.append("No product column found — a placeholder product ID was used.")
        out["product_id"] = "PROD_UNKNOWN"

    if colmap.get("price") is not None:
        out["price"] = pd.to_numeric(df[colmap["price"]], errors="coerce").fillna(0.0)
    else:
        warnings.append("No price/amount column found — prices were set to 0.")
        out["price"] = 0.0

    out["event_time"] = (pd.to_datetime(df[colmap["event_time"]], errors="coerce")
                          if colmap.get("event_time") is not None else pd.NaT)
    return out


def adapt_retail_dataset(df: pd.DataFrame, colmap: dict, warnings: list) -> pd.DataFrame:
    """Adapter for retail/transaction logs (UCI Online Retail-style):
    InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.
    Each line item becomes a 'purchase' event; revenue = Quantity x UnitPrice."""
    out = pd.DataFrame()

    if colmap.get("user_id") is None:
        raise ValueError(
            "Couldn't find a customer/user ID column. Please include a "
            "column such as 'CustomerID' or 'Customer ID'."
        )
    out["user_id"] = df[colmap["user_id"]].astype(str).str.strip()
    out["product_id"] = (df[colmap["product_id"]].astype(str)
                          if colmap.get("product_id") is not None else "PROD_UNKNOWN")
    out["event_type"] = "purchase"
    out["event_time"] = (pd.to_datetime(df[colmap["event_time"]], errors="coerce")
                          if colmap.get("event_time") is not None else pd.NaT)

    qty = (pd.to_numeric(df[colmap["quantity"]], errors="coerce").fillna(1)
           if colmap.get("quantity") is not None else pd.Series(1, index=df.index))
    unit_price = (pd.to_numeric(df[colmap["price"]], errors="coerce").fillna(0.0)
                  if colmap.get("price") is not None else pd.Series(0.0, index=df.index))
    out["price"] = (qty * unit_price).clip(lower=0)

    # UCI-style datasets encode returns/cancellations as negative quantity —
    # drop them since they're not forward purchase signal for the graph.
    if colmap.get("quantity") is not None:
        cancelled = qty <= 0
        if cancelled.any():
            warnings.append(f"Dropped {int(cancelled.sum())} return/cancellation rows (quantity ≤ 0).")
            out = out[~cancelled.values]

    # Rows with no customer ID (common in retail exports — guest checkouts)
    missing_user = out["user_id"].isin(["nan", "none", ""])
    if missing_user.any():
        warnings.append(f"Dropped {int(missing_user.sum())} rows with no customer ID (e.g. guest checkouts).")
        out = out[~missing_user]

    return out


def adapt_churn_dataset(df: pd.DataFrame, colmap: dict, warnings: list) -> pd.DataFrame:
    """Adapter for one-row-per-customer churn datasets (tenure, monthly
    charges, churn label — no event-level history). We synthesize an
    approximate monthly purchase history per customer so the temporal
    graph has something to work with; churned customers' synthetic
    history is cut off early to mimic the dormancy pattern a churn label
    implies. This is inherently weaker than real event data — the churn
    label itself is not a per-event signal, so segmentation quality here
    is only approximate."""
    warnings.append(
        "This looks like a churn dataset (one row per customer, no event-level "
        "history). A synthetic monthly purchase history was generated from "
        "tenure/charges so the temporal graph has something to work with — "
        "treat segmentation results as approximate."
    )
    now = datetime.now()
    rows = []
    user_col = colmap.get("user_id")
    tenure_col = colmap.get("tenure")
    price_col = colmap.get("price")
    churn_col = colmap.get("churn_label")

    for i, row in df.iterrows():
        uid = str(row[user_col]).strip() if user_col else f"CUST_{i}"
        try:
            tenure = int(float(row[tenure_col])) if tenure_col else 6
        except (TypeError, ValueError):
            tenure = 6
        tenure = max(1, min(tenure, 72))

        try:
            monthly = float(row[price_col]) if price_col else 20.0
        except (TypeError, ValueError):
            monthly = 20.0

        churned = False
        if churn_col:
            val = str(row[churn_col]).strip().lower()
            churned = val in ("yes", "true", "1", "churned", "y")
        end_offset_months = random.randint(1, 4) if churned else 0

        for m in range(tenure):
            months_ago = tenure - m - 1 + end_offset_months
            rows.append({
                "user_id": uid,
                "event_type": "purchase",
                "product_id": "SUBSCRIPTION",
                "price": round(monthly, 2),
                "event_time": now - pd.Timedelta(days=30 * months_ago),
            })

    return pd.DataFrame(rows, columns=["user_id", "event_type", "product_id", "price", "event_time"])


def ingest_uploaded_dataframe(raw_df: pd.DataFrame):
    """Full ingestion pipeline: normalize columns → detect dataset type →
    run the matching adapter → return a unified event-shaped DataFrame."""
    warnings = []
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    colmap = _build_column_map(df)
    dataset_type = detect_dataset_schema(colmap)

    if dataset_type == "retail":
        out = adapt_retail_dataset(df, colmap, warnings)
    elif dataset_type == "churn":
        out = adapt_churn_dataset(df, colmap, warnings)
    else:  # "event" or "generic"
        out = adapt_event_dataset(df, colmap, warnings)

    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
    missing_time = out["event_time"].isna()
    if missing_time.any():
        warnings.append(f"{int(missing_time.sum())} rows had a missing/unreadable date — "
                         f"spread across the last 30 days.")
        offsets = np.random.uniform(0, 30 * 24 * 3600, size=int(missing_time.sum()))
        out.loc[missing_time, "event_time"] = [
            datetime.now() - pd.Timedelta(seconds=s) for s in offsets
        ]

    out = out.dropna(subset=["user_id"])
    out = out[out["user_id"].astype(str).str.strip() != ""]
    out = out.reset_index(drop=True)

    if len(out) < 50:
        warnings.append(
            f"Only {len(out)} usable rows found — the DP-GCN model needs at "
            f"least 50 events to train, so segments may be limited until more data arrives."
        )

    return out, warnings, dataset_type


def load_dataframe_into_db(df: pd.DataFrame) -> int:
    reset_database()
    conn = get_connection()
    df[["user_id", "event_type", "product_id", "price", "event_time"]].to_sql(
        "user_events", conn, if_exists="append", index=False
    )
    conn.commit()
    conn.close()
    return len(df)


# ── Real-time streaming worker (TVAE-powered) ────────────────────
# NOTE: a plain module-level dict here would get reset on every Streamlit
# rerun (Streamlit re-executes the whole script top-to-bottom on every
# interaction), which would make stop_streaming() unable to find/stop an
# already-running thread. st.cache_resource keeps this dict alive across
# reruns (and across the whole app's lifetime) so start/stop actually work.
@st.cache_resource
def _get_stream_state():
    return {"thread": None, "stop_event": None}


def _streaming_worker(stop_event: threading.Event):
    from tvae_streamer import sample_synthetic_events

    conn = sqlite3.connect(DB_FILE, timeout=30, check_same_thread=False)
    cur = conn.cursor()

    try:
        # Warm up here (loads the cached model, or trains it once from the
        # seed CSV if no cached copy exists) so the loop below doesn't pay
        # that cost mid-stream.
        sample_synthetic_events(1)
    except Exception as e:
        logging.warning(f"TVAE warm-up failed: {e}")

    while not stop_event.is_set():
        try:
            batch = sample_synthetic_events(random.randint(1, 3))
            for _, row in batch.iterrows():
                cur.execute(
                    """INSERT INTO user_events (user_id, event_type, product_id, price, event_time)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        row["user_id"], row["event_type"], row["product_id"],
                        float(row["price"]), row["event_time"].to_pydatetime(),
                    ),
                )
            conn.commit()
        except Exception as e:
            logging.warning(f"TVAE streaming insert failed: {e}")
        time.sleep(random.uniform(0.3, 0.8))
    conn.close()


def start_streaming():
    state = _get_stream_state()
    if state["thread"] is not None and state["thread"].is_alive():
        return
    stop_event = threading.Event()
    t = threading.Thread(target=_streaming_worker, args=(stop_event,), daemon=True)
    state["thread"] = t
    state["stop_event"] = stop_event
    t.start()


def stop_streaming():
    state = _get_stream_state()
    if state["stop_event"] is not None:
        state["stop_event"].set()
    state["thread"] = None
    state["stop_event"] = None


def is_streaming_active() -> bool:
    state = _get_stream_state()
    return state["thread"] is not None and state["thread"].is_alive()


# ── Initialise DP-GCN (needed on the landing page too, for auto-train) ──
if 'dpgcn' not in st.session_state:
    with st.spinner("Loading DP-GCN model..."):
        st.session_state.dpgcn = get_dpgcn()


# ── Screen 1: choose a data source ──────────────────────────────
def render_landing_page():
    st.title("🚀 DP-GCN Customer Intelligence")
    st.markdown("### Dual-Path Graph Convolution Network for Retention Analysis")
    st.markdown("Choose how you'd like to feed **your store's** data into the model to get started.")
    st.markdown("---")

    col1 = st.container()

    with col1:
        st.markdown("## 📁 Upload Your Store Data")
        st.caption(
            "Upload a CSV or Excel export of your store's customer events "
            "(views, cart adds, purchases) and get an instant dashboard built from it."
        )
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file", type=["csv", "xlsx", "xls"], key="landing_uploader"
        )
        if uploaded_file is not None:
            try:
                raw_df = read_uploaded_file(uploaded_file)
                mapped_df, warnings, dataset_type = ingest_uploaded_dataframe(raw_df)
                n_unique_customers = mapped_df["user_id"].nunique()

                # ── File analysis: show what's actually in the file first ──
                st.markdown("#### 🔍 File analysis")
                st.caption(f"Detected dataset type: **{DATASET_TYPE_LABELS[dataset_type]}**")

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Rows (raw file)", f"{len(raw_df):,}")
                p2.metric("Columns (raw file)", f"{raw_df.shape[1]:,}")
                p3.metric("Usable events", f"{len(mapped_df):,}")
                p4.metric("Unique customers", f"{n_unique_customers:,}")

                if len(mapped_df) > 0:
                    date_min = mapped_df["event_time"].min()
                    date_max = mapped_df["event_time"].max()
                    st.caption(f"📅 Event date range: {date_min:%Y-%m-%d} → {date_max:%Y-%m-%d}")

                with st.expander(f"📋 Raw columns found in your file ({raw_df.shape[1]})"):
                    col_info = pd.DataFrame({
                        "column": raw_df.columns.astype(str),
                        "dtype": [str(t) for t in raw_df.dtypes],
                        "sample value": [
                            str(raw_df[c].dropna().iloc[0]) if raw_df[c].notna().any() else ""
                            for c in raw_df.columns
                        ],
                    })
                    st.dataframe(col_info, use_container_width=True, hide_index=True)

                for w in warnings:
                    st.warning(w)
                if n_unique_customers < 10:
                    st.error(
                        f"Only {n_unique_customers} unique customer(s) found — the DP-GCN "
                        f"model needs at least **10 distinct customers** to build a graph and "
                        f"train. Upload a file with more customers to see segments and charts."
                    )

                with st.expander("👀 Preview mapped data (first 10 rows)"):
                    st.dataframe(mapped_df.head(10), use_container_width=True)

                st.markdown("")
                if st.button("🚀 Generate Dashboard", type="primary", key="load_upload_btn",
                             use_container_width=True):
                    with st.spinner("Loading your data and training the DP-GCN model..."):
                        stop_streaming()
                        n_rows = load_dataframe_into_db(mapped_df)
                        trained = False
                        if n_rows > 50 and n_unique_customers >= 10:
                            trained = st.session_state.dpgcn.train(mapped_df, epochs=20)
                    st.session_state.app_mode = "uploaded"
                    st.session_state.data_source_label = (
                        f"📁 {DATASET_TYPE_LABELS[dataset_type].split(' (')[0]} · {n_rows:,} events"
                    )
                    if trained:
                        st.session_state.upload_train_msg = ("success", "✅ DP-GCN trained on your uploaded data.")
                    elif n_unique_customers < 10:
                        st.session_state.upload_train_msg = (
                            "warning",
                            f"⚠️ Data loaded, but training was skipped — only {n_unique_customers} "
                            f"unique customers (need ≥10). Upload more data or use 'Train DP-GCN' "
                            f"in the sidebar once you have more.",
                        )
                    else:
                        st.session_state.upload_train_msg = (
                            "warning",
                            "⚠️ Data loaded, but training didn't complete. Try 'Train DP-GCN' "
                            "in the sidebar, or check that your file has enough rows per customer.",
                        )
                    st.rerun()
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")

    st.markdown("---")


if st.session_state.get("app_mode") is None:
    render_landing_page()
    st.stop()

st.title("🚀 DP-GCN Customer Segmentation Dashboard")
st.markdown("### Dual-Path Graph Convolution Network for Retention Analysis")

if "upload_train_msg" in st.session_state:
    level, msg = st.session_state.pop("upload_train_msg")
    getattr(st, level)(msg)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎮 Controls")
    st.caption(f"Data source: {st.session_state.get('data_source_label', '—')}")

    st.markdown("---")
    st.subheader("🔴 Real-Time Streaming")
    if is_streaming_active():
        st.caption("🟢 Live stream running — new SDV/TVAE events are being added to this dashboard.")
        if st.button("⏹️ Stop streaming", use_container_width=True):
            stop_streaming()
            st.rerun()
    else:
        st.caption("Adds live, SDV/TVAE-generated events on top of whatever data is loaded now.")
        if st.button("▶️ Start real-time streaming", use_container_width=True):
            with st.spinner("Loading the TVAE model..."):
                start_streaming()
            if st.session_state.get("app_mode") != "streaming":
                existing_label = st.session_state.get("data_source_label", "")
                st.session_state.data_source_label = f"{existing_label} + 🔴 live stream (TVAE)".strip(" +")
            st.rerun()

    st.markdown("---")
    if st.button("🔁 Change data source"):
        stop_streaming()
        for key in ["app_mode", "data_source_label", "predictions", "last_pred_time",
                    "actions_df", "last_action_time", "graph_html_cache", "graph_cache_key",
                    "exp_cache", "exp_cache_key", "cached_events", "last_event_time",
                    "temp_summary", "last_temp_time"]:
            st.session_state.pop(key, None)
        st.rerun()
    st.markdown("---")
    if st.button("🔄 Train DP-GCN"):
        success, err = train_model()
        if success:
            st.success("✅ DP-GCN trained successfully!")
            # Clear caches after training
            for key in ['predictions', 'last_pred_time', 'actions_df',
                        'last_action_time', 'graph_html_cache',
                        'graph_cache_key', 'exp_cache', 'exp_cache_key']:
                st.session_state.pop(key, None)
        else:
            st.warning(err)

    st.markdown("---")
    st.header("🧠 Model Architecture")
    st.markdown("""
    **DP-GCN Components:**
    - **C-GCN**: Connectivity patterns
    - **T-GCN**: Structural role similarity
    - **Multi-Head Attention**: Fuses paths
    - **Residual Path**: Prevents collapse
    - **Anomaly Detection**: Churn prediction
    """)

    st.markdown("---")
    st.header("📊 Segment Definitions")
    st.markdown("""
    - 👑 **VIP_ACTIVE**: High spend + active
    - 🔥 **HIGH_POTENTIAL**: Low history + high activity
    - 💤 **DORMANT_VIP**: High spend + inactive
    - ✅ **LOYAL_REGULAR**: Consistent moderate
    - ⚠️ **AT_RISK**: Declining engagement
    """)

# ── Main Loop ──────────────────────────────────────────────────
placeholder = st.empty()

while True:
    with placeholder.container():
        try:
            # ── Load events (cached 30s) ───────────────────────
            last_event_time = st.session_state.get('last_event_time', 0)
            if time.time() - last_event_time > 30:
                conn = get_connection()
                df_events = pd.read_sql_query(
                    "SELECT * FROM user_events ORDER BY event_time DESC LIMIT 100000",
                    conn
                )
                conn.close()
                st.session_state.cached_events = df_events
                st.session_state.last_event_time = time.time()
            else:
                df_events = st.session_state.get('cached_events', pd.DataFrame())

            # ── Auto-train disabled — manual only ──────────────
            if len(df_events) > 0 and st.session_state.dpgcn.is_trained:

                # ── Predictions (cached 60s) ───────────────────
                last_pred_time = st.session_state.get('last_pred_time', 0)
                if time.time() - last_pred_time > 60:
                    results = st.session_state.dpgcn.predict_customers(df_events)
                    st.session_state.predictions = results
                    st.session_state.last_pred_time = time.time()
                else:
                    results = st.session_state.get('predictions', pd.DataFrame())

                if not results.empty:

                    # ── Metrics ────────────────────────────────
                    col1, col2, col3, col4, col5 = st.columns(5)
                    total_customers = len(results)
                    high_potential  = len(results[results['segment'].str.contains('HIGH_POTENTIAL', na=False)])
                    vip_active      = len(results[results['segment'].str.contains('VIP_ACTIVE', na=False)])
                    at_risk         = len(results[results['segment'].str.contains('AT_RISK', na=False)])
                    high_anomaly    = len(results[results['anomaly_score'] > 0.7])

                    with col1: st.metric("👥 Total Customers", total_customers)
                    with col2: st.metric("🔥 HIGH_POTENTIAL", high_potential, delta="AI Discovered")
                    with col3: st.metric("👑 VIP Active", vip_active)
                    with col4: st.metric("⚠️ At Risk", at_risk)
                    with col5: st.metric("🚨 High Anomaly", high_anomaly, delta="Churn Risk")

                    # ── Charts ─────────────────────────────────
                    col6, col7 = st.columns(2)
                    with col6:
                        seg_counts = results['segment'].value_counts().reset_index()
                        seg_counts.columns = ['segment', 'count']
                        fig_pie = px.pie(
                            seg_counts, values='count', names='segment',
                            title="DP-GCN Customer Segments",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig_pie, use_container_width=True,
                                        key="pie_chart_3")

                    with col7:
                        fig_hist = px.histogram(
                            results, x='anomaly_score', nbins=20,
                            title="Anomaly Score Distribution (Churn Risk)",
                            labels={'anomaly_score': 'Churn Risk Score'},
                            color_discrete_sequence=['#FF6B6B']
                        )
                        fig_hist.add_vline(x=0.7, line_dash="dash",
                                           line_color="red",
                                           annotation_text="High Risk Threshold")
                        st.plotly_chart(fig_hist, use_container_width=True,
                                        key="histogram_chart")

                    # ── Customer Table ─────────────────────────
                    st.subheader("📊 Real-Time Customer Intelligence")
                    display_df = results[['user_id', 'lifetime_value', 'segment',
                                          'anomaly_score', 'confidence']].head(20).copy()
                    display_df['user_id']       = display_df['user_id'].str[:8] + "..."
                    display_df['lifetime_value'] = display_df['lifetime_value'].apply(lambda x: f"${x:.2f}")
                    display_df['anomaly_score']  = display_df['anomaly_score'].apply(lambda x: f"{x:.2%}")
                    display_df['confidence']     = display_df['confidence'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(
                        display_df,
                        column_config={
                            "user_id":       "Customer ID",
                            "lifetime_value":"Lifetime Value",
                            "segment":       "DP-GCN Segment",
                            "anomaly_score": "Churn Risk",
                            "confidence":    "Model Confidence",
                        },
                        use_container_width=True
                    )

                    # ── Customer Relationship Graph ────────────
                    st.markdown("---")
                    st.subheader("🕸️ Customer Relationship Graph")
                    col_g1, col_g2 = st.columns([1, 4])
                    with col_g1:
                        graph_type = st.radio(
                            "Graph type",
                            ["C-GCN (Similarity)", "T-GCN (Topology)"],
                            index=0, key="graph_type_selector"
                        )
                        max_nodes = st.slider("Max nodes", 50, 200, 120, 10,
                                              key="max_nodes_slider")
                    gtype = "cgcn" if "C-GCN" in graph_type else "tgcn"

                    with col_g2:
                        pred_df = st.session_state.get('predictions')
                        if (pred_df is not None and not pred_df.empty
                                and 'segment' in pred_df.columns
                                and pred_df['segment'].notna().any()):

                            graph_cache_key = (f"{gtype}_{max_nodes}_"
                                               f"{st.session_state.get('last_pred_time',0)}")
                            if st.session_state.get('graph_cache_key') != graph_cache_key:
                                graph_html = get_graph_html(pred_df, graph_type=gtype,
                                                            max_nodes=max_nodes)
                                st.session_state.graph_html_cache = graph_html
                                st.session_state.graph_cache_key  = graph_cache_key

                            components.html(
                                st.session_state.get('graph_html_cache',
                                    '<p style="color:#888">Loading...</p>'),
                                height=560, scrolling=False
                            )
                        else:
                            st.info("Train the DP-GCN model first to see the graph.")

                    # ── Explainability ─────────────────────────
                    st.markdown("---")
                    st.subheader("🧠 Why did the model predict these segments?")
                    exp_col1, exp_col2 = st.columns([1, 3])

                    with exp_col1:
                        seg_filter = st.selectbox(
                            "Filter by segment",
                            ["All", "VIP_ACTIVE", "HIGH_POTENTIAL",
                             "DORMANT_VIP", "LOYAL_REGULAR", "AT_RISK"],
                            key="exp_segment_filter"
                        )
                        n_explain = st.slider(
                            "Customers to explain", 1, 5, 2,
                            key="n_explain_slider"
                        )

                    with exp_col2:
                        if st.session_state.get('predictions') is not None:
                            pred_df2 = st.session_state.predictions
                            filter_val = None if seg_filter == "All" else seg_filter

                            # Find matching customers
                            if filter_val:
                                matching = pred_df2[pred_df2['segment'].apply(
                                    lambda s: filter_val.upper() in str(s).upper()
                                )]
                            else:
                                matching = pred_df2

                            st.caption(f"Found {len(matching)} customers matching filter")

                            if len(matching) == 0:
                                st.info("No customers found for this filter.")
                            else:
                                # Cache by filter
                                cache_key = f"exp_{seg_filter}"
                                if st.session_state.get('exp_cache_key') != cache_key:
                                    sample = matching.nlargest(
                                        min(n_explain, len(matching)),
                                        'anomaly_score'
                                    )
                                    conn2 = sqlite3.connect(
                                        "customer_data.db", timeout=30
                                    )
                                    df_ev = pd.read_sql_query(
                                        "SELECT * FROM user_events "
                                        "ORDER BY event_time DESC LIMIT 5000",
                                        conn2
                                    )
                                    conn2.close()
                                    explanations = explain_top_customers(
                                        dpgcn_integrator=st.session_state.dpgcn,
                                        df_events=df_ev,
                                        predictions_df=sample,
                                        n=n_explain,
                                        segment_filter=None
                                    )
                                    st.session_state.exp_cache     = explanations
                                    st.session_state.exp_cache_key = cache_key
                                else:
                                    explanations = st.session_state.get('exp_cache', [])

                                st.write(f"Explanations computed: {len(explanations)}")
                                if explanations:
                                    for exp in explanations:
                                        components.html(
                                            render_explanation_html(exp),
                                            height=450, scrolling=True
                                        )
                                else:
                                    st.info("Could not compute explanations.")
                        else:
                            st.info("Train the DP-GCN model first.")

                    # ── Retention Action Engine ────────────────
                    st.markdown("---")
                    st.subheader("🎯 Retention Action Engine")

                    last_action_time = st.session_state.get('last_action_time', 0)
                    if time.time() - last_action_time > 60:
                        actions_df = generate_actions(st.session_state.predictions)
                        st.session_state.actions_df      = actions_df
                        st.session_state.last_action_time = time.time()
                    else:
                        actions_df = st.session_state.get('actions_df', pd.DataFrame())

                    if not actions_df.empty:
                        summary = get_campaign_summary(actions_df)
                        st.markdown(render_action_summary_html(summary),
                                    unsafe_allow_html=True)

                        act_col1, act_col2, act_col3 = st.columns(3)
                        with act_col1:
                            urgency_filter = st.selectbox(
                                "Filter by urgency",
                                ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                key="urgency_filter"
                            )
                        with act_col2:
                            channel_filter = st.selectbox(
                                "Filter by channel",
                                ["All", "email", "sms", "push"],
                                key="channel_filter"
                            )
                        with act_col3:
                            n_actions = st.slider("Rows to show", 5, 50, 20,
                                                  key="n_actions_slider")

                        filtered = actions_df.copy()
                        if urgency_filter != "All":
                            filtered = filtered[filtered['urgency_label'] == urgency_filter]
                        if channel_filter != "All":
                            filtered = filtered[filtered['channel'] == channel_filter]

                        display_actions = filtered[[
                            'customer_id', 'segment', 'lifetime_value',
                            'churn_risk_pct', 'urgency_label',
                            'channel', 'discount_pct', 'action_label'
                        ]].head(n_actions).copy()
                        display_actions['customer_id']    = display_actions['customer_id'].str[:8] + '...'
                        display_actions['lifetime_value'] = display_actions['lifetime_value'].apply(lambda x: f"${x:,.2f}")
                        display_actions['churn_risk_pct'] = display_actions['churn_risk_pct'].apply(lambda x: f"{x:.1f}%")
                        display_actions['discount_pct']   = display_actions['discount_pct'].apply(lambda x: f"{x}% off")

                        st.dataframe(
                            display_actions,
                            column_config={
                                "customer_id":    "Customer ID",
                                "segment":        "Segment",
                                "lifetime_value": "LTV",
                                "churn_risk_pct": "Churn Risk",
                                "urgency_label":  "Urgency",
                                "channel":        "Channel",
                                "discount_pct":   "Offer",
                                "action_label":   "Action",
                            },
                            use_container_width=True,
                            height=400
                        )

                        csv_bytes = to_csv_bytes(actions_df)
                        st.download_button(
                            label="📥 Download Full Campaign CSV",
                            data=csv_bytes,
                            file_name=f"retention_campaigns_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            key="download_campaigns"
                        )
                        st.caption(
                            f"Generated {len(actions_df):,} personalised actions | "
                            f"Critical: {len(actions_df[actions_df['urgency_label']=='CRITICAL']):,} | "
                            f"High: {len(actions_df[actions_df['urgency_label']=='HIGH']):,}"
                        )

                    # ── Temporal Analysis ──────────────────────
                    st.markdown("---")
                    st.subheader("⏱️ Temporal Behaviour Analysis")
                    last_temp_time = st.session_state.get('last_temp_time', 0)
                    if time.time() - last_temp_time > 120:
                        try:
                            conn3 = sqlite3.connect("customer_data.db", timeout=30)
                            df_temp = pd.read_sql_query(
                                "SELECT * FROM user_events "
                                "ORDER BY event_time DESC LIMIT 10000",
                                conn3
                            )
                            conn3.close()
                            sample_ids = list(
                                st.session_state.predictions['user_id'].head(200)
                            )
                            temp_summary = get_temporal_summary(df_temp, sample_ids)
                            st.session_state.temp_summary  = temp_summary
                            st.session_state.last_temp_time = time.time()
                        except Exception:
                            temp_summary = {}
                    else:
                        temp_summary = st.session_state.get('temp_summary', {})

                    if temp_summary:
                        t1, t2, t3, t4 = st.columns(4)
                        with t1:
                            st.metric("📉 Churning Momentum",
                                      temp_summary.get('churning_momentum', 0))
                        with t2:
                            st.metric("📈 Growing Spend",
                                      temp_summary.get('growing_spend', 0))
                        with t3:
                            st.metric("📉 Declining Spend",
                                      temp_summary.get('declining_spend', 0))
                        with t4:
                            st.metric("♻️ Reactivating",
                                      temp_summary.get('reactivating', 0))

                    # ── Actionable Insights ────────────────────
                    st.markdown("---")
                    st.subheader("💡 Real-Time Actionable Insights")
                    col8, col9, col10 = st.columns(3)
                    with col8:
                        if high_potential > 0:
                            st.success(f"🎯 **{high_potential} HIGH_POTENTIAL customers!**\n"
                                       f"→ Send immediate engagement offers")
                    with col9:
                        if at_risk > 0:
                            st.warning(f"⚠️ **{at_risk} customers AT RISK of churn!**\n"
                                       f"→ Launch retention campaign")
                    with col10:
                        if high_anomaly > 0:
                            st.error(f"🚨 **{high_anomaly} high anomaly scores!**\n"
                                     f"→ Priority intervention needed")

                    # ── Why DP-GCN is Novel ────────────────────
                    st.markdown("---")
                    st.subheader("🧠 Why DP-GCN is Novel")
                    col11, col12 = st.columns(2)
                    with col11:
                        st.info("""
                        **🔗 Dual-Path Architecture**
                        - **C-GCN**: Models customer connectivity
                        - **T-GCN**: Captures structural roles
                        - **Attention**: Intelligently fuses both
                        """)
                    with col12:
                        st.success("""
                        **🎯 Novel Application**
                        - First DP-GCN for customer segmentation
                        - Built-in anomaly detection
                        - Real-time graph-based inference
                        """)

            else:
                st.info("📊 Waiting for DP-GCN model...")
                if len(df_events) < 50:
                    st.warning(f"📈 Collecting data: {len(df_events)}/50 events needed")
                else:
                    st.info("Click 'Train DP-GCN' in the sidebar to start")

            st.caption(
                f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')} | "
                f"DP-GCN Active | Events: {len(df_events)}"
            )

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure data_generator.py is running in another terminal")

    time.sleep(30)