# tvae_streamer.py — real-time synthetic event streaming powered by SDV's TVAE
# ─────────────────────────────────────────────────────────────
# Replaces the old rule-based / Faker-flavoured generator for the
# "Real-Time Data Streaming" mode on the landing screen.
#
# Instead of hand-coded personas and weighted random.choices(), a
# TVAE (Tabular Variational Autoencoder, from the Synthetic Data
# Vault library) is trained once on a seed sample of realistic
# customer-event data (tvae_seed_events.csv, bundled with the
# project — itself drawn from a persistent 300-customer pool so the
# learned distribution still reflects returning-customer behaviour).
# It learns the joint distribution across user_id / event_type /
# product_id / price. At stream time we simply sample new rows from
# the fitted model and stamp them with the current wall-clock time,
# so "real-time" events keep landing while still respecting the
# statistical relationships the model learned (which users tend to
# buy, what price range goes with which product/event-type, etc.)
# rather than a scripted persona state machine.
# ─────────────────────────────────────────────────────────────

import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_FILE = os.path.join(_THIS_DIR, "tvae_seed_events.csv")
MODEL_FILE = os.path.join(_THIS_DIR, "tvae_synthesizer.pkl")
DEFAULT_EPOCHS = 100

_synthesizer = None  # in-process singleton so we only train/load once per run


def _build_metadata(df: pd.DataFrame):
    from sdv.metadata import Metadata
    metadata = Metadata.detect_from_dataframe(data=df, table_name="events")
    # user_id must stay "categorical" (not "id") so TVAE resamples from the
    # learned pool of existing customers instead of inventing brand-new IDs
    # every row — that's what keeps the "returning customer" signal that
    # the temporal graph relies on.
    metadata.update_column(table_name="events", column_name="user_id", sdtype="categorical")
    metadata.update_column(table_name="events", column_name="event_type", sdtype="categorical")
    metadata.update_column(table_name="events", column_name="product_id", sdtype="categorical")
    metadata.update_column(table_name="events", column_name="price", sdtype="numerical")
    return metadata


def train_synthesizer(seed_df: pd.DataFrame, epochs: int = DEFAULT_EPOCHS):
    from sdv.single_table import TVAESynthesizer
    cols = seed_df[["user_id", "event_type", "product_id", "price"]]
    metadata = _build_metadata(cols)
    synthesizer = TVAESynthesizer(metadata, epochs=epochs, enable_gpu=False)
    synthesizer.fit(cols)
    return synthesizer


def get_or_train_synthesizer(force_retrain: bool = False):
    """Return a cached TVAE synthesizer: load it from disk if a pretrained
    copy is bundled/saved, otherwise train one from the seed CSV and cache
    it to disk so the next run doesn't pay the training cost again."""
    global _synthesizer
    if _synthesizer is not None and not force_retrain:
        return _synthesizer

    from sdv.single_table import TVAESynthesizer

    if not force_retrain and os.path.exists(MODEL_FILE):
        try:
            _synthesizer = TVAESynthesizer.load(MODEL_FILE)
            logging.info("✅ Loaded cached TVAE synthesizer from %s", MODEL_FILE)
            return _synthesizer
        except Exception as e:
            logging.warning("Could not load cached TVAE model (%s) — retraining.", e)

    if not os.path.exists(SEED_FILE):
        raise FileNotFoundError(
            f"TVAE seed file not found at {SEED_FILE}. Cannot train the "
            f"real-time streaming model without a seed dataset."
        )

    logging.info("🧠 Training TVAE synthesizer on seed data (%s)...", SEED_FILE)
    seed_df = pd.read_csv(SEED_FILE)
    _synthesizer = train_synthesizer(seed_df, epochs=DEFAULT_EPOCHS)
    try:
        _synthesizer.save(MODEL_FILE)
        logging.info("✅ TVAE synthesizer trained and cached to %s", MODEL_FILE)
    except Exception as e:
        logging.warning("Could not cache TVAE model to disk: %s", e)
    return _synthesizer


def sample_synthetic_events(n: int = 1) -> pd.DataFrame:
    """Sample n new synthetic customer events from the fitted TVAE model
    and stamp them with the current time, ready to insert straight into
    user_events for real-time streaming."""
    synthesizer = get_or_train_synthesizer()
    batch = synthesizer.sample(num_rows=n)
    batch["price"] = pd.to_numeric(batch["price"], errors="coerce").fillna(0.0).clip(lower=0).round(2)
    batch["event_time"] = pd.Timestamp.now()
    return batch[["user_id", "event_type", "product_id", "price", "event_time"]]


def pretrain_and_cache():
    """Convenience entry point to pretrain + save the model ahead of time
    (e.g. at build time), so the first real-time stream in the app starts
    instantly instead of waiting on training."""
    get_or_train_synthesizer(force_retrain=True)


if __name__ == "__main__":
    pretrain_and_cache()
    print(sample_synthetic_events(10))
