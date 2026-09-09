# temporal_gcn.py  — Tier 4 new file
# ─────────────────────────────────────────────────────────────
# What this does:
#   Extends the DP-GCN with a Temporal Graph Network (TGN)
#   memory module that captures HOW customer behaviour changes
#   over time — not just what it is right now.
#
#   The core insight:
#   A customer who spent $500 last month and $50 this month
#   looks DIFFERENT from one who spent $50 last month and $500
#   this month — even though both have identical static features.
#   The DP-GCN cannot distinguish these. The TGN can.
#
#   Architecture:
#   1. TIME ENCODER       — encodes event timestamps as sinusoidal
#                           positional embeddings (same idea as
#                           transformers but for time)
#   2. MEMORY MODULE      — maintains a persistent memory vector
#                           per customer that accumulates over time
#   3. MESSAGE FUNCTION   — computes update messages from new events
#   4. MEMORY UPDATER     — GRU that updates memory with new messages
#   5. TEMPORAL EMBEDDING — combines memory + current features
#
#   Integration with DP-GCN:
#   The temporal embedding is concatenated with the DP-GCN's
#   attention-fused representation before the classifier head.
#   This gives the model both static graph structure AND
#   temporal behavioural trajectory for each customer.
#
#   Usage:
#   from temporal_gcn import TemporalGCN, extract_temporal_features
#   tgn = TemporalGCN(input_dim=8, memory_dim=32, time_dim=16)
#   temporal_emb = tgn(events_df, user_ids, features)
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
#  Time Encoder
# ─────────────────────────────────────────────────────────────

class TimeEncoder(nn.Module):
    """
    Encodes a scalar time value into a dense vector using
    learnable sinusoidal basis functions.

    Inspired by the Time2Vec paper (Kazemi et al., 2019) and
    the original TGN paper (Rossi et al., 2020).

    For time t:
        encoding[0]   = w0 * t + b0         (linear component)
        encoding[1:d] = sin(wi * t + bi)     (periodic components)

    This allows the model to learn both linear trends and
    periodic patterns (daily/weekly cycles) simultaneously.
    """

    def __init__(self, time_dim: int = 16):
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        # Initialise with geometric progression for multi-scale coverage
        with torch.no_grad():
            self.w.weight.data = torch.tensor(
                [[1.0 / (10000 ** (2 * i / time_dim))
                  for i in range(time_dim)]],
                dtype=torch.float
            ).T
            self.w.bias.data = torch.zeros(time_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor of shape [N] or [N, 1]
            Time values in hours since reference point

        Returns
        -------
        torch.Tensor of shape [N, time_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        z = self.w(t.float())
        # First component linear, rest sinusoidal
        return torch.cat([z[:, :1], torch.sin(z[:, 1:])], dim=1)


# ─────────────────────────────────────────────────────────────
#  Memory Module
# ─────────────────────────────────────────────────────────────

class MemoryModule(nn.Module):
    """
    Maintains a persistent memory vector per customer.
    Memory summarises everything that happened to a customer
    up to the current time point.

    The memory is updated using a GRU cell whenever new
    interaction messages arrive for a customer.

    Key properties:
    - Memory is initialised to zero for new customers
    - Memory persists across training batches
    - Updated in chronological event order
    """

    def __init__(self, memory_dim: int = 32, n_customers: int = 10000):
        super().__init__()
        self.memory_dim  = memory_dim
        self.n_customers = n_customers

        # GRU updates memory given a new message
        self.gru = nn.GRUCell(
            input_size  = memory_dim,   # message dimension
            hidden_size = memory_dim    # memory dimension
        )

        # Register memory as a buffer (not a parameter — not trained,
        # but saved with the model and moved to correct device)
        self.register_buffer(
            'memory',
            torch.zeros(n_customers, memory_dim)
        )
        self.register_buffer(
            'last_update',
            torch.zeros(n_customers)
        )

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Retrieve memory for a batch of customer indices."""
        return self.memory[node_ids]

    def update_memory(self, node_ids: torch.Tensor,
                      messages: torch.Tensor):
        """
        Update memory for a batch of customers with new messages.

        Parameters
        ----------
        node_ids : torch.Tensor [B]    customer indices
        messages : torch.Tensor [B, memory_dim]   update messages
        """
        current_memory = self.memory[node_ids]
        new_memory     = self.gru(messages, current_memory)
        self.memory[node_ids] = new_memory.detach()

    def reset_memory(self, node_ids: torch.Tensor = None):
        """Reset memory for specific customers or all customers."""
        if node_ids is None:
            self.memory.zero_()
            self.last_update.zero_()
        else:
            self.memory[node_ids] = 0
            self.last_update[node_ids] = 0


# ─────────────────────────────────────────────────────────────
#  Message Function
# ─────────────────────────────────────────────────────────────

class MessageFunction(nn.Module):
    """
    Computes update messages from interaction events.

    For each event (customer_i, event_type, time, features):
        message = MLP(memory_i || features || time_encoding)

    Where || denotes concatenation.
    """

    def __init__(self, memory_dim: int = 32,
                 feature_dim: int = 8,
                 time_dim: int = 16):
        super().__init__()
        input_size = memory_dim + feature_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_size, memory_dim * 2),
            nn.ReLU(),
            nn.Linear(memory_dim * 2, memory_dim)
        )

    def forward(self, memory: torch.Tensor,
                features: torch.Tensor,
                time_enc: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        memory   : [N, memory_dim]
        features : [N, feature_dim]
        time_enc : [N, time_dim]

        Returns
        -------
        messages : [N, memory_dim]
        """
        combined = torch.cat([memory, features, time_enc], dim=1)
        return self.mlp(combined)


# ─────────────────────────────────────────────────────────────
#  Temporal Embedding Layer
# ─────────────────────────────────────────────────────────────

class TemporalEmbedding(nn.Module):
    """
    Combines memory and current features into a temporal embedding.

    temporal_emb = MLP(memory || current_features || time_since_last)

    This is what gets concatenated with the DP-GCN output.
    """

    def __init__(self, memory_dim: int = 32,
                 feature_dim: int = 8,
                 time_dim: int = 16,
                 output_dim: int = 32):
        super().__init__()
        input_size = memory_dim + feature_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, memory: torch.Tensor,
                features: torch.Tensor,
                time_enc: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([memory, features, time_enc], dim=1)
        return self.mlp(combined)


# ─────────────────────────────────────────────────────────────
#  Full Temporal GCN
# ─────────────────────────────────────────────────────────────

class TemporalGCN(nn.Module):
    """
    Temporal Graph Network (TGN) module for customer intelligence.

    Processes the sequence of customer events in chronological
    order to build a memory representation that captures how
    each customer's behaviour has evolved over time.

    Parameters
    ----------
    input_dim   : int   number of static features (8 in our case)
    memory_dim  : int   dimension of customer memory vectors
    time_dim    : int   dimension of time encodings
    output_dim  : int   output embedding dimension
    n_customers : int   maximum number of customers to track
    """

    def __init__(self,
                 input_dim:   int = 8,
                 memory_dim:  int = 32,
                 time_dim:    int = 16,
                 output_dim:  int = 32,
                 n_customers: int = 10000):
        super().__init__()
        self.memory_dim  = memory_dim
        self.time_dim    = time_dim
        self.output_dim  = output_dim
        self.n_customers = n_customers

        self.time_encoder   = TimeEncoder(time_dim)
        self.memory_module  = MemoryModule(memory_dim, n_customers)
        self.message_fn     = MessageFunction(memory_dim, input_dim, time_dim)
        self.temporal_emb   = TemporalEmbedding(
            memory_dim, input_dim, time_dim, output_dim
        )

        # Reference time — events are encoded as hours from this point
        self.reference_time = None

    def _hours_since_reference(self, timestamps: pd.Series) -> torch.Tensor:
        """Convert timestamps to hours since reference time."""
        if self.reference_time is None:
            self.reference_time = pd.to_datetime(
                timestamps, format='mixed'
            ).min()

        times = pd.to_datetime(timestamps, format='mixed')
        hours = (times - self.reference_time).dt.total_seconds() / 3600.0
        return torch.tensor(hours.values, dtype=torch.float)

    def process_events(self,
                       df_events: pd.DataFrame,
                       user_id_to_idx: dict) -> None:
        """
        Process all events in chronological order to build memories.

        This is called once before inference to populate the memory
        module with the full event history.

        Parameters
        ----------
        df_events      : pd.DataFrame   full event log
        user_id_to_idx : dict           maps user_id → integer index
        """
        if df_events.empty:
            return

        # Sort events chronologically
        df = df_events.copy()
        df['event_time_parsed'] = pd.to_datetime(
            df['event_time'], format='mixed'
        )
        df = df.sort_values('event_time_parsed').reset_index(drop=True)

        # Only process events for known customers
        df = df[df['user_id'].isin(user_id_to_idx)].copy()
        if df.empty:
            return

        df['node_idx'] = df['user_id'].map(user_id_to_idx)

        # Set reference time from data
        self.reference_time = df['event_time_parsed'].min()

        # Process in chronological batches of 500 events
        batch_size = 500
        n_events   = len(df)

        self.eval()
        with torch.no_grad():
            for start in range(0, n_events, batch_size):
                batch = df.iloc[start:start + batch_size]

                node_ids = torch.tensor(
                    batch['node_idx'].values, dtype=torch.long
                )
                hours = torch.tensor(
                    (batch['event_time_parsed'] - self.reference_time
                     ).dt.total_seconds().values / 3600.0,
                    dtype=torch.float
                )

                # Simple 3-feature event representation
                # (purchase=1, cart=0.5, view=0)
                event_type_map = {'purchase': 1.0, 'cart': 0.5, 'view': 0.0}
                event_vals = torch.tensor(
                    batch['event_type'].map(event_type_map).fillna(0).values,
                    dtype=torch.float
                ).unsqueeze(1)

                price_vals = torch.tensor(
                    batch['price'].fillna(0).values / 1000.0,
                    dtype=torch.float
                ).unsqueeze(1)

                # Pad to input_dim with zeros
                pad_dim    = max(0, 8 - 2)
                event_feat = torch.cat([
                    event_vals,
                    price_vals,
                    torch.zeros(len(batch), pad_dim)
                ], dim=1)

                # Time encoding
                time_enc = self.time_encoder(hours)

                # Get current memory for these customers
                current_mem = self.memory_module.get_memory(node_ids)

                # Compute messages
                messages = self.message_fn(current_mem, event_feat, time_enc)

                # Update memory
                self.memory_module.update_memory(node_ids, messages)

    def forward(self,
                features: torch.Tensor,
                user_indices: torch.Tensor,
                current_times: torch.Tensor = None) -> torch.Tensor:
        """
        Compute temporal embeddings for a batch of customers.

        Parameters
        ----------
        features     : [N, input_dim]   current static features
        user_indices : [N]              integer customer indices
        current_times: [N]              hours since reference (optional)

        Returns
        -------
        temporal_emb : [N, output_dim]   temporal embeddings
        """
        # Retrieve accumulated memories
        memory = self.memory_module.get_memory(user_indices)

        # Time since reference (or zero if not provided)
        if current_times is None:
            current_times = torch.zeros(len(user_indices))
        time_enc = self.time_encoder(current_times)

        # Compute temporal embedding
        return self.temporal_emb(memory, features, time_enc)


# ─────────────────────────────────────────────────────────────
#  DP-GCN + TGN Combined Model
# ─────────────────────────────────────────────────────────────

class DP_GCN_Temporal(nn.Module):
    """
    Full model: DP-GCN + Temporal GCN combined.

    Architecture:
        Static path:   C-GCN + T-GCN + Attention → [N, output_dim]
        Temporal path: TGN memory + time encoding → [N, tgn_dim]
        Combined:      concat → [N, output_dim + tgn_dim]
        Classifier:    Linear → [N, 5]
        Anomaly:       Linear → [N, 1]

    The temporal path adds trajectory awareness on top of the
    existing structural graph awareness.
    """

    def __init__(self,
                 dp_gcn_model,          # existing trained DP_GCN
                 dp_gcn_output_dim: int = 32,
                 tgn_output_dim:    int = 32,
                 num_classes:       int = 5,
                 n_customers:       int = 10000):
        super().__init__()

        # Freeze the existing DP-GCN (don't retrain it)
        self.dp_gcn = dp_gcn_model
        for param in self.dp_gcn.parameters():
            param.requires_grad = False

        # TGN module (trainable)
        self.tgn = TemporalGCN(
            input_dim   = 8,
            memory_dim  = 32,
            time_dim    = 16,
            output_dim  = tgn_output_dim,
            n_customers = n_customers
        )

        # New classifier heads that use combined representation
        combined_dim = dp_gcn_output_dim + tgn_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),           nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(combined_dim, 16), nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, c_edge_index, t_edge_index,
                user_indices, current_times=None):
        """
        Parameters
        ----------
        x              : [N, 8]    customer features
        c_edge_index   : C-GCN edges
        t_edge_index   : T-GCN edges
        user_indices   : [N]       integer customer indices
        current_times  : [N]       hours since reference (optional)

        Returns
        -------
        seg_probs   : [N, 5]   segment probabilities
        anom_scores : [N, 1]   churn risk scores
        """
        # ── Static DP-GCN path (frozen) ───────────────────────
        with torch.no_grad():
            c_feat   = self.dp_gcn.c_gcn(x, c_edge_index)
            t_feat   = self.dp_gcn.t_gcn(x, t_edge_index)
            combined = self.dp_gcn.attention(
                c_feat.unsqueeze(0), t_feat.unsqueeze(0)
            ).squeeze(0)

        # ── Temporal TGN path ─────────────────────────────────
        temporal_emb = self.tgn(x, user_indices, current_times)

        # ── Combine ───────────────────────────────────────────
        full = torch.cat([combined, temporal_emb], dim=1)

        seg_logits  = self.classifier(full)
        seg_probs   = F.softmax(seg_logits, dim=1)
        anom_scores = self.anomaly_head(full)

        return seg_probs, anom_scores


# ─────────────────────────────────────────────────────────────
#  Temporal Feature Extraction (standalone utility)
# ─────────────────────────────────────────────────────────────

def extract_temporal_features(df_events: pd.DataFrame,
                               user_ids: list) -> pd.DataFrame:
    """
    Extract temporal behavioural features per customer.
    These complement the static 8 features with trajectory signals.

    Features computed:
    - spend_trend      : slope of spend over time (positive = growing)
    - activity_trend   : slope of event count over time
    - session_velocity : events per day in recent vs overall period
    - churn_momentum   : rate of decline in last 7 vs previous 7 days
    - purchase_recency_trend : is gap between purchases increasing?

    Parameters
    ----------
    df_events : pd.DataFrame   full event log
    user_ids  : list           ordered list of user IDs

    Returns
    -------
    pd.DataFrame with one row per user, columns = temporal features
    """
    results = []

    for uid in user_ids:
        ud = df_events[df_events['user_id'] == uid].copy()

        if ud.empty:
            results.append({
                'user_id':             uid,
                'spend_trend':         0.0,
                'activity_trend':      0.0,
                'session_velocity':    0.0,
                'churn_momentum':      0.0,
                'purchase_recency_trend': 0.0,
            })
            continue

        try:
            ud['ts'] = pd.to_datetime(ud['event_time'], format='mixed')
            ud = ud.sort_values('ts')

            # Reference: hours since first event
            t0  = ud['ts'].min()
            ud['hours'] = (ud['ts'] - t0).dt.total_seconds() / 3600.0
            max_hours   = ud['hours'].max()

            # ── Spend trend ─────────────────────────────────────
            purchases = ud[ud['event_type'] == 'purchase']
            if len(purchases) >= 2:
                x = purchases['hours'].values
                y = purchases['price'].fillna(0).values
                if x.std() > 0:
                    spend_trend = float(np.polyfit(x, y, 1)[0])
                else:
                    spend_trend = 0.0
            else:
                spend_trend = 0.0

            # ── Activity trend ───────────────────────────────────
            if max_hours > 24 and len(ud) >= 4:
                # Count events in first half vs second half
                mid = max_hours / 2
                first_half  = len(ud[ud['hours'] <= mid])
                second_half = len(ud[ud['hours'] > mid])
                activity_trend = float(second_half - first_half) / max(1, len(ud))
            else:
                activity_trend = 0.0

            # ── Session velocity ─────────────────────────────────
            now_h     = max_hours
            recent_7d = len(ud[ud['hours'] >= now_h - 168])   # last 7 days
            overall   = len(ud) / max(1, max_hours / 24)      # events/day overall
            recent_rate = recent_7d / 7.0
            session_velocity = float(recent_rate / max(0.001, overall) - 1.0)

            # ── Churn momentum ───────────────────────────────────
            # Negative = declining activity (churn signal)
            last_7d   = len(ud[ud['hours'] >= now_h - 168])
            prev_7d   = len(ud[(ud['hours'] >= now_h - 336) &
                               (ud['hours'] < now_h - 168)])
            churn_momentum = float(last_7d - prev_7d) / max(1, prev_7d + 1)

            # ── Purchase recency trend ───────────────────────────
            # Positive = gaps between purchases getting shorter (good)
            # Negative = gaps getting longer (churn signal)
            if len(purchases) >= 3:
                gaps = np.diff(purchases['hours'].values)
                if len(gaps) >= 2:
                    prec_trend = float(gaps[-1] - gaps[0]) / max(1, gaps[0])
                    prec_trend = -prec_trend   # negate so positive = good
                else:
                    prec_trend = 0.0
            else:
                prec_trend = 0.0

        except Exception:
            spend_trend = activity_trend = session_velocity = 0.0
            churn_momentum = prec_trend = 0.0

        results.append({
            'user_id':                uid,
            'spend_trend':            round(spend_trend, 4),
            'activity_trend':         round(activity_trend, 4),
            'session_velocity':       round(session_velocity, 4),
            'churn_momentum':         round(churn_momentum, 4),
            'purchase_recency_trend': round(prec_trend, 4),
        })

    return pd.DataFrame(results)


def get_temporal_summary(df_events: pd.DataFrame,
                          user_ids: list) -> dict:
    """
    Compute population-level temporal statistics.
    Used for dashboard display.

    Returns
    -------
    dict with summary statistics about behavioural trends
    """
    tf = extract_temporal_features(df_events, user_ids)

    if tf.empty:
        return {}

    churning = (tf['churn_momentum'] < -0.3).sum()
    growing  = (tf['spend_trend'] > 0).sum()
    declining = (tf['spend_trend'] < 0).sum()
    reactivating = (tf['activity_trend'] > 0.2).sum()

    return {
        'total_customers':   len(tf),
        'churning_momentum': int(churning),
        'growing_spend':     int(growing),
        'declining_spend':   int(declining),
        'reactivating':      int(reactivating),
        'avg_churn_momentum': round(float(tf['churn_momentum'].mean()), 4),
        'avg_spend_trend':    round(float(tf['spend_trend'].mean()), 4),
    }
