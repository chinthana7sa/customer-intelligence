# # dpgcn_model.py  — Tier 2 extended version
# # ─────────────────────────────────────────────────────────────
# # What changed vs the original:
# #   NEW — 80/20 train/validation split inside train()
# #   NEW — per-epoch val accuracy + val loss tracked separately
# #   NEW — per-class F1 score computed after training
# #   NEW — full training run logged to SQLite (training_history table)
# #   NEW — get_training_history() for dashboard accuracy trend chart
# #   NEW — best-model checkpointing (saves weights at best val accuracy)
# #   NEW — early stopping (stops if val loss doesn't improve for 10 epochs)
# #   KEPT — everything original: C_GCN, T_GCN, MultiHeadAttention,
# #           DP_GCN, CustomerGraphBuilder, pseudo-labels, predict_customers
# # ─────────────────────────────────────────────────────────────

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool
# from torch_geometric.data import Data, Batch
# import numpy as np
# import pandas as pd
# import networkx as nx
# import sqlite3
# import pickle
# import json
# from datetime import datetime, timedelta
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import f1_score, classification_report
# from sklearn.model_selection import train_test_split
# from collections import defaultdict
# import warnings
# warnings.filterwarnings('ignore')

# DB_FILE = "customer_data.db"

# # ─────────────────────────────────────────────────────────────
# #  Neural network layers — unchanged from original
# # ─────────────────────────────────────────────────────────────

# class C_GCN(nn.Module):
#     """Connectivity GCN — captures direct behavioural similarity edges."""
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.3, training=self.training)
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.conv3(x, edge_index)
#         return x


# class T_GCN(nn.Module):
#     """Topology GCN — captures structural role similarity via KMeans clusters."""
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x


# class MultiHeadAttention(nn.Module):
#     """4-head self-attention — fuses C-GCN and T-GCN representations."""
#     def __init__(self, in_channels, num_heads=4):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim  = in_channels // num_heads
#         assert self.head_dim * num_heads == in_channels, \
#             "in_channels must be divisible by num_heads"
#         self.query    = nn.Linear(in_channels, in_channels)
#         self.key      = nn.Linear(in_channels, in_channels)
#         self.value    = nn.Linear(in_channels, in_channels)
#         self.out_proj = nn.Linear(in_channels, in_channels)

#     def forward(self, c_features, t_features):
#         combined   = c_features + t_features
#         bs, n, d   = combined.shape
#         Q = self.query(combined).view(bs, n, self.num_heads, self.head_dim)
#         K = self.key(combined).view(bs, n, self.num_heads, self.head_dim)
#         V = self.value(combined).view(bs, n, self.num_heads, self.head_dim)
#         attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attn = F.softmax(attn, dim=-1)
#         out  = torch.matmul(attn, V)
#         out  = out.transpose(1, 2).contiguous().view(bs, n, d)
#         return self.out_proj(out)


# class DP_GCN(nn.Module):
#     """
#     Dual-Path Graph Convolutional Network.
#     Two independent GCN paths → attention fusion → two task heads:
#       1. Segment classifier  (5-class softmax)
#       2. Anomaly detector    (churn risk sigmoid)
#     """
#     def __init__(self, input_dim, hidden_dim=128, output_dim=32, num_classes=5):
#         super().__init__()
#         self.c_gcn      = C_GCN(input_dim, hidden_dim, output_dim)
#         self.t_gcn      = T_GCN(input_dim, hidden_dim, output_dim)
#         self.attention  = MultiHeadAttention(output_dim, num_heads=4)
#         self.classifier = nn.Sequential(
#             nn.Linear(output_dim, 64), nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),         nn.ReLU(),
#             nn.Linear(32, num_classes)
#         )
#         self.anomaly_head = nn.Sequential(
#             nn.Linear(output_dim, 16), nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x, c_edge_index, t_edge_index):
#         c_feat   = self.c_gcn(x, c_edge_index)
#         t_feat   = self.t_gcn(x, t_edge_index)
#         combined = self.attention(
#             c_feat.unsqueeze(0), t_feat.unsqueeze(0)
#         ).squeeze(0)
#         seg_logits  = self.classifier(combined)
#         seg_probs   = F.softmax(seg_logits, dim=1)
#         anom_scores = self.anomaly_head(combined)
#         return seg_probs, anom_scores


# # ─────────────────────────────────────────────────────────────
# #  Graph builder — unchanged from original
# # ─────────────────────────────────────────────────────────────

# class CustomerGraphBuilder:
#     """Extracts 8 behavioural features per customer and builds both graphs."""

#     def __init__(self):
#         self.scaler = StandardScaler()
#         self.feature_names = [
#             'total_spent', 'purchase_count', 'avg_order_value',
#             'views_last_7d', 'carts_last_7d', 'days_since_last_purchase',
#             'browsing_frequency', 'cart_abandonment_rate'
#         ]

#     def extract_features(self, df_events):
#         features = []
#         user_ids = df_events['user_id'].unique()
#         seven_days_ago = datetime.now() - timedelta(days=7)

#         for uid in user_ids:
#             ud = df_events[df_events['user_id'] == uid]
#             purchases = ud[ud['event_type'] == 'purchase']
#             views     = ud[ud['event_type'] == 'view']
#             carts     = ud[ud['event_type'] == 'cart']

#             total_spent     = purchases['price'].sum() if not purchases.empty else 0
#             purchase_count  = len(purchases)
#             avg_order_value = total_spent / purchase_count if purchase_count > 0 else 0

#             recent_views = len(views[pd.to_datetime(views['event_time'], format='mixed', errors='coerce') > seven_days_ago])
#             recent_carts = len(carts[pd.to_datetime(carts['event_time'], format='mixed', errors='coerce') > seven_days_ago])

#             if purchases.empty:
#                 days_since = 365
#             else:
#                 last_purchase = pd.to_datetime(
#                     purchases["event_time"].max(),
#                     errors="coerce",
#                     format="mixed"
#                 )
#                 if pd.isna(last_purchase):
#                     days_since = 365
#                 else:
#                     days_since = (datetime.now() - last_purchase).days

#             time_span = max(1, (
#                 pd.to_datetime(ud['event_time'], format='mixed', errors='coerce').max() -
#                 pd.to_datetime(ud['event_time'], format='mixed', errors='coerce').min()
#             ).days)
#             browsing_freq     = len(views) / time_span
#             cart_abandonment  = 1 - (purchase_count / max(1, len(carts)))

#             features.append([
#                 total_spent, purchase_count, avg_order_value,
#                 recent_views, recent_carts, days_since,
#                 browsing_freq, cart_abandonment
#             ])

#         features_scaled = self.scaler.fit_transform(features)
#         return user_ids, features_scaled

#     def build_connectivity_graph(self, user_ids, features):
#         """C-GCN edges: top-5 cosine-similar customers per node."""
#         from sklearn.metrics.pairwise import cosine_similarity
#         sim    = cosine_similarity(features)
#         edges  = []
#         k      = min(5, len(user_ids) - 1)
#         for i in range(len(user_ids)):
#             top_k = np.argsort(sim[i])[::-1][1:k + 1]
#             for j in top_k:
#                 edges.append([i, j])
#         return torch.tensor(edges, dtype=torch.long).t().contiguous()

#     def build_topology_graph(self, user_ids, features):
#         """T-GCN edges: customers in the same KMeans cluster."""
#         from sklearn.cluster import KMeans
#         n_clusters = min(10, len(user_ids) // 10 + 1)
#         kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         clusters   = kmeans.fit_predict(features)
#         edges = []
#         for cid in range(n_clusters):
#             idx = np.where(clusters == cid)[0]
#             for i in range(len(idx)):
#                 for j in range(i + 1, len(idx)):
#                     edges.append([idx[i], idx[j]])
#         if not edges:
#             return torch.tensor([[0], [0]], dtype=torch.long)
#         return torch.tensor(edges, dtype=torch.long).t().contiguous()


# # ─────────────────────────────────────────────────────────────
# #  Training history — NEW in Tier 2
# # ─────────────────────────────────────────────────────────────

# def _ensure_history_table():
#     """Create training_history table if it doesn't exist."""
#     conn = sqlite3.connect(DB_FILE, timeout=30)
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS training_history (
#             id            INTEGER PRIMARY KEY AUTOINCREMENT,
#             trained_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             num_customers INTEGER,
#             num_events    INTEGER,
#             epochs        INTEGER,
#             train_loss    REAL,
#             val_loss      REAL,
#             train_acc     REAL,
#             val_acc       REAL,
#             f1_macro      REAL,
#             f1_per_class  TEXT,
#             stopped_early INTEGER DEFAULT 0
#         )
#     """)
#     conn.commit()
#     conn.close()


# def _log_training_run(num_customers, num_events, epochs_run,
#                       train_loss, val_loss, train_acc, val_acc,
#                       f1_macro, f1_per_class, stopped_early):
#     """Append one training run record to SQLite."""
#     _ensure_history_table()
#     conn = sqlite3.connect(DB_FILE, timeout=30)
#     conn.execute("""
#         INSERT INTO training_history
#             (num_customers, num_events, epochs, train_loss, val_loss,
#              train_acc, val_acc, f1_macro, f1_per_class, stopped_early)
#         VALUES (?,?,?,?,?,?,?,?,?,?)
#     """, (
#         num_customers, num_events, epochs_run,
#         round(train_loss, 6), round(val_loss, 6),
#         round(train_acc, 4),  round(val_acc, 4),
#         round(f1_macro, 4),   json.dumps(f1_per_class),
#         int(stopped_early)
#     ))
#     conn.commit()
#     conn.close()


# def get_training_history(limit: int = 50) -> pd.DataFrame:
#     """
#     Returns the last `limit` training runs as a DataFrame.
#     Used by the dashboard to draw the accuracy trend chart.
#     """
#     _ensure_history_table()
#     conn = sqlite3.connect(DB_FILE)
#     df = pd.read_sql_query(f"""
#         SELECT trained_at, num_customers, epochs,
#                train_loss, val_loss,
#                train_acc,  val_acc,
#                f1_macro,   stopped_early
#         FROM   training_history
#         ORDER  BY id DESC
#         LIMIT  {limit}
#     """, conn)
#     conn.close()
#     return df.iloc[::-1].reset_index(drop=True)   # chronological order


# # ─────────────────────────────────────────────────────────────
# #  Main integrator — extended with Tier 2 additions
# # ─────────────────────────────────────────────────────────────

# class DP_GCN_Integrator:
#     """
#     Wraps the DP_GCN model with:
#     — feature extraction & graph building (original)
#     — 80/20 train/val split              (Tier 2 NEW)
#     — per-epoch val tracking             (Tier 2 NEW)
#     — early stopping (patience=10)       (Tier 2 NEW)
#     — best-model checkpointing           (Tier 2 NEW)
#     — F1 score per class                 (Tier 2 NEW)
#     — SQLite training history logging    (Tier 2 NEW)
#     """

#     SEGMENT_MAP = {
#         0: '👑 VIP_ACTIVE',
#         1: '🔥 HIGH_POTENTIAL',
#         2: '💤 DORMANT_VIP',
#         3: '✅ LOYAL_REGULAR',
#         4: '⚠️ AT_RISK',
#     }
#     SEGMENT_NAMES = ['VIP_ACTIVE', 'HIGH_POTENTIAL', 'DORMANT_VIP',
#                      'LOYAL_REGULAR', 'AT_RISK']

#     def __init__(self):
#         self.model         = None
#         self.graph_builder = CustomerGraphBuilder()
#         self.user_ids      = None
#         self.is_trained    = False
#         # Last training metrics — exposed to dashboard
#         self.last_val_acc   = 0.0
#         self.last_val_loss  = 0.0
#         self.last_f1_macro  = 0.0
#         self.last_f1_class  = {}

#     # ── Pseudo-label generation — FINAL (v4) ───────────────────
#     # Calibrated against real DB stats:
#     #   14,623 events | 1,264 users | 598 purchasers
#     #   avg spend $440 | max spend $4,676
#     #
#     # Strategy: percentiles over purchasers only for spend tiers.
#     #           total event count (not recent window) for activity,
#     #           so labels don't depend on what time of day training runs.
#     def _generate_pseudo_labels(self, df_events):
#         purchases_all = df_events[df_events['event_type'] == 'purchase']
#         spend_by_user = purchases_all.groupby('user_id')['price'].sum()
#         all_spend     = spend_by_user.reindex(self.user_ids, fill_value=0)

#         # ── All-time activity per user (no time window) ─────────
#         views_all = df_events[df_events['event_type'] == 'view'].groupby('user_id').size()
#         carts_all = df_events[df_events['event_type'] == 'cart'].groupby('user_id').size()
#         total_events_user = df_events.groupby('user_id').size()

#         # ── Spend tiers from purchasers only ─────────────────────
#         nonzero = all_spend[all_spend > 0]
#         n       = len(nonzero)

#         if n >= 10:
#             vip_cutoff = nonzero.quantile(0.80)   # top 20% of buyers → VIP
#             mid_low    = nonzero.quantile(0.20)   # bottom 20% of buyers → LOYAL low
#             mid_high   = nonzero.quantile(0.80)   # up to VIP → LOYAL high
#         elif n >= 2:
#             vip_cutoff = nonzero.max() * 0.70
#             mid_low    = nonzero.min()
#             mid_high   = nonzero.max() * 0.70
#         elif n == 1:
#             vip_cutoff = nonzero.iloc[0]
#             mid_low    = 0.01
#             mid_high   = nonzero.iloc[0]
#         else:
#             vip_cutoff = mid_low = mid_high = float('inf')

#         labels = []
#         for uid in self.user_ids:
#             total_spent  = float(all_spend.get(uid, 0.0))
#             v_all        = int(views_all.get(uid, 0))
#             c_all        = int(carts_all.get(uid, 0))
#             n_events     = int(total_events_user.get(uid, 0))
#             is_buyer     = total_spent > 0
#             is_active    = (v_all + c_all) >= 2   # at least 2 browse/cart events

#             if   is_buyer and total_spent >= vip_cutoff and is_active:
#                 labels.append(0)   # VIP_ACTIVE     — top spender, engaged
#             elif is_buyer and total_spent >= vip_cutoff and not is_active:
#                 labels.append(2)   # DORMANT_VIP    — top spender, low activity
#             elif is_buyer and mid_low <= total_spent < mid_high:
#                 labels.append(3)   # LOYAL_REGULAR  — mid spender
#             elif not is_buyer and n_events >= 3:
#                 labels.append(1)   # HIGH_POTENTIAL — browsing, never bought
#             else:
#                 labels.append(4)   # AT_RISK        — low engagement
#         return labels

#     # ── Training — Tier 2 extended ────────────────────────────
#     def train(self, df_events, epochs: int = 50, patience: int = 10,
#               db_file: str = DB_FILE):
#         """
#         Train DP-GCN with 80/20 train/val split, early stopping,
#         best-model checkpointing, and SQLite history logging.

#         Parameters
#         ----------
#         df_events : pd.DataFrame   all events from SQLite
#         epochs    : int            max training epochs
#         patience  : int            early-stopping patience
#         db_file   : str            SQLite path for history logging
#         """
#         print("🏗️  Building customer graphs...")
#         self.user_ids, features = self.graph_builder.extract_features(df_events)
#         n = len(self.user_ids)

#         if n < 10:
#             print("⚠️  Need at least 10 customers for training")
#             return False

#         # ── Build both graphs ──────────────────────────────────
#         c_edge_index = self.graph_builder.build_connectivity_graph(self.user_ids, features)
#         t_edge_index = self.graph_builder.build_topology_graph(self.user_ids, features)
#         x            = torch.tensor(features, dtype=torch.float)

#         # ── Pseudo-labels ──────────────────────────────────────
#         all_labels = self._generate_pseudo_labels(df_events)
#         all_labels = all_labels[:n]

#         # ── 80/20 train / val split (NEW) ─────────────────────
#         all_idx   = np.arange(n)
#         train_idx, val_idx = train_test_split(
#             all_idx, test_size=0.2, random_state=42,
#             stratify=all_labels if len(set(all_labels)) > 1 else None
#         )
#         train_mask = torch.zeros(n, dtype=torch.bool)
#         val_mask   = torch.zeros(n, dtype=torch.bool)
#         train_mask[train_idx] = True
#         val_mask[val_idx]     = True

#         labels_tensor = torch.tensor(all_labels, dtype=torch.long)
#         at_risk_mask  = (labels_tensor == 4).float().unsqueeze(1)

#         print(f"   Train: {train_mask.sum().item()} customers | "
#               f"Val: {val_mask.sum().item()} customers")

#         # ── Model & optimiser ──────────────────────────────────
#         input_dim  = features.shape[1]
#         self.model = DP_GCN(input_dim=input_dim, hidden_dim=64,
#                             output_dim=32, num_classes=5)
#         optimizer      = torch.optim.Adam(self.model.parameters(),
#                                           lr=0.001, weight_decay=1e-5)
#         class_weights = torch.tensor([2.0, 3.0, 2.0, 2.0, 3.0])
#         criterion_seg  = nn.CrossEntropyLoss(weight=class_weights)
#         criterion_anom = nn.BCELoss()

#         # ── Early stopping & checkpointing state (NEW) ─────────
#         best_val_loss   = float('inf')
#         best_val_acc    = 0.0
#         best_state      = None
#         patience_count  = 0
#         stopped_early   = False

#         final_train_loss = 0.0
#         final_val_loss   = 0.0
#         final_train_acc  = 0.0
#         final_val_acc    = 0.0
#         epochs_run       = 0

#         print(f"🚀 Training DP-GCN on {n} customers for up to {epochs} epochs...")

#         for epoch in range(epochs):
#             # ── Train step ──────────────────────────────────────
#             self.model.train()
#             optimizer.zero_grad()

#             seg_probs, anom_scores = self.model(x, c_edge_index, t_edge_index)

#             # Loss only on train nodes
#             loss_seg  = criterion_seg(seg_probs[train_mask],
#                                       labels_tensor[train_mask])
#             loss_anom = criterion_anom(anom_scores[train_mask],
#                                        at_risk_mask[train_mask])
#             total_loss = loss_seg + 0.5 * loss_anom
#             total_loss.backward()
#             optimizer.step()

#             train_acc = (
#                 seg_probs[train_mask].argmax(dim=1) == labels_tensor[train_mask]
#             ).float().mean().item()

#             # ── Validation step (NEW) ───────────────────────────
#             self.model.eval()
#             with torch.no_grad():
#                 seg_probs_v, anom_scores_v = self.model(x, c_edge_index, t_edge_index)
#                 val_loss_seg  = criterion_seg(seg_probs_v[val_mask],
#                                               labels_tensor[val_mask])
#                 val_loss_anom = criterion_anom(anom_scores_v[val_mask],
#                                                at_risk_mask[val_mask])
#                 val_loss = (val_loss_seg + 0.5 * val_loss_anom).item()
#                 val_acc  = (
#                     seg_probs_v[val_mask].argmax(dim=1) == labels_tensor[val_mask]
#                 ).float().mean().item()

#             epochs_run       = epoch + 1
#             final_train_loss = total_loss.item()
#             final_val_loss   = val_loss
#             final_train_acc  = train_acc
#             final_val_acc    = val_acc

#             # ── Best checkpoint (NEW) ───────────────────────────
#             if val_loss < best_val_loss:
#                 best_val_loss  = val_loss
#                 best_val_acc   = val_acc
#                 best_state     = {k: v.clone() for k, v in
#                                   self.model.state_dict().items()}
#                 patience_count = 0
#             else:
#                 patience_count += 1

#             if (epoch + 1) % 10 == 0:
#                 print(f"   Epoch {epoch+1:3d}/{epochs} | "
#                       f"Train loss: {total_loss.item():.4f} acc: {train_acc:.3f} | "
#                       f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}")

#             # ── Early stopping (NEW) ────────────────────────────
#             if patience_count >= patience:
#                 print(f"   ⏹️  Early stopping at epoch {epoch+1} "
#                       f"(no val improvement for {patience} epochs)")
#                 stopped_early = True
#                 break

#         # ── Restore best weights (NEW) ─────────────────────────
#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#             print(f"   ✅ Restored best weights "
#                   f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.3f})")

#         # ── F1 score per class (NEW) ───────────────────────────
#         self.model.eval()
#         with torch.no_grad():
#             seg_probs_f, _ = self.model(x, c_edge_index, t_edge_index)
#             preds  = seg_probs_f.argmax(dim=1).numpy()
#             truths = labels_tensor.numpy()

#         f1_macro = f1_score(truths, preds, average='macro', zero_division=0)
#         try:
#             f1_per = f1_score(truths, preds, average=None, zero_division=0)
#             f1_lst = list(f1_per)
#             while len(f1_lst) < 5:
#                 f1_lst.append(0.0)
#         except Exception:
#             f1_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
#         f1_class = {
#             'VIP_ACTIVE':     round(float(f1_lst[0]), 4),
#             'HIGH_POTENTIAL': round(float(f1_lst[1]), 4),
#             'DORMANT_VIP':    round(float(f1_lst[2]), 4),
#             'LOYAL_REGULAR':  round(float(f1_lst[3]), 4),
#             'AT_RISK':        round(float(f1_lst[4]), 4),
#         }
#         print(f"   📊 F1 macro: {f1_macro:.4f}")
#         for seg, score in f1_class.items():
#             print(f"      {seg}: {score:.4f}")

#         # ── Save model & scaler ────────────────────────────────
#         torch.save(self.model.state_dict(), 'dpgcn_model.pt')
#         with open('dpgcn_scaler.pkl', 'wb') as f:
#             pickle.dump(self.graph_builder.scaler, f)

#         # ── Expose metrics to dashboard ────────────────────────
#         self.last_val_acc  = best_val_acc
#         self.last_val_loss = best_val_loss
#         self.last_f1_macro = f1_macro
#         self.last_f1_class = f1_class
#         self.is_trained    = True

#         # ── Log to SQLite training_history (NEW) ───────────────
#         _log_training_run(
#             num_customers = n,
#             num_events    = len(df_events),
#             epochs_run    = epochs_run,
#             train_loss    = final_train_loss,
#             val_loss      = best_val_loss,
#             train_acc     = final_train_acc,
#             val_acc       = best_val_acc,
#             f1_macro      = f1_macro,
#             f1_per_class  = f1_class,
#             stopped_early = stopped_early
#         )
#         print(f"✅ Training complete — run logged to {db_file}")
#         return True

#     # ── Inference — unchanged from original ───────────────────
#     def predict_customers(self, df_events):
#         """Run DP-GCN inference on all customers. Returns a DataFrame."""
#         if not self.is_trained:
#             return pd.DataFrame()

#         self.user_ids, features = self.graph_builder.extract_features(df_events)
#         if len(self.user_ids) == 0:
#             return pd.DataFrame()

#         c_edge_index = self.graph_builder.build_connectivity_graph(
#             self.user_ids, features)
#         t_edge_index = self.graph_builder.build_topology_graph(
#             self.user_ids, features)

#         self.model.eval()
#         with torch.no_grad():
#             x = torch.tensor(features, dtype=torch.float)
#             seg_probs, anom_scores = self.model(x, c_edge_index, t_edge_index)
#             segments   = seg_probs.argmax(dim=1).numpy()
#             anomaly    = anom_scores.numpy().flatten()

#         results = pd.DataFrame({
#             'user_id':      self.user_ids,
#             'segment':      [self.SEGMENT_MAP[s] for s in segments],
#             'anomaly_score': anomaly,
#             'confidence':   seg_probs.max(dim=1)[0].numpy(),
#         })

#         # Lifetime value
#         ltv = (df_events.groupby('user_id')['price']
#                .sum().reset_index()
#                .rename(columns={'price': 'lifetime_value'}))
#         results = results.merge(ltv, on='user_id', how='left')

#         # Recent views count
#         rv = (df_events[df_events['event_type'] == 'view']
#               .groupby('user_id').size()
#               .reset_index(name='recent_views'))
#         results = results.merge(rv, on='user_id', how='left')

#         return results


# # ─────────────────────────────────────────────────────────────
# #  Singleton accessor — used by dashboard_dpgcn.py
# # ─────────────────────────────────────────────────────────────

# _dpgcn_instance = None


# def get_dpgcn() -> DP_GCN_Integrator:
#     global _dpgcn_instance
#     if _dpgcn_instance is None:
#         _dpgcn_instance = DP_GCN_Integrator()
#         try:
#             _dpgcn_instance.model = DP_GCN(
#                 input_dim=8, hidden_dim=64, output_dim=32, num_classes=5
#             )
#             _dpgcn_instance.model.load_state_dict(
#                 torch.load('dpgcn_model.pt', map_location='cpu')
#             )
#             with open('dpgcn_scaler.pkl', 'rb') as f:
#                 _dpgcn_instance.graph_builder.scaler = pickle.load(f)
#             _dpgcn_instance.is_trained = True
#             print("✅ Loaded pre-trained DP-GCN model")
#         except Exception:
#             print("⚠️  No pre-trained model found — click Train DP-GCN in sidebar")
#     return _dpgcn_instance





# dpgcn_model.py  — Tier 2 extended version
# ─────────────────────────────────────────────────────────────
# What changed vs the original:
#   NEW — 80/20 train/validation split inside train()
#   NEW — per-epoch val accuracy + val loss tracked separately
#   NEW — per-class F1 score computed after training
#   NEW — full training run logged to SQLite (training_history table)
#   NEW — get_training_history() for dashboard accuracy trend chart
#   NEW — best-model checkpointing (saves weights at best val accuracy)
#   NEW — early stopping (stops if val loss doesn't improve for 10 epochs)
#   KEPT — everything original: C_GCN, T_GCN, MultiHeadAttention,
#           DP_GCN, CustomerGraphBuilder, pseudo-labels, predict_customers
# ─────────────────────────────────────────────────────────────


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
import networkx as nx
import sqlite3
import pickle
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DB_FILE = "customer_data.db"

# ─────────────────────────────────────────────────────────────
#  Neural network layers — unchanged from original
# ─────────────────────────────────────────────────────────────

class C_GCN(nn.Module):
    """Connectivity GCN — captures direct behavioural similarity edges."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


class T_GCN(nn.Module):
    """Topology GCN — captures structural role similarity via KMeans clusters."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class MultiHeadAttention(nn.Module):
    """4-head self-attention — fuses C-GCN and T-GCN representations."""
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, \
            "in_channels must be divisible by num_heads"
        self.query    = nn.Linear(in_channels, in_channels)
        self.key      = nn.Linear(in_channels, in_channels)
        self.value    = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, c_features, t_features):
        combined   = c_features + t_features
        bs, n, d   = combined.shape
        Q = self.query(combined).view(bs, n, self.num_heads, self.head_dim)
        K = self.key(combined).view(bs, n, self.num_heads, self.head_dim)
        V = self.value(combined).view(bs, n, self.num_heads, self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(bs, n, d)
        return self.out_proj(out)


class DP_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_classes=5):
        super().__init__()
        self.c_gcn     = C_GCN(input_dim, hidden_dim, output_dim)
        self.t_gcn     = T_GCN(input_dim, hidden_dim, output_dim)
        self.attention = MultiHeadAttention(output_dim, num_heads=4)

        # Direct feature path — bypasses graph entirely
        # Prevents embedding collapse when graph smoothing dominates
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, output_dim),  nn.ReLU()
        )

        # Classifier uses BOTH graph embeddings AND raw features
        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 2, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(output_dim * 2, 16), nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, c_edge_index, t_edge_index):
        c_feat   = self.c_gcn(x, c_edge_index)
        t_feat   = self.t_gcn(x, t_edge_index)
        combined = self.attention(
            c_feat.unsqueeze(0), t_feat.unsqueeze(0)
        ).squeeze(0)

        # Raw feature projection — residual path
        feat_proj = self.feature_proj(x)

        # Concatenate graph embedding + raw features
        full = torch.cat([combined, feat_proj], dim=1)

        seg_logits  = self.classifier(full)
        seg_probs   = F.softmax(seg_logits, dim=1)
        anom_scores = self.anomaly_head(full)
        return seg_probs, anom_scores


# ─────────────────────────────────────────────────────────────
#  Graph builder — unchanged from original
# ─────────────────────────────────────────────────────────────

class CustomerGraphBuilder:
    """Extracts 8 behavioural features per customer and builds both graphs."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            'total_spent', 'purchase_count', 'avg_order_value',
            'views_last_7d', 'carts_last_7d', 'days_since_last_purchase',
            'browsing_frequency', 'cart_abandonment_rate'
        ]

    def extract_features(self, df_events):
        """Vectorized feature extraction via pandas groupby — O(customers + events)
        instead of the original O(customers × events) per-user filter-and-scan loop,
        which was the dominant cost in 'Generate Dashboard' / 'Train DP-GCN' for any
        realistically sized upload (benchmarked at 8s+ for just 2,000 customers /
        44k events; scales quadratically from there)."""
        now = datetime.now()
        seven_days_ago = now - timedelta(days=7)

        df = df_events.copy()
        df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce', format='mixed')

        user_ids = df['user_id'].unique()

        purchases = df[df['event_type'] == 'purchase']
        views     = df[df['event_type'] == 'view']
        carts     = df[df['event_type'] == 'cart']
        recent_views_df = views[views['event_time'] > seven_days_ago]
        recent_carts_df = carts[carts['event_time'] > seven_days_ago]

        spend_by_user        = purchases.groupby('user_id')['price'].sum()
        pcount_by_user        = purchases.groupby('user_id').size()
        last_purchase_by_user = purchases.groupby('user_id')['event_time'].max()
        view_count_by_user    = views.groupby('user_id').size()
        cart_count_by_user    = carts.groupby('user_id').size()
        recent_view_by_user   = recent_views_df.groupby('user_id').size()
        recent_cart_by_user   = recent_carts_df.groupby('user_id').size()
        span_min_by_user      = df.groupby('user_id')['event_time'].min()
        span_max_by_user      = df.groupby('user_id')['event_time'].max()

        feat_df = pd.DataFrame(index=user_ids)
        feat_df['total_spent']    = spend_by_user.reindex(user_ids).fillna(0.0)
        feat_df['purchase_count'] = pcount_by_user.reindex(user_ids).fillna(0).astype(int)

        purchase_count_safe = feat_df['purchase_count'].replace(0, np.nan)
        feat_df['avg_order_value'] = (feat_df['total_spent'] / purchase_count_safe).fillna(0.0)

        feat_df['views_last_7d'] = recent_view_by_user.reindex(user_ids).fillna(0).astype(int)
        feat_df['carts_last_7d'] = recent_cart_by_user.reindex(user_ids).fillna(0).astype(int)

        last_purchase = last_purchase_by_user.reindex(user_ids)
        days_since = (now - last_purchase).dt.days
        feat_df['days_since_last_purchase'] = days_since.fillna(365)

        span_min = span_min_by_user.reindex(user_ids)
        span_max = span_max_by_user.reindex(user_ids)
        time_span = (span_max - span_min).dt.days.fillna(0).clip(lower=1)

        view_count = view_count_by_user.reindex(user_ids).fillna(0)
        cart_count = cart_count_by_user.reindex(user_ids).fillna(0)
        feat_df['browsing_frequency']    = view_count / time_span
        feat_df['cart_abandonment_rate'] = 1 - (feat_df['purchase_count'] / cart_count.clip(lower=1))

        features = feat_df[self.feature_names].to_numpy(dtype=float)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = self.scaler.fit_transform(features)
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return user_ids, features_scaled

    def build_connectivity_graph(self, user_ids, features):
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        sim    = cosine_similarity(features)
        edges  = []
        k      = min(5, len(user_ids) - 1)
        for i in range(len(user_ids)):
            top_k = np.argsort(sim[i])[::-1][1:k + 1]
            for j in top_k:
                edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def build_topology_graph(self, user_ids, features, max_pairs_per_cluster: int = 20000):
        """Vectorized within-cluster pair generation via np.triu_indices instead of
        a Python double-for-loop (same O(m^2) pairs per cluster, but generated in
        one numpy call instead of millions of individual list.append() calls — this
        is the difference between ~1s and tens of seconds once a cluster gets large).
        A per-cluster pair cap with random subsampling guards against a single huge
        cluster generating an unmanageable number of edges."""
        import numpy as np
        from sklearn.cluster import KMeans
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        n_clusters = min(10, len(user_ids) // 10 + 1)
        kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters   = kmeans.fit_predict(features)

        rng = np.random.RandomState(42)
        edge_chunks = []
        for cid in range(n_clusters):
            idx = np.where(clusters == cid)[0]
            m = len(idx)
            if m < 2:
                continue
            i_idx, j_idx = np.triu_indices(m, k=1)
            if len(i_idx) > max_pairs_per_cluster:
                sel = rng.choice(len(i_idx), max_pairs_per_cluster, replace=False)
                i_idx, j_idx = i_idx[sel], j_idx[sel]
            edge_chunks.append(np.stack([idx[i_idx], idx[j_idx]], axis=1))

        if not edge_chunks:
            return torch.tensor([[0], [0]], dtype=torch.long)
        edges = np.concatenate(edge_chunks, axis=0)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ─────────────────────────────────────────────────────────────
#  Training history — NEW in Tier 2
# ─────────────────────────────────────────────────────────────

def _ensure_history_table():
    """Create training_history table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE,timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            num_customers INTEGER,
            num_events    INTEGER,
            epochs        INTEGER,
            train_loss    REAL,
            val_loss      REAL,
            train_acc     REAL,
            val_acc       REAL,
            f1_macro      REAL,
            f1_per_class  TEXT,
            stopped_early INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def _log_training_run(num_customers, num_events, epochs_run,
                      train_loss, val_loss, train_acc, val_acc,
                      f1_macro, f1_per_class, stopped_early):
    """Append one training run record to SQLite."""
    _ensure_history_table()
    conn = sqlite3.connect(DB_FILE,timeout=30)
    conn.execute("""
        INSERT INTO training_history
            (num_customers, num_events, epochs, train_loss, val_loss,
             train_acc, val_acc, f1_macro, f1_per_class, stopped_early)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        num_customers, num_events, epochs_run,
        round(train_loss, 6), round(val_loss, 6),
        round(train_acc, 4),  round(val_acc, 4),
        round(f1_macro, 4),   json.dumps(f1_per_class),
        int(stopped_early)
    ))
    conn.commit()
    conn.close()


def get_training_history(limit: int = 50) -> pd.DataFrame:
    """
    Returns the last `limit` training runs as a DataFrame.
    Used by the dashboard to draw the accuracy trend chart.
    """
    _ensure_history_table()
    conn = sqlite3.connect(DB_FILE,timeout=30)
    df = pd.read_sql_query(f"""
        SELECT trained_at, num_customers, epochs,
               train_loss, val_loss,
               train_acc,  val_acc,
               f1_macro,   stopped_early
        FROM   training_history
        ORDER  BY id DESC
        LIMIT  {limit}
    """, conn)
    conn.close()
    return df.iloc[::-1].reset_index(drop=True)   # chronological order


# ─────────────────────────────────────────────────────────────
#  Main integrator — extended with Tier 2 additions
# ─────────────────────────────────────────────────────────────

class DP_GCN_Integrator:
    """
    Wraps the DP_GCN model with:
    — feature extraction & graph building (original)
    — 80/20 train/val split              (Tier 2 NEW)
    — per-epoch val tracking             (Tier 2 NEW)
    — early stopping (patience=10)       (Tier 2 NEW)
    — best-model checkpointing           (Tier 2 NEW)
    — F1 score per class                 (Tier 2 NEW)
    — SQLite training history logging    (Tier 2 NEW)
    """

    SEGMENT_MAP = {
        0: '👑 VIP_ACTIVE',
        1: '🔥 HIGH_POTENTIAL',
        2: '💤 DORMANT_VIP',
        3: '✅ LOYAL_REGULAR',
        4: '⚠️ AT_RISK',
    }
    SEGMENT_NAMES = ['VIP_ACTIVE', 'HIGH_POTENTIAL', 'DORMANT_VIP',
                     'LOYAL_REGULAR', 'AT_RISK']

    def __init__(self):
        self.model         = None
        self.graph_builder = CustomerGraphBuilder()
        self.user_ids      = None
        self.is_trained    = False
        # Last training metrics — exposed to dashboard
        self.last_val_acc   = 0.0
        self.last_val_loss  = 0.0
        self.last_f1_macro  = 0.0
        self.last_f1_class  = {}

    # ── Pseudo-label generation — FIXED (v2) in Tier 2 ─────────
    # v1 bug: when most users have $0 lifetime spend (common early
    # in simulation), quantile(0.25) and quantile(0.80) both collapse
    # to near-zero. That makes "low_spend_cutoff" almost equal to
    # "high_spend_cutoff", so the VIP/DORMANT_VIP branches become
    # unreachable and everyone with ANY browsing activity gets
    # mislabeled HIGH_POTENTIAL — even $500+ spenders.
    #
    # v2 fix: compute percentiles only over users who have actually
    # purchased something (spend > 0), so the tiers reflect real
    # spending behaviour instead of being dragged down by the many
    # zero-spend browsers. Zero-spend users are handled separately.
    def _generate_pseudo_labels(self, df_events):
        purchases_all = df_events[df_events['event_type'] == 'purchase']
        spend_by_user = purchases_all.groupby('user_id')['price'].sum()
        all_spend     = spend_by_user.reindex(self.user_ids, fill_value=0)

        views_all = df_events[df_events['event_type'] == 'view'].groupby('user_id').size()
        carts_all = df_events[df_events['event_type'] == 'cart'].groupby('user_id').size()
        total_events_user = df_events.groupby('user_id').size()

        nonzero = all_spend[all_spend > 0]
        n       = len(nonzero)

        if n >= 10:
            vip_cutoff = nonzero.quantile(0.80)
            mid_low    = nonzero.quantile(0.20)
            mid_high   = nonzero.quantile(0.80)
        elif n >= 2:
            vip_cutoff = nonzero.max() * 0.70
            mid_low    = nonzero.min()
            mid_high   = nonzero.max() * 0.70
        elif n == 1:
            vip_cutoff = nonzero.iloc[0]
            mid_low    = 0.01
            mid_high   = nonzero.iloc[0]
        else:
            vip_cutoff = mid_low = mid_high = float('inf')

        labels = []
        for uid in self.user_ids:
            total_spent  = float(all_spend.get(uid, 0.0))
            v_all        = int(views_all.get(uid, 0))
            c_all        = int(carts_all.get(uid, 0))
            n_events     = int(total_events_user.get(uid, 0))
            is_buyer     = total_spent > 0
            is_active    = (v_all + c_all) >= 2

            if   is_buyer and total_spent >= vip_cutoff and is_active:
                labels.append(0)   # VIP_ACTIVE
            elif is_buyer and total_spent >= vip_cutoff and not is_active:
                labels.append(2)   # DORMANT_VIP
            elif is_buyer and mid_low <= total_spent < mid_high:
                labels.append(3)   # LOYAL_REGULAR
            elif not is_buyer and n_events >= 3:
                labels.append(1)   # HIGH_POTENTIAL
            else:
                labels.append(4)   # AT_RISK
        return labels

    # ── Training — Tier 2 extended ────────────────────────────
    def train(self, df_events, epochs: int = 100, patience: int = 10,
              db_file: str = DB_FILE):
        """
        Train DP-GCN with 80/20 train/val split, early stopping,
        best-model checkpointing, and SQLite history logging.

        Parameters
        ----------
        df_events : pd.DataFrame   all events from SQLite
        epochs    : int            max training epochs
        patience  : int            early-stopping patience
        db_file   : str            SQLite path for history logging
        """
        print("🏗️  Building customer graphs...")
        self.user_ids, features = self.graph_builder.extract_features(df_events)
        n = len(self.user_ids)

        if n < 10:
            print("⚠️  Need at least 10 customers for training")
            return False

        # ── Build both graphs ──────────────────────────────────
        c_edge_index = self.graph_builder.build_connectivity_graph(self.user_ids, features)
        t_edge_index = self.graph_builder.build_topology_graph(self.user_ids, features)
        x            = torch.tensor(features, dtype=torch.float)

        # ── Pseudo-labels ──────────────────────────────────────
        all_labels = self._generate_pseudo_labels(df_events)
        all_labels = all_labels[:n]

        # ── 80/20 train / val split (NEW) ─────────────────────
        all_idx   = np.arange(n)
        from collections import Counter
        label_counts = Counter(all_labels)
        can_stratify = all(c >= 2 for c in label_counts.values()) and len(set(all_labels)) > 1
        train_idx, val_idx = train_test_split(
            all_idx, test_size=0.2, random_state=42,
            stratify=all_labels if can_stratify else None
        )
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask   = torch.zeros(n, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx]     = True

        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        at_risk_mask  = (labels_tensor == 4).float().unsqueeze(1)

        print(f"   Train: {train_mask.sum().item()} customers | "
              f"Val: {val_mask.sum().item()} customers")

        # ── Model & optimiser ──────────────────────────────────
        input_dim  = features.shape[1]
        self.model = DP_GCN(input_dim=input_dim, hidden_dim=64,
                    output_dim=32, num_classes=5)
        optimizer = torch.optim.Adam(self.model.parameters(),
                              lr=0.005, weight_decay=1e-5)
        # Class weights — penalise minority classes more heavily
        # so the model doesn't collapse to predicting HIGH_POTENTIAL
        class_weights = torch.tensor([5.0, 2.0, 5.0, 1.0, 3.0])
        criterion_seg  = nn.CrossEntropyLoss(weight=class_weights)
        criterion_anom = nn.BCELoss()

        # ── Early stopping & checkpointing state (NEW) ─────────
        best_val_loss   = float('inf')
        best_val_acc    = 0.0
        best_state      = None
        patience_count  = 0
        stopped_early   = False

        final_train_loss = 0.0
        final_val_loss   = 0.0
        final_train_acc  = 0.0
        final_val_acc    = 0.0
        epochs_run       = 0

        print(f"🚀 Training DP-GCN on {n} customers for up to {epochs} epochs...")

        for epoch in range(epochs):
            # ── Train step ──────────────────────────────────────
            self.model.train()
            optimizer.zero_grad()

            seg_probs, anom_scores = self.model(x, c_edge_index, t_edge_index)

            # Loss only on train nodes
            loss_seg  = criterion_seg(seg_probs[train_mask],
                                      labels_tensor[train_mask])
            loss_anom = criterion_anom(anom_scores[train_mask],
                                       at_risk_mask[train_mask])
            total_loss = loss_seg + 0.5 * loss_anom
            total_loss.backward()
            optimizer.step()

            train_acc = (
                seg_probs[train_mask].argmax(dim=1) == labels_tensor[train_mask]
            ).float().mean().item()

            # ── Validation step (NEW) ───────────────────────────
            self.model.eval()
            with torch.no_grad():
                seg_probs_v, anom_scores_v = self.model(x, c_edge_index, t_edge_index)
                val_loss_seg  = criterion_seg(seg_probs_v[val_mask],
                                              labels_tensor[val_mask])
                val_loss_anom = criterion_anom(anom_scores_v[val_mask],
                                               at_risk_mask[val_mask])
                val_loss = (val_loss_seg + 0.5 * val_loss_anom).item()
                val_acc  = (
                    seg_probs_v[val_mask].argmax(dim=1) == labels_tensor[val_mask]
                ).float().mean().item()

            epochs_run       = epoch + 1
            final_train_loss = total_loss.item()
            final_val_loss   = val_loss
            final_train_acc  = train_acc
            final_val_acc    = val_acc

            # ── Best checkpoint (NEW) ───────────────────────────
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_val_acc   = val_acc
                best_state     = {k: v.clone() for k, v in
                                  self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs} | "
                      f"Train loss: {total_loss.item():.4f} acc: {train_acc:.3f} | "
                      f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}")

            # ── Early stopping (NEW) ────────────────────────────
            if patience_count >= patience:
                print(f"   ⏹️  Early stopping at epoch {epoch+1} "
                      f"(no val improvement for {patience} epochs)")
                stopped_early = True
                break

        # ── Restore best weights (NEW) ─────────────────────────
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"   ✅ Restored best weights "
                  f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.3f})")

        # ── F1 score per class (NEW) ───────────────────────────
        self.model.eval()
        with torch.no_grad():
            seg_probs_f, _ = self.model(x, c_edge_index, t_edge_index)
            preds  = seg_probs_f.argmax(dim=1).numpy()
            truths = labels_tensor.numpy()

        f1_macro = f1_score(truths, preds, average='macro', zero_division=0)
        f1_raw   = f1_score(truths, preds, average=None, zero_division=0)
        f1_each = list(f1_raw) + [0.0] * (5 - len(f1_raw))
        f1_class = {
            'VIP_ACTIVE':     round(f1_each[0], 4),
            'HIGH_POTENTIAL': round(f1_each[1], 4),
            'DORMANT_VIP':    round(f1_each[2], 4),
            'LOYAL_REGULAR':  round(f1_each[3], 4),
            'AT_RISK':        round(f1_each[4], 4),
        }
        

        print(f"   📊 F1 macro: {f1_macro:.4f}")
        for seg, score in f1_class.items():
            print(f"      {seg}: {score:.4f}")

        # ── Save model & scaler ────────────────────────────────
        torch.save(self.model.state_dict(), 'dpgcn_model.pt')
        with open('dpgcn_scaler.pkl', 'wb') as f:
            pickle.dump(self.graph_builder.scaler, f)

        # ── Expose metrics to dashboard ────────────────────────
        self.last_val_acc  = best_val_acc
        self.last_val_loss = best_val_loss
        self.last_f1_macro = f1_macro
        self.last_f1_class = f1_class
        self.is_trained    = True

        # ── Log to SQLite training_history (NEW) ───────────────
        _log_training_run(
            num_customers = n,
            num_events    = len(df_events),
            epochs_run    = epochs_run,
            train_loss    = final_train_loss,
            val_loss      = best_val_loss,
            train_acc     = final_train_acc,
            val_acc       = best_val_acc,
            f1_macro      = f1_macro,
            f1_per_class  = f1_class,
            stopped_early = stopped_early
        )
        print(f"✅ Training complete — run logged to {db_file}")
        return True

    # ── Inference — unchanged from original ───────────────────
    def predict_customers(self, df_events):
        """Run DP-GCN inference on all customers. Returns a DataFrame."""
        if not self.is_trained:
            return pd.DataFrame()

        self.user_ids, features = self.graph_builder.extract_features(df_events)
        if len(self.user_ids) == 0:
            return pd.DataFrame()

        c_edge_index = self.graph_builder.build_connectivity_graph(
            self.user_ids, features)
        t_edge_index = self.graph_builder.build_topology_graph(
            self.user_ids, features)

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float)
            seg_probs, anom_scores = self.model(x, c_edge_index, t_edge_index)
            segments   = seg_probs.argmax(dim=1).numpy()
            anomaly    = anom_scores.numpy().flatten()

        results = pd.DataFrame({
            'user_id':      self.user_ids,
            'segment':      [self.SEGMENT_MAP[s] for s in segments],
            'anomaly_score': anomaly,
            'confidence':   seg_probs.max(dim=1)[0].numpy(),
        })

        # Lifetime value
        ltv = (df_events.groupby('user_id')['price']
               .sum().reset_index()
               .rename(columns={'price': 'lifetime_value'}))
        results = results.merge(ltv, on='user_id', how='left')

        # Recent views count
        rv = (df_events[df_events['event_type'] == 'view']
              .groupby('user_id').size()
              .reset_index(name='recent_views'))
        results = results.merge(rv, on='user_id', how='left')

        return results


# ─────────────────────────────────────────────────────────────
#  Singleton accessor — used by dashboard_dpgcn.py
# ─────────────────────────────────────────────────────────────

_dpgcn_instance = None


def get_dpgcn() -> DP_GCN_Integrator:
    global _dpgcn_instance
    if _dpgcn_instance is None:
        _dpgcn_instance = DP_GCN_Integrator()
        try:
            _dpgcn_instance.model = DP_GCN(
                input_dim=8, hidden_dim=128, output_dim=64, num_classes=5
            )
            _dpgcn_instance.model.load_state_dict(
                torch.load('dpgcn_model.pt', map_location='cpu')
            )
            with open('dpgcn_scaler.pkl', 'rb') as f:
                _dpgcn_instance.graph_builder.scaler = pickle.load(f)
            _dpgcn_instance.is_trained = True
            print("✅ Loaded pre-trained DP-GCN model")
        except Exception:
            print("⚠️  No pre-trained model found — click Train DP-GCN in sidebar")
    return _dpgcn_instance
