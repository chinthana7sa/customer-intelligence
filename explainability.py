# explainability.py  — Tier 3 new file
# ─────────────────────────────────────────────────────────────
# What this does:
#   Provides human-readable explanations for every DP-GCN
#   customer segment prediction. Two complementary approaches:
#
#   1. FEATURE ATTRIBUTION
#      Computes which of the 8 behavioural features most
#      influenced the segment prediction for a given customer.
#      Uses gradient-based saliency — the gradient of the
#      predicted class score w.r.t. each input feature tells
#      us how sensitive the prediction is to that feature.
#      Higher gradient magnitude = more influential feature.
#
#   2. NEIGHBOUR INFLUENCE
#      Shows which graph neighbours most influenced this
#      customer's prediction. A customer surrounded by VIPs
#      gets a VIP signal from the graph; one surrounded by
#      churners gets an AT_RISK signal.
#
#   3. NATURAL LANGUAGE EXPLANATION
#      Converts feature attributions into plain English:
#      "This customer is VIP_ACTIVE because:
#       (1) High lifetime spend ($3,200) — primary driver
#       (2) Low cart abandonment (0.12) — confirms purchase intent
#       (3) Recent browsing activity (14 views last 7 days)"
#
#   Used in dashboard_dpgcn.py as an "Explain Prediction" panel.
# ─────────────────────────────────────────────────────────────


import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# ── Feature metadata ──────────────────────────────────────────
FEATURE_NAMES = [
    'total_spent',
    'purchase_count',
    'avg_order_value',
    'views_last_7d',
    'carts_last_7d',
    'days_since_last_purchase',
    'browsing_frequency',
    'cart_abandonment_rate',
]

FEATURE_LABELS = {
    'total_spent':               'Lifetime spend',
    'purchase_count':            'Number of purchases',
    'avg_order_value':           'Average order value',
    'views_last_7d':             'Views in last 7 days',
    'carts_last_7d':             'Cart additions last 7 days',
    'days_since_last_purchase':  'Days since last purchase',
    'browsing_frequency':        'Browsing frequency',
    'cart_abandonment_rate':     'Cart abandonment rate',
}

FEATURE_DIRECTIONS = {
    # +1 = higher value → more positive (good for retention)
    # -1 = higher value → more negative (bad for retention)
    'total_spent':               +1,
    'purchase_count':            +1,
    'avg_order_value':           +1,
    'views_last_7d':             +1,
    'carts_last_7d':             +1,
    'days_since_last_purchase':  -1,   # more days = worse
    'browsing_frequency':        +1,
    'cart_abandonment_rate':     -1,   # higher abandonment = worse
}

SEGMENT_COLORS = {
    'VIP_ACTIVE':     '#FFD700',
    'HIGH_POTENTIAL': '#00CED1',
    'DORMANT_VIP':    '#9370DB',
    'LOYAL_REGULAR':  '#32CD32',
    'AT_RISK':        '#FF6347',
    'UNKNOWN':        '#888888',
}

SEGMENT_EXPLANATIONS = {
    'VIP_ACTIVE': {
        'summary': 'High-value customer actively engaging',
        'key_signals': ['High lifetime spend', 'Recent purchase activity', 'Low abandonment rate'],
        'action': 'Offer loyalty rewards and early access to new products',
        'urgency': 'LOW',
    },
    'HIGH_POTENTIAL': {
        'summary': 'Heavy browser who has not yet purchased',
        'key_signals': ['High browsing frequency', 'Multiple cart additions', 'Zero purchases'],
        'action': 'Send first-purchase incentive — discount or free shipping',
        'urgency': 'HIGH',
    },
    'DORMANT_VIP': {
        'summary': 'Previously high-value, now inactive',
        'key_signals': ['High historical spend', 'Long inactivity period', 'Low recent views'],
        'action': 'Win-back campaign with personalised offer based on past purchases',
        'urgency': 'HIGH',
    },
    'LOYAL_REGULAR': {
        'summary': 'Consistent mid-tier customer',
        'key_signals': ['Steady purchase history', 'Regular browsing', 'Moderate spend'],
        'action': 'Upsell to premium tier with bundle offers',
        'urgency': 'MEDIUM',
    },
    'AT_RISK': {
        'summary': 'Declining engagement — churn likely',
        'key_signals': ['Reducing activity', 'High days since last purchase', 'Low spend'],
        'action': 'Immediate retention outreach — discount or survey',
        'urgency': 'CRITICAL',
    },
}


def get_segment_key(segment_str: str) -> str:
    """Strip emoji prefix to get clean segment key."""
    clean_map = {
        'VIP_ACTIVE':     'VIP_ACTIVE',
        'HIGH_POTENTIAL': 'HIGH_POTENTIAL',
        'DORMANT_VIP':    'DORMANT_VIP',
        'LOYAL_REGULAR':  'LOYAL_REGULAR',
        'AT_RISK':        'AT_RISK',
    }
    for key in clean_map:
        if key in str(segment_str):
            return key
    return 'UNKNOWN'


def compute_feature_attribution(
    model,
    x: torch.Tensor,
    c_edge_index: torch.Tensor,
    t_edge_index: torch.Tensor,
    node_idx: int,
    predicted_class: int
) -> np.ndarray:
    """
    Gradient-based feature attribution for a single customer node.

    Computes: d(score_predicted_class) / d(x_node_features)
    The magnitude of each gradient component shows how sensitive
    the prediction is to that feature.

    Parameters
    ----------
    model           : trained DP_GCN model
    x               : full node feature tensor [N, 8]
    c_edge_index    : C-GCN edge index
    t_edge_index    : T-GCN edge index
    node_idx        : index of the customer to explain
    predicted_class : the class predicted for this customer

    Returns
    -------
    np.ndarray of shape [8] — attribution score per feature
    """
    model.eval()

    # Enable gradients on input
    x_grad = x.clone().detach().requires_grad_(True)

    try:
        seg_probs, _ = model(x_grad, c_edge_index, t_edge_index)
        target_score = seg_probs[node_idx, predicted_class]
        target_score.backward()

        if x_grad.grad is not None:
            # Attribution = absolute gradient at this node's features
            attribution = x_grad.grad[node_idx].abs().detach().numpy()
        else:
            attribution = np.ones(len(FEATURE_NAMES))

    except Exception:
        # Fallback: uniform attribution
        attribution = np.ones(len(FEATURE_NAMES))

    # Normalise to [0, 1]
    total = attribution.sum()
    if total > 0:
        attribution = attribution / total

    return attribution


def get_feature_values_for_node(
    user_id: str,
    df_events: pd.DataFrame
) -> dict:
    """Get raw feature values for a customer."""
    from datetime import datetime, timedelta

    ud        = df_events[df_events['user_id'] == user_id]
    purchases = ud[ud['event_type'] == 'purchase']
    views     = ud[ud['event_type'] == 'view']
    carts     = ud[ud['event_type'] == 'cart']

    total_spent     = float(purchases['price'].sum()) if not purchases.empty else 0.0
    purchase_count  = len(purchases)
    avg_order_value = total_spent / purchase_count if purchase_count > 0 else 0.0

    seven_days_ago = datetime.now() - timedelta(days=7)
    recent_views   = 0
    recent_carts   = 0
    days_since     = 365
    browsing_freq  = 0.0
    cart_abandonment = 0.0

    try:
        ev_times     = pd.to_datetime(ud['event_time'], format='mixed', errors='coerce')
        recent_mask  = ev_times > seven_days_ago
        recent_views = int(ud[recent_mask & (ud['event_type'] == 'view')].shape[0])
        recent_carts = int(ud[recent_mask & (ud['event_type'] == 'cart')].shape[0])
    except Exception:
        pass

    try:
        last_purchase = pd.to_datetime(
            purchases['event_time'], format='mixed', errors='coerce'
        ).max()
        if pd.notna(last_purchase):
            days_since = (datetime.now() - last_purchase).days
    except Exception:
        pass

    try:
        ev_times_all = pd.to_datetime(ud['event_time'], format='mixed', errors='coerce')
        time_span    = max(1, (ev_times_all.max() - ev_times_all.min()).days)
        browsing_freq = float(len(views)) / float(time_span)
        if np.isnan(browsing_freq) or np.isinf(browsing_freq):
            browsing_freq = 0.0
    except Exception:
        browsing_freq = 0.0

    try:
        cart_abandonment = 1.0 - (float(purchase_count) / float(max(1, len(carts))))
        if np.isnan(cart_abandonment) or np.isinf(cart_abandonment):
            cart_abandonment = 0.0
    except Exception:
        cart_abandonment = 0.0

    return {
        'total_spent':              round(total_spent, 2),
        'purchase_count':           purchase_count,
        'avg_order_value':          round(avg_order_value, 2),
        'views_last_7d':            recent_views,
        'carts_last_7d':            recent_carts,
        'days_since_last_purchase': days_since,
        'browsing_frequency':       round(browsing_freq, 2),
        'cart_abandonment_rate':    round(cart_abandonment, 2),
    }


def format_feature_value(feature_name: str, value: float) -> str:
    """Format a feature value for display in explanation text."""
    if feature_name == 'total_spent':
        return f'${value:,.2f}'
    elif feature_name == 'avg_order_value':
        return f'${value:,.2f}'
    elif feature_name == 'cart_abandonment_rate':
        return f'{value:.0%}'
    elif feature_name == 'browsing_frequency':
        return f'{value:.1f}/day'
    elif feature_name == 'days_since_last_purchase':
        return f'{int(value)} days'
    elif feature_name in ('purchase_count', 'views_last_7d',
                          'carts_last_7d'):
        return str(int(value))
    return str(value)


def generate_explanation(
    user_id: str,
    segment: str,
    attribution: np.ndarray,
    feature_values: dict,
    anomaly_score: float,
    confidence: float,
    neighbour_segments: Optional[list] = None
) -> dict:
    """
    Generate a complete human-readable explanation for one customer.

    Returns a dict with all components needed to render the
    explanation panel in the dashboard.
    """
    seg_key  = get_segment_key(segment)
    seg_meta = SEGMENT_EXPLANATIONS.get(seg_key, {})

    # ── Top 3 driving features ────────────────────────────────
    sorted_idx    = np.argsort(attribution)[::-1]
    top3_features = []
    for rank, idx in enumerate(sorted_idx[:3]):
        fname   = FEATURE_NAMES[idx]
        fval    = feature_values.get(fname, 0)
        fattr   = float(attribution[idx])
        flabel  = FEATURE_LABELS.get(fname, fname)
        fvalstr = format_feature_value(fname, fval)
        top3_features.append({
            'rank':        rank + 1,
            'name':        fname,
            'label':       flabel,
            'value':       fval,
            'value_str':   fvalstr,
            'attribution': round(fattr * 100, 1),
            'direction':   FEATURE_DIRECTIONS.get(fname, 1),
        })

    # ── Natural language reason ───────────────────────────────
    reasons = []
    for f in top3_features:
        direction_word = (
            'High' if f['direction'] == 1 and f['value'] > 0
            else 'Low' if f['direction'] == -1
            else 'Notable'
        )
        reasons.append(
            f"({f['rank']}) {direction_word} {f['label'].lower()} "
            f"({f['value_str']}) — {f['attribution']}% influence"
        )
    nl_reason = '\n'.join(reasons)

    # ── Neighbour influence summary ───────────────────────────
    neighbour_summary = None
    if neighbour_segments:
        from collections import Counter
        counts = Counter(neighbour_segments)
        most_common = counts.most_common(2)
        neighbour_summary = ', '.join(
            [f"{get_segment_key(s)} ({c})" for s, c in most_common]
        )

    # ── Urgency colour ────────────────────────────────────────
    urgency_colors = {
        'CRITICAL': '#FF4444',
        'HIGH':     '#FF8C00',
        'MEDIUM':   '#FFD700',
        'LOW':      '#32CD32',
    }
    urgency     = seg_meta.get('urgency', 'MEDIUM')
    urgency_col = urgency_colors.get(urgency, '#888888')

    return {
        'user_id':            user_id,
        'segment':            seg_key,
        'segment_color':      SEGMENT_COLORS.get(seg_key, '#888'),
        'summary':            seg_meta.get('summary', ''),
        'confidence':         round(confidence * 100, 1),
        'anomaly_score':      round(anomaly_score * 100, 1),
        'action':             seg_meta.get('action', ''),
        'urgency':            urgency,
        'urgency_color':      urgency_col,
        'top_features':       top3_features,
        'natural_language':   nl_reason,
        'neighbour_summary':  neighbour_summary,
        'all_attributions':   {
            FEATURE_NAMES[i]: round(float(attribution[i]) * 100, 1)
            for i in range(len(FEATURE_NAMES))
        },
    }


def explain_customer(
    user_id: str,
    dpgcn_integrator,
    df_events: pd.DataFrame,
    predictions_df: pd.DataFrame
) -> Optional[dict]:
    """
    Main entry point — explain one customer's DP-GCN prediction.

    Parameters
    ----------
    user_id            : the customer UUID to explain
    dpgcn_integrator   : DP_GCN_Integrator instance (from get_dpgcn())
    df_events          : full events DataFrame from SQLite
    predictions_df     : output of dpgcn.predict_customers()

    Returns
    -------
    dict with full explanation, or None if user not found
    """
    if (dpgcn_integrator is None
            or not dpgcn_integrator.is_trained
            or dpgcn_integrator.model is None):
        return None

    # Find this user in predictions
    pred_row = predictions_df[predictions_df['user_id'] == user_id]
    if pred_row.empty:
        return None

    pred_row     = pred_row.iloc[0]
    segment      = str(pred_row.get('segment', 'UNKNOWN'))
    anomaly      = float(pred_row.get('anomaly_score', 0))
    confidence   = float(pred_row.get('confidence', 0))

    # Find node index in the trained graph
    if dpgcn_integrator.user_ids is None:
        return None

    user_ids_list = list(dpgcn_integrator.user_ids)
    user_ids_str  = [str(u) for u in user_ids_list]
    user_id_str   = str(user_id)
    if user_id_str not in user_ids_str:
        return None

    node_idx = user_ids_str.index(user_id_str)

    # Rebuild features and graphs using only trained users
    try:
        df_filtered = df_events[df_events['user_id'].isin(user_ids_list)].copy()
        returned_ids, features = dpgcn_integrator.graph_builder.extract_features(df_filtered)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Reorder features to match trained user order exactly
        returned_ids_str = [str(u) for u in returned_ids]
        reorder_idx = [returned_ids_str.index(uid) if uid in returned_ids_str else 0
                       for uid in user_ids_str]
        features = features[reorder_idx]

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        features = features.astype(np.float32)



        c_edge = dpgcn_integrator.graph_builder.build_connectivity_graph(
            dpgcn_integrator.user_ids, features)
        t_edge = dpgcn_integrator.graph_builder.build_topology_graph(
            dpgcn_integrator.user_ids, features)
        x = torch.tensor(features, dtype=torch.float)
    except Exception as e:
        print(f'Explainability graph build failed: {e}')
        return None

    # Predicted class index
    seg_key_map = {v: k for k, v in dpgcn_integrator.SEGMENT_MAP.items()}
    seg_key     = get_segment_key(segment)
    seg_name_to_idx = {
        'VIP_ACTIVE': 0, 'HIGH_POTENTIAL': 1, 'DORMANT_VIP': 2,
        'LOYAL_REGULAR': 3, 'AT_RISK': 4
    }
    predicted_class = seg_name_to_idx.get(seg_key, 4)

    # Compute feature attributions
    attribution = compute_feature_attribution(
        dpgcn_integrator.model, x, c_edge, t_edge,
        node_idx, predicted_class
    )

    # Raw feature values for display
    feature_values = get_feature_values_for_node(user_id, df_events)

    # Neighbour segments from C-GCN graph
    neighbour_segments = []
    try:
        edge_arr = c_edge.numpy()
        mask     = edge_arr[0] == node_idx
        nb_idx   = edge_arr[1][mask]

        dpgcn_integrator.model.eval()
        with torch.no_grad():
            seg_probs, _ = dpgcn_integrator.model(x, c_edge, t_edge)
            nb_preds = seg_probs[nb_idx].argmax(dim=1).numpy()
            neighbour_segments = [
                dpgcn_integrator.SEGMENT_MAP.get(int(p), 'UNKNOWN')
                for p in nb_preds
            ]
    except Exception:
        pass

    return generate_explanation(
        user_id         = user_id,
        segment         = segment,
        attribution     = attribution,
        feature_values  = feature_values,
        anomaly_score   = anomaly,
        confidence      = confidence,
        neighbour_segments = neighbour_segments
    )


def explain_top_customers(
    dpgcn_integrator,
    df_events: pd.DataFrame,
    predictions_df: pd.DataFrame,
    n: int = 5,
    segment_filter: Optional[str] = None
) -> list:
    """
    Explain the top N customers by anomaly score (or per segment).
    Used to populate the bulk explanation panel in the dashboard.

    Parameters
    ----------
    n               : number of customers to explain
    segment_filter  : if set, only explain customers in this segment

    Returns
    -------
    list of explanation dicts
    """
    if predictions_df is None or predictions_df.empty:
        return []

    df = predictions_df.copy()

    if segment_filter:
        df = df[df['segment'].str.contains(segment_filter, na=False)]

    # Prioritise high anomaly scores
    df = df.sort_values('anomaly_score', ascending=False).head(n)

    explanations = []
    for _, row in df.iterrows():
        exp = explain_customer(
            user_id          = row['user_id'],
            dpgcn_integrator = dpgcn_integrator,
            df_events        = df_events,
            predictions_df   = predictions_df
        )
        if exp:
            explanations.append(exp)

    return explanations


def render_explanation_html(explanation: dict) -> str:
    """
    Render one customer explanation as a styled HTML card.
    Used with st.markdown(..., unsafe_allow_html=True) in dashboard.
    """
    if not explanation:
        return '<p style="color:#888">No explanation available.</p>'

    seg_color    = explanation['segment_color']
    urgency_col  = explanation['urgency_color']
    uid_short    = explanation['user_id'][:8] + '...'

    # Feature bar chart (horizontal bars)
    bars_html = ''
    for f in explanation['top_features']:
        bar_color = '#32CD32' if f['direction'] == 1 else '#FF6347'
        bar_width = max(5, f['attribution'])
        bars_html += f"""
        <div style="margin:6px 0">
          <div style="display:flex;justify-content:space-between;
                      font-size:11px;color:#ccc;margin-bottom:3px">
            <span>{f['label']}</span>
            <span style="color:{bar_color}">{f['value_str']} &nbsp;
              <strong>{f['attribution']}%</strong></span>
          </div>
          <div style="background:#222;border-radius:4px;height:6px">
            <div style="width:{bar_width}%;background:{bar_color};
                        border-radius:4px;height:6px;
                        transition:width 0.5s"></div>
          </div>
        </div>
        """

    neighbour_html = ''
    if explanation.get('neighbour_summary'):
        neighbour_html = f"""
        <div style="margin-top:12px;padding:8px 12px;
                    background:#1a1a2e;border-radius:6px;
                    font-size:11px;color:#aaa">
          <strong style="color:#888">Graph neighbours:</strong>
          {explanation['neighbour_summary']}
        </div>
        """

    return f"""
<div style="background:#111827;border:1px solid #2d3748;
            border-radius:12px;padding:20px;margin:8px 0;
            border-left:4px solid {seg_color}">

  <div style="display:flex;justify-content:space-between;
              align-items:flex-start;margin-bottom:14px">
    <div>
      <div style="font-size:13px;color:#888;margin-bottom:4px">
        Customer {uid_short}
      </div>
      <div style="font-size:18px;font-weight:700;
                  color:{seg_color}">{explanation['segment']}</div>
      <div style="font-size:12px;color:#aaa;margin-top:4px">
        {explanation['summary']}
      </div>
    </div>
    <div style="text-align:right">
      <div style="background:{urgency_col}22;border:1px solid {urgency_col};
                  border-radius:6px;padding:4px 10px;
                  color:{urgency_col};font-size:11px;font-weight:600">
        {explanation['urgency']} URGENCY
      </div>
      <div style="font-size:11px;color:#888;margin-top:6px">
        Confidence: <strong style="color:#fff">{explanation['confidence']}%</strong>
      </div>
      <div style="font-size:11px;color:#888">
        Churn Risk: <strong style="color:#FF6347">{explanation['anomaly_score']}%</strong>
      </div>
    </div>
  </div>

  <div style="margin-bottom:14px">
    <div style="font-size:12px;font-weight:600;color:#888;
                text-transform:uppercase;letter-spacing:0.5px;
                margin-bottom:8px">Top driving features</div>
    {bars_html}
  </div>

  <div style="background:#0f172a;border-radius:8px;padding:12px;
              font-size:12px;color:#94a3b8;margin-bottom:12px;
              border-left:3px solid {seg_color}">
    <strong style="color:{seg_color}">Why this segment:</strong><br>
    <span style="white-space:pre-line">{explanation['natural_language']}</span>
  </div>

  <div style="background:#1a2744;border-radius:8px;padding:10px 14px;
              font-size:12px">
    <strong style="color:#60a5fa">Recommended action:</strong>
    <span style="color:#cbd5e1"> {explanation['action']}</span>
  </div>

  {neighbour_html}
</div>
"""