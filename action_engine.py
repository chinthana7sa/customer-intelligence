# action_engine.py  — Tier 3 new file
# ─────────────────────────────────────────────────────────────
# What this does:
#   Generates personalised retention actions for every customer
#   based on their DP-GCN predicted segment and anomaly score.
#
#   For each customer it produces:
#   1. URGENCY SCORE     — 0.0 to 1.0, combines segment risk
#                          and anomaly churn probability
#   2. RECOMMENDED CHANNEL — email / push / SMS
#                          chosen based on customer value and urgency
#   3. DISCOUNT OFFER    — percentage discount calibrated to
#                          segment and lifetime value
#   4. MESSAGE TEMPLATE  — personalised message text ready to send
#   5. CAMPAIGN TYPE     — retention / winback / upsell / reward
#
#   Output:
#   - Returns a DataFrame with one row per customer
#   - Renders as a styled table in the dashboard
#   - Exports as downloadable CSV
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from datetime import datetime
import io

# ── Campaign configuration per segment ───────────────────────
SEGMENT_CAMPAIGNS = {
    'VIP_ACTIVE': {
        'campaign_type':  'reward',
        'base_urgency':   0.2,
        'channel':        'email',
        'discount_range': (5, 15),
        'message_template': (
            "Dear Valued Customer, as one of our top customers "
            "you have exclusive access to our VIP rewards program. "
            "Enjoy {discount}% off your next purchase as a thank you "
            "for your continued loyalty."
        ),
        'subject': 'Your Exclusive VIP Reward Is Ready',
        'action_label': 'Send VIP Reward',
        'color': '#FFD700',
    },
    'HIGH_POTENTIAL': {
        'campaign_type':  'conversion',
        'base_urgency':   0.6,
        'channel':        'push',
        'discount_range': (15, 25),
        'message_template': (
            "Hi there! We noticed you have been exploring our products. "
            "Here is a special first-purchase offer just for you — "
            "get {discount}% off when you complete your first order today. "
            "Your cart is waiting!"
        ),
        'subject': 'Complete Your First Purchase — Special Offer Inside',
        'action_label': 'Send First Purchase Offer',
        'color': '#00CED1',
    },
    'DORMANT_VIP': {
        'campaign_type':  'winback',
        'base_urgency':   0.8,
        'channel':        'email',
        'discount_range': (20, 30),
        'message_template': (
            "We miss you! You were one of our most valued customers "
            "and we would love to welcome you back. "
            "As a special thank you, here is {discount}% off your next purchase — "
            "no minimum spend required. Come back and see what is new!"
        ),
        'subject': 'We Miss You — Here Is {discount}% Off To Welcome You Back',
        'action_label': 'Send Win-Back Campaign',
        'color': '#9370DB',
    },
    'LOYAL_REGULAR': {
        'campaign_type':  'upsell',
        'base_urgency':   0.3,
        'channel':        'email',
        'discount_range': (10, 20),
        'message_template': (
            "Thank you for being a loyal customer! "
            "Based on your purchase history, we think you would love "
            "our premium collection. "
            "Enjoy {discount}% off as a loyalty reward — "
            "upgrade your experience today."
        ),
        'subject': 'Loyalty Reward — {discount}% Off Premium Products',
        'action_label': 'Send Loyalty Upsell',
        'color': '#32CD32',
    },
    'AT_RISK': {
        'campaign_type':  'retention',
        'base_urgency':   0.9,
        'channel':        'sms',
        'discount_range': (25, 40),
        'message_template': (
            "Hey! We noticed you have not visited in a while and we do not "
            "want to lose you. Here is {discount}% off — our biggest offer yet — "
            "because you matter to us. "
            "Tap to shop now before this offer expires."
        ),
        'subject': 'We Do Not Want To Lose You — {discount}% Off Inside',
        'action_label': 'Send Retention Offer',
        'color': '#FF6347',
    },
}

URGENCY_LABELS = {
    (0.0, 0.3): 'LOW',
    (0.3, 0.6): 'MEDIUM',
    (0.6, 0.8): 'HIGH',
    (0.8, 1.0): 'CRITICAL',
}

CHANNEL_ICONS = {
    'email':           '📧',
    'push':            '📱',
    'sms':             '💬',
}


def get_segment_key(segment_str: str) -> str:
    """Strip emoji prefix to get clean segment key."""
    s = str(segment_str).upper()
    if 'VIP_ACTIVE' in s and 'DORMANT' not in s:
        return 'VIP_ACTIVE'
    elif 'HIGH_POTENTIAL' in s:
        return 'HIGH_POTENTIAL'
    elif 'DORMANT_VIP' in s:
        return 'DORMANT_VIP'
    elif 'LOYAL_REGULAR' in s:
        return 'LOYAL_REGULAR'
    elif 'AT_RISK' in s:
        return 'AT_RISK'
    return 'AT_RISK'   # default to AT_RISK for unknowns


def compute_urgency(segment_key: str, anomaly_score: float,
                    lifetime_value: float) -> float:
    """
    Compute urgency score 0.0–1.0 for one customer.

    Combines:
    - Segment base urgency (from SEGMENT_CAMPAIGNS config)
    - Anomaly/churn score from DP-GCN anomaly head
    - Lifetime value weight (high-value customers get higher urgency)

    Formula:
        urgency = 0.5 * base + 0.4 * anomaly + 0.1 * ltv_weight
    """
    cfg         = SEGMENT_CAMPAIGNS.get(segment_key, SEGMENT_CAMPAIGNS['AT_RISK'])
    base        = cfg['base_urgency']
    anom        = float(anomaly_score or 0)

    # LTV weight: normalise to 0–1 assuming max reasonable LTV = $5000
    ltv_weight  = min(1.0, float(lifetime_value or 0) / 5000.0)

    urgency = (0.5 * base) + (0.4 * anom) + (0.1 * ltv_weight)
    return round(min(1.0, urgency), 4)


def get_urgency_label(urgency: float) -> str:
    """Convert urgency float to label string."""
    for (low, high), label in URGENCY_LABELS.items():
        if low <= urgency < high:
            return label
    return 'CRITICAL'


def compute_discount(segment_key: str, anomaly_score: float,
                     lifetime_value: float) -> int:
    """
    Compute personalised discount percentage.

    Higher anomaly = higher discount needed to retain.
    Higher LTV = lower discount needed (they're already engaged).
    """
    cfg   = SEGMENT_CAMPAIGNS.get(segment_key, SEGMENT_CAMPAIGNS['AT_RISK'])
    low, high = cfg['discount_range']

    anom  = float(anomaly_score or 0)
    ltv   = float(lifetime_value or 0)

    # High churn risk → push toward high end of range
    churn_pull = anom  # 0 to 1

    # High LTV → pull toward low end (don't over-discount loyal customers)
    ltv_pull   = min(1.0, ltv / 3000.0)

    raw = low + (high - low) * churn_pull * (1 - 0.3 * ltv_pull)
    return int(round(max(low, min(high, raw))))


def generate_message(segment_key: str, discount: int,
                     customer_id: str) -> str:
    """Fill in message template with customer-specific values."""
    cfg      = SEGMENT_CAMPAIGNS.get(segment_key, SEGMENT_CAMPAIGNS['AT_RISK'])
    template = cfg['message_template']
    return template.format(discount=discount, customer_id=customer_id[:8])


def select_channel(segment_key: str, urgency: float,
                   lifetime_value: float) -> str:
    """
    Override channel selection based on urgency and LTV.

    Low urgency → downgrade to email (less intrusive).
    """
    cfg     = SEGMENT_CAMPAIGNS.get(segment_key, SEGMENT_CAMPAIGNS['AT_RISK'])
    channel = cfg['channel']

    # Downgrade if low urgency
    if urgency < 0.25:
        channel = 'email'

    return channel


def generate_actions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate retention actions for all customers.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output of dpgcn.predict_customers() with columns:
        user_id, segment, anomaly_score, confidence, lifetime_value

    Returns
    -------
    pd.DataFrame with columns:
        customer_id, segment, lifetime_value, urgency_score,
        urgency_label, campaign_type, channel, discount_pct,
        message, subject, action_label, generated_at
    """
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame()

    df      = predictions_df.copy()
    df      = df[df['segment'].notna()].copy()
    df      = df[df['segment'].astype(str) != 'None'].copy()

    if df.empty:
        return pd.DataFrame()

    rows = []
    now  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for _, row in df.iterrows():
        seg_key  = get_segment_key(str(row.get('segment', 'AT_RISK')))
        anom     = float(row.get('anomaly_score', 0) or 0)
        ltv      = float(row.get('lifetime_value', 0) or 0)
        conf     = float(row.get('confidence', 0) or 0)
        uid      = str(row['user_id'])

        cfg      = SEGMENT_CAMPAIGNS.get(seg_key, SEGMENT_CAMPAIGNS['AT_RISK'])
        urgency  = compute_urgency(seg_key, anom, ltv)
        discount = compute_discount(seg_key, anom, ltv)
        channel  = select_channel(seg_key, urgency, ltv)
        message  = generate_message(seg_key, discount, uid)
        subject  = cfg['subject'].format(discount=discount)

        rows.append({
            'customer_id':    uid,
            'segment':        seg_key,
            'lifetime_value': round(ltv, 2),
            'churn_risk_pct': round(anom * 100, 1),
            'confidence_pct': round(conf * 100, 1),
            'urgency_score':  urgency,
            'urgency_label':  get_urgency_label(urgency),
            'campaign_type':  cfg['campaign_type'],
            'channel':        channel,
            'channel_icon':   CHANNEL_ICONS.get(channel, '📧'),
            'discount_pct':   discount,
            'subject':        subject,
            'message':        message,
            'action_label':   cfg['action_label'],
            'segment_color':  cfg['color'],
            'generated_at':   now,
        })

    result = pd.DataFrame(rows)

    # Sort by urgency descending — most urgent first
    result = result.sort_values('urgency_score', ascending=False)
    result = result.reset_index(drop=True)

    return result


def get_campaign_summary(actions_df: pd.DataFrame) -> dict:
    """
    Summarise campaign actions by segment and urgency.
    Used for the dashboard summary cards.
    """
    if actions_df is None or actions_df.empty:
        return {}

    summary = {}
    for seg in actions_df['segment'].unique():
        seg_df = actions_df[actions_df['segment'] == seg]
        summary[seg] = {
            'count':          len(seg_df),
            'avg_urgency':    round(seg_df['urgency_score'].mean(), 3),
            'avg_discount':   round(seg_df['discount_pct'].mean(), 1),
            'critical_count': len(seg_df[seg_df['urgency_label'] == 'CRITICAL']),
            'high_count':     len(seg_df[seg_df['urgency_label'] == 'HIGH']),
            'top_channel':    seg_df['channel'].mode().iloc[0],
            'campaign_type':  seg_df['campaign_type'].iloc[0],
            'color':          SEGMENT_CAMPAIGNS.get(seg, {}).get('color', '#888'),
        }
    return summary


def to_csv_bytes(actions_df: pd.DataFrame) -> bytes:
    """
    Convert actions DataFrame to CSV bytes for Streamlit download button.
    Excludes internal columns not needed in export.
    """
    export_cols = [
        'customer_id', 'segment', 'lifetime_value',
        'churn_risk_pct', 'urgency_score', 'urgency_label',
        'campaign_type', 'channel', 'discount_pct',
        'subject', 'message', 'generated_at'
    ]
    export_df = actions_df[[c for c in export_cols if c in actions_df.columns]]
    return export_df.to_csv(index=False).encode('utf-8')


def render_action_summary_html(summary: dict) -> str:
    """
    Render campaign summary as styled HTML cards.
    One card per segment showing count, avg urgency, recommended action.
    """
    if not summary:
        return '<p style="color:#888">No action data available.</p>'

    cards = []
    for seg, data in summary.items():
        color        = data.get('color', '#888')
        channel_icon = CHANNEL_ICONS.get(data['top_channel'], '📧')
        cards.append(f"""
<div style="background:#111827;border:1px solid #2d3748;
            border-radius:10px;padding:16px;
            border-left:4px solid {color};
            display:inline-block;width:180px;
            margin:6px;vertical-align:top">
  <div style="color:{color};font-weight:700;font-size:13px;
              margin-bottom:8px">{seg}</div>
  <div style="color:#fff;font-size:22px;font-weight:800">
    {data['count']:,}
  </div>
  <div style="color:#888;font-size:11px;margin-top:4px">customers</div>
  <div style="margin-top:10px;font-size:11px;color:#aaa">
    Avg urgency: <strong style="color:#fff">
      {data['avg_urgency']:.2f}</strong><br>
    Avg discount: <strong style="color:#fff">
      {data['avg_discount']:.0f}%</strong><br>
    Channel: <strong style="color:#fff">
      {channel_icon} {data['top_channel']}</strong><br>
    Critical: <strong style="color:#FF4444">
      {data['critical_count']}</strong>
  </div>
</div>
""")

    return f"""
<div style="margin:12px 0">
  {''.join(cards)}
</div>
"""
