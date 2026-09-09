# graph_visualizer.py  — Tier 3 new file
# ─────────────────────────────────────────────────────────────
# What this does:
#   Builds a live interactive HTML network graph of the customer
#   relationship structure learned by the DP-GCN model.
#
#   Two graph views:
#   1. C-GCN graph — customers connected by behavioural similarity
#                    (cosine similarity of feature vectors)
#   2. T-GCN graph — customers connected by structural role
#                    (same KMeans cluster)
#
#   Visual encoding:
#   - Node colour    = DP-GCN predicted segment
#   - Node size      = lifetime value (bigger = higher value)
#   - Edge thickness = similarity strength
#   - Edge colour    = graph type (C-GCN blue, T-GCN orange)
#
#   Used in dashboard_dpgcn.py as a new "Graph Network" tab.
#   Returns an HTML string rendered via st.components.v1.html()
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import sqlite3
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DB_FILE = "customer_data.db"

# ── Segment colour palette ─────────────────────────────────────
SEGMENT_COLORS = {
    'VIP_ACTIVE':     '#FFD700',   # gold
    'HIGH_POTENTIAL': '#00CED1',   # teal
    'DORMANT_VIP':    '#9370DB',   # purple
    'LOYAL_REGULAR':  '#32CD32',   # green
    'AT_RISK':        '#FF6347',   # red-orange
    'UNKNOWN':        '#888888',   # grey fallback
}

SEGMENT_LABELS = {
    '👑 VIP_ACTIVE':     'VIP_ACTIVE',
    '🔥 HIGH_POTENTIAL': 'HIGH_POTENTIAL',
    '💤 DORMANT_VIP':    'DORMANT_VIP',
    '✅ LOYAL_REGULAR':  'LOYAL_REGULAR',
    '⚠️ AT_RISK':        'AT_RISK',
}


def get_segment_key(segment_str: str) -> str:
    """Strip emoji prefix from segment string."""
    for k, v in SEGMENT_LABELS.items():
        if k in segment_str or v in segment_str:
            return v
    return 'UNKNOWN'


def build_graph_data(
    predictions_df: pd.DataFrame,
    max_nodes: int = 150,
    graph_type: str = 'cgcn',
    top_k: int = 3
) -> dict:
    """
    Build graph nodes and edges from prediction DataFrame.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output of dpgcn.predict_customers() with columns:
        user_id, segment, anomaly_score, confidence, lifetime_value
    max_nodes : int
        Cap nodes for performance (default 150)
    graph_type : str
        'cgcn' = cosine similarity edges
        'tgcn' = KMeans cluster edges
    top_k : int
        Number of edges per node (default 3)

    Returns
    -------
    dict with 'nodes' and 'edges' lists for D3/vis.js rendering
    """
    if predictions_df is None or predictions_df.empty:
        return {'nodes': [], 'edges': []}

    # Sample nodes for performance
    df = predictions_df.copy()
    if len(df) > max_nodes:
        # Stratified sample — keep proportional segment representation
        df = df.groupby('segment', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(max_nodes * len(x) / len(df)))))
        ).reset_index(drop=True)
        df = df.head(max_nodes)

    # Normalise lifetime value for node sizing
    ltv = df['lifetime_value'].fillna(0).values
    ltv_min, ltv_max = ltv.min(), ltv.max()
    if ltv_max > ltv_min:
        ltv_norm = 10 + 30 * (ltv - ltv_min) / (ltv_max - ltv_min)
    else:
        ltv_norm = np.full(len(df), 20.0)

    # Build nodes
    nodes = []
    for i, (_, row) in enumerate(df.iterrows()):
        seg_key   = get_segment_key(str(row.get('segment', 'UNKNOWN')))
        color     = SEGMENT_COLORS.get(seg_key, '#888888')
        ltv_val   = float(row.get('lifetime_value', 0) or 0)
        anom      = float(row.get('anomaly_score', 0) or 0)
        conf      = float(row.get('confidence', 0) or 0)
        uid       = str(row['user_id'])[:8] + '...'

        nodes.append({
            'id':       i,
            'label':    uid,
            'segment':  seg_key,
            'color':    color,
            'size':     float(ltv_norm[i]),
            'ltv':      round(ltv_val, 2),
            'anomaly':  round(anom * 100, 1),
            'conf':     round(conf * 100, 1),
            'title':    (
                f"ID: {uid}\n"
                f"Segment: {seg_key}\n"
                f"LTV: ${ltv_val:.2f}\n"
                f"Churn Risk: {anom*100:.1f}%\n"
                f"Confidence: {conf*100:.1f}%"
            )
        })

    # ── Feature matrix for edge computation ───────────────────
    # Recompute simplified 3-feature version from predictions
    # (full 8-feature version requires raw events; this is fast)
    feature_matrix = np.column_stack([
        df['lifetime_value'].fillna(0).values,
        df['anomaly_score'].fillna(0).values,
        df['confidence'].fillna(0).values,
    ])
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    # Normalise
    scaler = StandardScaler()
    if feature_matrix.std() > 0:
        feature_matrix = scaler.fit_transform(feature_matrix)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    edges = []

    if graph_type == 'cgcn':
        # C-GCN: cosine similarity edges
        sim = cosine_similarity(feature_matrix)
        np.fill_diagonal(sim, 0)
        k = min(top_k, len(df) - 1)
        seen = set()
        for i in range(len(df)):
            top_k_idx = np.argsort(sim[i])[::-1][:k]
            for j in top_k_idx:
                edge_key = (min(i, j), max(i, j))
                if edge_key not in seen and sim[i][j] > 0:
                    seen.add(edge_key)
                    edges.append({
                        'source':  i,
                        'target':  int(j),
                        'weight':  round(float(sim[i][j]), 3),
                        'color':   '#4A90D9',
                        'type':    'similarity'
                    })

    elif graph_type == 'tgcn':
        # T-GCN: KMeans cluster edges
        n_clusters = min(8, len(df) // 5 + 1)
        try:
            kmeans   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(feature_matrix)
            seen     = set()
            for cid in range(n_clusters):
                cluster_idx = np.where(clusters == cid)[0]
                # Connect each node to top_k others in same cluster
                for i in cluster_idx:
                    peers = [j for j in cluster_idx if j != i][:top_k]
                    for j in peers:
                        edge_key = (min(int(i), int(j)), max(int(i), int(j)))
                        if edge_key not in seen:
                            seen.add(edge_key)
                            edges.append({
                                'source': int(i),
                                'target': int(j),
                                'weight': 0.7,
                                'color':  '#FF8C00',
                                'type':   'cluster'
                            })
        except Exception:
            pass

    return {'nodes': nodes, 'edges': edges}


def build_graph_html(graph_data: dict, graph_type: str = 'cgcn') -> str:
    """
    Generate self-contained HTML with D3.js force-directed graph.

    Parameters
    ----------
    graph_data : dict   Output of build_graph_data()
    graph_type : str    'cgcn' or 'tgcn' (affects title/legend)

    Returns
    -------
    str   Complete HTML page for st.components.v1.html()
    """
    nodes_json = json.dumps(graph_data['nodes'])
    edges_json = json.dumps(graph_data['edges'])

    title = (
        "C-GCN Customer Connectivity Graph"
        if graph_type == 'cgcn'
        else "T-GCN Customer Topology Graph"
    )
    subtitle = (
        "Edges connect behaviourally similar customers (cosine similarity)"
        if graph_type == 'cgcn'
        else "Edges connect structurally similar customers (KMeans clusters)"
    )
    edge_color = '#4A90D9' if graph_type == 'cgcn' else '#FF8C00'

    legend_html = ''.join([
        f'<div style="display:flex;align-items:center;margin:4px 0">'
        f'<div style="width:14px;height:14px;border-radius:50%;'
        f'background:{color};margin-right:8px"></div>'
        f'<span style="color:#ccc;font-size:12px">{seg}</span></div>'
        for seg, color in SEGMENT_COLORS.items()
        if seg != 'UNKNOWN'
    ])

    node_count = len(graph_data['nodes'])
    edge_count = len(graph_data['edges'])

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0; padding: 0;
    background: #0e1117;
    font-family: 'Segoe UI', sans-serif;
    overflow: hidden;
  }}
  #graph-container {{
    width: 100%;
    height: 520px;
    position: relative;
  }}
  svg {{
    width: 100%;
    height: 100%;
  }}
  .node circle {{
    stroke: #fff;
    stroke-width: 1.5px;
    cursor: pointer;
    transition: opacity 0.2s;
  }}
  .node circle:hover {{
    stroke: #fff;
    stroke-width: 3px;
  }}
  .node text {{
    font-size: 9px;
    fill: #eee;
    pointer-events: none;
    text-anchor: middle;
    dominant-baseline: middle;
  }}
  .link {{
    stroke-opacity: 0.4;
  }}
  #tooltip {{
    position: absolute;
    background: rgba(20,20,35,0.95);
    border: 1px solid #444;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #eee;
    pointer-events: none;
    display: none;
    white-space: pre-line;
    line-height: 1.6;
    max-width: 220px;
    z-index: 100;
  }}
  #legend {{
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(20,20,35,0.9);
    border: 1px solid #333;
    border-radius: 8px;
    padding: 12px 16px;
  }}
  #legend h4 {{
    margin: 0 0 8px 0;
    color: #fff;
    font-size: 12px;
    font-weight: 600;
  }}
  #stats {{
    position: absolute;
    top: 12px;
    left: 12px;
    background: rgba(20,20,35,0.9);
    border: 1px solid #333;
    border-radius: 8px;
    padding: 10px 14px;
    color: #aaa;
    font-size: 11px;
    line-height: 1.8;
  }}
  #title-bar {{
    position: absolute;
    bottom: 12px;
    left: 50%;
    transform: translateX(-50%);
    text-align: center;
    color: #888;
    font-size: 11px;
  }}
</style>
</head>
<body>
<div id="graph-container">
  <svg id="graph-svg"></svg>
  <div id="tooltip"></div>
  <div id="legend">
    <h4>Segments</h4>
    {legend_html}
    <div style="margin-top:10px;padding-top:8px;border-top:1px solid #333">
      <div style="display:flex;align-items:center;margin:4px 0">
        <div style="width:30px;height:2px;background:{edge_color};margin-right:8px"></div>
        <span style="color:#ccc;font-size:11px">
          {"Similarity" if graph_type == "cgcn" else "Cluster"} edge
        </span>
      </div>
    </div>
  </div>
  <div id="stats">
    <strong style="color:#fff">{title}</strong><br>
    {subtitle}<br>
    Nodes: <strong style="color:#4A90D9">{node_count}</strong> &nbsp;
    Edges: <strong style="color:#4A90D9">{edge_count}</strong>
  </div>
  <div id="title-bar">
    Node size = Lifetime Value &nbsp;|&nbsp;
    Node colour = Segment &nbsp;|&nbsp;
    Hover for details
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const nodesData = {nodes_json};
const edgesData = {edges_json};

if (nodesData.length === 0) {{
  document.getElementById('graph-container').innerHTML =
    '<div style="color:#888;text-align:center;padding:200px 0;font-size:14px">' +
    'No graph data — train the DP-GCN model first</div>';
}} else {{

const svg    = d3.select('#graph-svg');
const width  = document.getElementById('graph-container').clientWidth;
const height = 520;
svg.attr('viewBox', `0 0 ${{width}} ${{height}}`);

// Arrow marker
svg.append('defs').append('marker')
   .attr('id', 'arrow')
   .attr('viewBox', '0 -5 10 10')
   .attr('refX', 15).attr('refY', 0)
   .attr('markerWidth', 4).attr('markerHeight', 4)
   .attr('orient', 'auto')
   .append('path')
   .attr('d', 'M0,-5L10,0L0,5')
   .attr('fill', '#555');

const g = svg.append('g');

// Zoom
svg.call(d3.zoom()
  .scaleExtent([0.3, 4])
  .on('zoom', e => g.attr('transform', e.transform))
);

// Links
const link = g.append('g')
  .selectAll('line')
  .data(edgesData)
  .join('line')
  .attr('class', 'link')
  .attr('stroke', d => d.color)
  .attr('stroke-width', d => Math.max(1, (d.weight || 0.5) * 3));

// Nodes
const node = g.append('g')
  .selectAll('g')
  .data(nodesData)
  .join('g')
  .attr('class', 'node')
  .call(d3.drag()
    .on('start', dragStart)
    .on('drag',  dragged)
    .on('end',   dragEnd)
  );

node.append('circle')
  .attr('r',    d => d.size / 2)
  .attr('fill', d => d.color);

node.append('text')
  .text(d => d.label)
  .attr('dy', d => d.size / 2 + 10);

// Tooltip
const tooltip = document.getElementById('tooltip');
node.on('mouseover', (event, d) => {{
  tooltip.style.display = 'block';
  tooltip.innerHTML =
    `<strong style="color:${{d.color}}">${{d.segment}}</strong><br>` +
    `ID: ${{d.label}}<br>` +
    `LTV: $${{d.ltv}}<br>` +
    `Churn Risk: ${{d.anomaly}}%<br>` +
    `Confidence: ${{d.conf}}%`;
}})
.on('mousemove', event => {{
  tooltip.style.left = (event.offsetX + 15) + 'px';
  tooltip.style.top  = (event.offsetY - 10) + 'px';
}})
.on('mouseout', () => {{ tooltip.style.display = 'none'; }});

// Force simulation
const sim = d3.forceSimulation(nodesData)
  .force('link',   d3.forceLink(edgesData)
    .id(d => d.id)
    .distance(60)
    .strength(d => (d.weight || 0.5) * 0.3)
  )
  .force('charge', d3.forceManyBody().strength(-80))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collision', d3.forceCollide().radius(d => d.size / 2 + 4));

sim.on('tick', () => {{
  link
    .attr('x1', d => d.source.x)
    .attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x)
    .attr('y2', d => d.target.y);
  node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
}});

function dragStart(event, d) {{
  if (!event.active) sim.alphaTarget(0.3).restart();
  d.fx = d.x; d.fy = d.y;
}}
function dragged(event, d) {{
  d.fx = event.x; d.fy = event.y;
}}
function dragEnd(event, d) {{
  if (!event.active) sim.alphaTarget(0);
  d.fx = null; d.fy = null;
}}

}}
</script>
</body>
</html>
"""
    return html


def get_graph_html(predictions_df: pd.DataFrame,
                   graph_type: str = 'cgcn',
                   max_nodes: int = 120) -> str:
    """
    Main entry point called from dashboard_dpgcn.py.

    Parameters
    ----------
    predictions_df : pd.DataFrame   from dpgcn.predict_customers()
    graph_type     : str            'cgcn' or 'tgcn'
    max_nodes      : int            node cap for performance

    Returns
    -------
    str   HTML string to pass to st.components.v1.html()
    """
    graph_data = build_graph_data(
        predictions_df,
        max_nodes=max_nodes,
        graph_type=graph_type,
        top_k=3
    )
    return build_graph_html(graph_data, graph_type=graph_type)
