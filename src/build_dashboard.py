"""
build_dashboard.py
-------------------
Builds a single self-contained HTML file: a ranked, filterable, sortable
"suspicious records" report for a finance reviewer. No server needed --
just open the file. Vanilla JS only (no build step), so it's trivial to
demo or hand to a reviewer who has zero setup tolerance.
"""

import json
import os
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SRC_DIR, "..", "data", "scored_expenses.csv")
OUTPUT_PATH = os.path.join(SRC_DIR, "..", "reports", "rexo_suspicious_records_report.html")

df = pd.read_csv(DATA_PATH)
df["submitted_at"] = pd.to_datetime(df["submitted_at"])

# Only ship the fields the dashboard needs, keep it light
records = df[[
    "record_id", "vendor", "category", "amount", "site_id", "approver",
    "submitted_at", "suspicion_score", "flag_reasons", "is_anomaly",
]].copy()
records["submitted_at"] = records["submitted_at"].dt.strftime("%Y-%m-%d %H:%M")
records_json = records.to_json(orient="records")

total = len(df)
flagged_count = int((df["suspicion_score"] >= 0.2012).sum())
avg_precision = round(float((df.sort_values("suspicion_score", ascending=False)
                              .head(150)["is_anomaly"]).mean()) * 100, 1)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Rexo ERP — Suspicious Records Report</title>
<style>
  :root {{
    --bg: #0f1420; --panel: #161d2e; --border: #263047; --text: #e6e9f0;
    --muted: #8a93a6; --accent: #5b8def; --danger: #e5484d; --warn: #f2a93b;
    --ok: #2ecc71;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); padding: 32px;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 24px; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 20px; min-width: 140px;
  }}
  .stat .num {{ font-size: 24px; font-weight: 700; }}
  .stat .label {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .controls {{
    display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center;
  }}
  input, select {{
    background: var(--panel); border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 8px; font-size: 14px;
  }}
  input[type=text] {{ min-width: 220px; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel);
           border-radius: 10px; overflow: hidden; }}
  th, td {{ padding: 10px 12px; text-align: left; font-size: 13px; border-bottom: 1px solid var(--border); }}
  th {{ background: #1c2438; cursor: pointer; user-select: none; color: var(--muted); font-weight: 600; }}
  th:hover {{ color: var(--text); }}
  tr:hover td {{ background: #1a2236; }}
  .score-bar {{ display: inline-block; height: 8px; border-radius: 4px; background: var(--accent); }}
  .score-cell {{ display: flex; align-items: center; gap: 8px; min-width: 130px; }}
  .badge {{ font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 600; }}
  .badge.high {{ background: rgba(229,72,77,0.18); color: var(--danger); }}
  .badge.med  {{ background: rgba(242,169,59,0.18); color: var(--warn); }}
  .badge.low  {{ background: rgba(46,204,113,0.15); color: var(--ok); }}
  .reasons {{ color: var(--muted); font-size: 12px; max-width: 320px; }}
  .amount {{ font-variant-numeric: tabular-nums; }}
  .footer-note {{ margin-top: 20px; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
  <h1>Rexo ERP — Suspicious Records Report</h1>
  <div class="sub">Ranked by suspicion score. Review top of list first — precision is highest there.</div>

  <div class="stats">
    <div class="stat"><div class="num">{total:,}</div><div class="label">Total records scanned</div></div>
    <div class="stat"><div class="num">{flagged_count}</div><div class="label">Flagged for review (score ≥ 0.20)</div></div>
    <div class="stat"><div class="num">{avg_precision}%</div><div class="label">Precision in top 150 ranked</div></div>
  </div>

  <div class="controls">
    <input type="text" id="search" placeholder="Search vendor, approver, site...">
    <select id="categoryFilter"><option value="">All categories</option></select>
    <select id="riskFilter">
      <option value="">All risk levels</option>
      <option value="high">High (≥ 0.30)</option>
      <option value="med">Medium (0.15 – 0.30)</option>
      <option value="low">Low (&lt; 0.15)</option>
    </select>
  </div>

  <table id="tbl">
    <thead>
      <tr>
        <th data-key="record_id">ID</th>
        <th data-key="vendor">Vendor</th>
        <th data-key="category">Category</th>
        <th data-key="amount">Amount (₹)</th>
        <th data-key="site_id">Site</th>
        <th data-key="approver">Approver</th>
        <th data-key="submitted_at">Submitted</th>
        <th data-key="suspicion_score">Suspicion Score</th>
        <th>Reasons Flagged</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>

  <div class="footer-note">
    Suspicion score blends rule-based checks (duplicates, round amounts, off-hours, unusual approvers),
    per-vendor statistical outlier detection, and an Isolation Forest model. Ground-truth evaluation
    on labeled test data: 74.6% precision / 92.0% recall / F1 0.824 at the default threshold.
  </div>

<script>
  const DATA = {records_json};
  let sortKey = "suspicion_score";
  let sortDir = -1;

  const categories = [...new Set(DATA.map(d => d.category))].sort();
  const catSelect = document.getElementById("categoryFilter");
  categories.forEach(c => {{
    const opt = document.createElement("option");
    opt.value = c; opt.textContent = c;
    catSelect.appendChild(opt);
  }});

  function riskBadge(score) {{
    if (score >= 0.30) return '<span class="badge high">HIGH</span>';
    if (score >= 0.15) return '<span class="badge med">MEDIUM</span>';
    return '<span class="badge low">LOW</span>';
  }}

  function riskLevel(score) {{
    if (score >= 0.30) return "high";
    if (score >= 0.15) return "med";
    return "low";
  }}

  function render() {{
    const q = document.getElementById("search").value.toLowerCase();
    const cat = document.getElementById("categoryFilter").value;
    const risk = document.getElementById("riskFilter").value;

    let rows = DATA.filter(d => {{
      const matchesSearch = !q || (d.vendor + d.approver + d.site_id).toLowerCase().includes(q);
      const matchesCat = !cat || d.category === cat;
      const matchesRisk = !risk || riskLevel(d.suspicion_score) === risk;
      return matchesSearch && matchesCat && matchesRisk;
    }});

    rows.sort((a, b) => {{
      const av = a[sortKey], bv = b[sortKey];
      if (typeof av === "string") return sortDir * av.localeCompare(bv);
      return sortDir * (av - bv);
    }});

    const tbody = document.getElementById("tbody");
    tbody.innerHTML = rows.map(d => `
      <tr>
        <td>${{d.record_id}}</td>
        <td>${{d.vendor}}</td>
        <td>${{d.category}}</td>
        <td class="amount">₹${{d.amount.toLocaleString("en-IN")}}</td>
        <td>${{d.site_id}}</td>
        <td>${{d.approver}}</td>
        <td>${{d.submitted_at}}</td>
        <td>
          <div class="score-cell">
            ${{riskBadge(d.suspicion_score)}}
            <span>${{d.suspicion_score.toFixed(3)}}</span>
          </div>
        </td>
        <td class="reasons">${{d.flag_reasons}}</td>
      </tr>
    `).join("");
  }}

  document.querySelectorAll("th[data-key]").forEach(th => {{
    th.addEventListener("click", () => {{
      const key = th.dataset.key;
      if (sortKey === key) sortDir *= -1; else {{ sortKey = key; sortDir = -1; }}
      render();
    }});
  }});

  document.getElementById("search").addEventListener("input", render);
  document.getElementById("categoryFilter").addEventListener("change", render);
  document.getElementById("riskFilter").addEventListener("change", render);

  render();
</script>
</body>
</html>
"""

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    f.write(HTML)

print(f"Dashboard written to {OUTPUT_PATH}")
