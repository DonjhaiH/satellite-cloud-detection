"""part3_build_dashboard.py - builds the interactive HTML dashboard.
Reads the model_summary.csv and cv_results.csv files from each feature set,
packs them into a JSON blob, and embeds that in a self-contained HTML page
with Plotly charts. The result gets deployed to GitHub Pages."""

import json
import os
from pathlib import Path

import pandas as pd


def load_result_frames(base_dir: Path):
    """Load summary and CV CSVs for each feature set (A, B, C).
    Expects directories named part3_results_{A,B,C}_seed15 under base_dir."""
    out = {}
    for fs in ["A", "B", "C"]:
        folder = base_dir / f"part3_results_{fs}_seed15"
        summary_path = folder / "model_summary.csv"
        cv_path = folder / "cv_results.csv"
        if not summary_path.exists() or not cv_path.exists():
            raise FileNotFoundError(f"Missing result files under {folder}")
        out[fs] = {
            "summary": pd.read_csv(summary_path),
            "cv": pd.read_csv(cv_path),
        }
    return out


def build_payload(frames):
    """Convert dataframes to a nested dict that we can JSON-serialize
    and embed directly in the HTML file."""
    payload = {"summary": {}, "cv": {}}
    for fs, d in frames.items():
        payload["summary"][fs] = d["summary"].to_dict(orient="records")
        payload["cv"][fs] = d["cv"].to_dict(orient="records")
    return payload


def write_dashboard_html(payload, output_path: Path):
    data_json = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Model Evaluation Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --primary: #1b6ca8;
      --bg: #f8f9fa;
      --card-bg: #ffffff;
      --text: #2c3e50;
      --text-muted: #7f8c8d;
      --border: #e9ecef;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .header {{
      background: var(--card-bg);
      padding: 1.5rem 2rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .header h1 {{ margin: 0; font-size: 1.5rem; color: var(--primary); }}
    .header p {{ margin: 0.5rem 0 0 0; color: var(--text-muted); font-size: 0.95rem; }}
    
    .container {{ padding: 2rem; max-width: 1400px; margin: 0 auto; }}
    
    /* Layout */
    .top-section {{
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    
    .panel {{
      background: var(--card-bg);
      border-radius: 8px;
      padding: 1.25rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      border: 1px solid var(--border);
    }}
    .panel h2 {{ margin: 0 0 1rem 0; font-size: 1.1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}

    /* Controls */
    .control-group {{ margin-bottom: 1.25rem; }}
    .control-group label {{ display: block; font-size: 0.85rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }}
    select {{
      width: 100%;
      padding: 0.5rem;
      border: 1px solid var(--border);
      border-radius: 4px;
      background: #fff;
      font-size: 0.95rem;
    }}
    .radio-pill {{
      display: flex;
      gap: 0.5rem;
    }}
    .radio-pill label {{
      flex: 1;
      text-align: center;
      padding: 0.4rem;
      background: #f1f3f5;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: 500;
      text-transform: none;
      color: var(--text);
      transition: all 0.2s;
    }}
    .radio-pill input[type="radio"] {{ display: none; }}
    .radio-pill input[type="radio"]:checked + label {{
      background: var(--primary);
      color: white;
    }}

    /* KPIs */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1rem;
    }}
    .kpi-card {{
      background: #f8fafc;
      padding: 1.25rem;
      border-radius: 6px;
      border: 1px solid var(--border);
    }}
    .kpi-label {{ font-size: 0.85rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem; }}
    .kpi-val {{ font-size: 1.8rem; font-weight: 700; color: var(--primary); }}
    .kpi-sub {{ font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem; }}

    /* Charts */
    .chart-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>Polar Cloud Detection Dashboard</h1>
      <p>Interactive evaluation of model performance across feature sets and test images.</p>
    </div>
  </div>

  <div class="container">
    <div class="top-section">
      <!-- Sidebar Controls -->
      <div class="panel">
        <h2>Filters</h2>
        <div class="control-group">
          <label>Metric</label>
          <select id="metricSelect">
            <option value="ROC_AUC">ROC-AUC</option>
            <option value="PR_AUC">PR-AUC</option>
            <option value="F1">F1 Score</option>
            <option value="Bal_Acc">Balanced Accuracy</option>
          </select>
        </div>
        
        <div class="control-group">
          <label>Feature Setup</label>
          <div class="radio-pill">
            <input type="radio" name="runType" id="rt_full" value="full" checked>
            <label for="rt_full">Full Features</label>
            <input type="radio" name="runType" id="rt_top5" value="top5">
            <label for="rt_top5">Top-5 Only</label>
          </div>
        </div>

        <div class="control-group" style="margin-top: 2rem;">
          <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.4;">
            <b>Set A:</b> 8 base features<br>
            <b>Set B:</b> Base + AE8 embeddings (16 total)<br>
            <b>Set C:</b> Base + AE16 embeddings (24 total)
          </p>
        </div>
      </div>

      <!-- Main KPI & Main Chart -->
      <div>
        <div class="kpi-grid" style="margin-bottom: 1.5rem;">
          <div class="kpi-card">
            <div class="kpi-label">Best Model (Mean)</div>
            <div class="kpi-val" id="kpiModel">-</div>
            <div class="kpi-sub" id="kpiModelSub">Set - | -</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Peak Metric Score</div>
            <div class="kpi-val" id="kpiScore">-</div>
            <div class="kpi-sub" id="kpiScoreSub">± - std dev</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Hardest Image</div>
            <div class="kpi-val" style="color: #e74c3c;">O012791</div>
            <div class="kpi-sub">Consistently lowest AUC</div>
          </div>
        </div>
        
        <div class="panel">
          <div id="barChart" style="height: 380px;"></div>
        </div>
      </div>
    </div>

    <!-- Secondary Charts -->
    <div class="chart-grid">
      <div class="panel">
        <div id="lineChart" style="height: 350px;"></div>
      </div>
      <div class="panel">
        <div id="heatmapChart" style="height: 350px;"></div>
      </div>
    </div>
  </div>

  <script>
    const RAW = {data_json};
    const COLORS = {{ A: "#95a5a6", B: "#3498db", C: "#e74c3c" }};
    const MODEL_DISP = {{
      "LR_L1": "LR (L1)", "LR_L2": "LR (L2)", "RandomForest": "Random Forest",
      "GradientBoosting": "Grad. Boosting", "SVM_RBF": "SVM (RBF)"
    }};

    function getSelected() {{
      return {{
        metric: document.getElementById("metricSelect").value,
        runType: document.querySelector('input[name="runType"]:checked').value
      }};
    }}

    function getSummaryRows() {{
      let rows = [];
      ["A", "B", "C"].forEach(fs => {{
        RAW.summary[fs].forEach(r => rows.push({{ ...r, feature_set: fs }}));
      }});
      return rows;
    }}

    function renderAll() {{
      const opts = getSelected();
      const mKey = opts.metric + "_mean";
      const sKey = opts.metric + "_std";
      
      const allRows = getSummaryRows();
      const filtered = allRows.filter(r => r.run_type === opts.runType);

      // 1. Update KPIs
      if (filtered.length > 0) {{
        const best = [...filtered].sort((a, b) => b[mKey] - a[mKey])[0];
        document.getElementById("kpiModel").textContent = MODEL_DISP[best.model] || best.model;
        document.getElementById("kpiModelSub").textContent = `Set ${{best.feature_set}}`;
        document.getElementById("kpiScore").textContent = best[mKey].toFixed(4);
        document.getElementById("kpiScoreSub").textContent = `± ${{best[sKey].toFixed(4)}} across 3 folds`;
      }}

      // 2. Main Bar Chart
      const models = ["LR_L2", "LR_L1", "RandomForest", "GradientBoosting", "SVM_RBF"];
      const traces = ["A", "B", "C"].map(fs => {{
        const sub = filtered.filter(r => r.feature_set === fs);
        const dict = {{}};
        sub.forEach(r => dict[r.model] = r);
        
        return {{
          type: "bar",
          name: `Set ${{fs}}`,
          x: models.map(m => MODEL_DISP[m]),
          y: models.map(m => dict[m] ? dict[m][mKey] : null),
          error_y: {{
            type: "data",
            array: models.map(m => dict[m] ? dict[m][sKey] : null),
            visible: true, thickness: 1.5
          }},
          marker: {{ color: COLORS[fs] }},
          opacity: 0.85,
          hovertemplate: "%{{x}}<br>Set " + fs + "<br>Score: %{{y:.4f}}<extra></extra>"
        }};
      }});

      Plotly.newPlot("barChart", traces, {{
        title: "Model Performance Comparison",
        barmode: "group",
        template: "plotly_white",
        yaxis: {{ title: opts.metric, range: [0.55, 1.02] }},
        legend: {{ orientation: "h", y: 1.15 }}
      }}, {{responsive: true}});

      // 3. Line Chart (Feature Set Progression)
      const lineTraces = models.map(m => {{
        const sub = filtered.filter(r => r.model === m);
        const dict = {{}};
        sub.forEach(r => dict[r.feature_set] = r[mKey]);
        return {{
          type: "scatter", mode: "lines+markers",
          name: MODEL_DISP[m],
          x: ["A", "B", "C"],
          y: ["A", "B", "C"].map(fs => dict[fs] || null),
          marker: {{ size: 8 }},
          hovertemplate: "%{{x}}<br>Score: %{{y:.4f}}<extra></extra>"
        }};
      }});

      Plotly.newPlot("lineChart", lineTraces, {{
        title: "Impact of Autoencoder Embeddings (A → B → C)",
        template: "plotly_white",
        xaxis: {{ title: "Feature Set" }},
        yaxis: {{ title: opts.metric, range: [0.55, 1.02] }},
        legend: {{ font: {{ size: 10 }} }}
      }}, {{responsive: true}});

      // 4. Heatmap (Per-fold)
      let cvRows = [];
      ["A", "B", "C"].forEach(fs => {{
        RAW.cv[fs].forEach(r => cvRows.push({{ ...r, feature_set: fs }}));
      }});
      const cvFiltered = cvRows.filter(r => r.run_type === opts.runType);
      
      const rowLabels = [];
      models.forEach(m => {{
        ["A", "B", "C"].forEach(fs => rowLabels.push(`${{MODEL_DISP[m]}} (Set ${{fs}})`));
      }});
      
      const imgs = ["O012791", "O013257", "O013490"];
      const zData = rowLabels.map(rl => {{
        const m = Object.keys(MODEL_DISP).find(k => rl.includes(MODEL_DISP[k]));
        const fs = rl.includes("Set A") ? "A" : (rl.includes("Set B") ? "B" : "C");
        return imgs.map(img => {{
          const match = cvFiltered.find(r => r.model === m && r.feature_set === fs && r.test_image === img);
          return match ? match[opts.metric] : null;
        }});
      }});

      Plotly.newPlot("heatmapChart", [{{
        type: "heatmap", x: imgs, y: rowLabels, z: zData,
        colorscale: "RdYlGn", zmin: 0.5, zmax: 1.0,
        hovertemplate: "Test Img: %{{x}}<br>Model: %{{y}}<br>Score: %{{z:.4f}}<extra></extra>"
      }}], {{
        title: "Per-Image Breakdown (Folds)",
        template: "plotly_white",
        margin: {{ l: 150 }}
      }}, {{responsive: true}});
    }}

    document.getElementById("metricSelect").addEventListener("change", renderAll);
    document.querySelectorAll('input[name="runType"]').forEach(r => r.addEventListener("change", renderAll));
    
    renderAll();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main():
    # everything is relative to wherever this script lives
    code_dir = Path(__file__).resolve().parent
    frames = load_result_frames(code_dir)
    payload = build_payload(frames)
    out_html = code_dir / "part3_dashboard.html"
    write_dashboard_html(payload, out_html)
    print(f"Dashboard saved to {out_html}")


if __name__ == "__main__":
    main()
