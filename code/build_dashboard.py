
import sys
import io
import json
import pathlib
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_TABULAR = ROOT / "data" / "tabular"
DATA_META = ROOT / "data" / "metadata"
DASHBOARD_DIR = ROOT / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 65)
    print("  CoalSafe-Sim v1 — Building Interactive Web Dashboard")
    print("=" * 65)

    sensor_path = DATA_TABULAR / "sensor_data.csv"
    latent_path = DATA_META / "latent_state.csv"
    thermal_path = DATA_META / "thermal_metadata.csv"
    scenarios_path = DATA_META / "scenarios.csv"

    if not sensor_path.exists():
        print(f"[ERROR] sensor_data.csv not found at {sensor_path}")
        sys.exit(1)

    sensor_df = pd.read_csv(sensor_path)
    latent_df = pd.read_csv(latent_path) if latent_path.exists() else pd.DataFrame()
    thermal_df = pd.read_csv(thermal_path) if thermal_path.exists() else pd.DataFrame()
    scenarios_df = pd.read_csv(scenarios_path) if scenarios_path.exists() else pd.DataFrame()

    if not latent_df.empty:
        merged_df = pd.merge(
            sensor_df,
            latent_df[[
                'scenario_id', 'timestamp_min', 'oxidation_state', 'oxidation_rate',
                'internal_temperature', 'deep_core_temperature',
                'middle_core_temperature', 'near_surface_temperature', 'heat_generation', 'heat_loss'
            ]],
            on=['scenario_id', 'timestamp_min'],
            how='left'
        )
    else:
        merged_df = sensor_df.copy()
        merged_df['deep_core_temperature'] = merged_df['surface_temperature_mean']
        merged_df['middle_core_temperature'] = merged_df['surface_temperature_mean']
        merged_df['near_surface_temperature'] = merged_df['surface_temperature_mean']
        merged_df['oxidation_state'] = 0.0

    if not thermal_df.empty:
        merged_df = pd.merge(
            merged_df,
            thermal_df[['scenario_id', 'timestamp_min', 'image_path']],
            on=['scenario_id', 'timestamp_min'],
            how='left'
        )
    else:
        merged_df['image_path'] = ''

    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    
    merged_df['CO'] = merged_df['CO'].fillna(-1)
    merged_df['surface_temperature_mean'] = merged_df['surface_temperature_mean'].fillna(-1)
    merged_df['surface_temperature_max'] = merged_df['surface_temperature_max'].fillna(-1)
    merged_df['hotspot_x'] = merged_df['hotspot_x'].fillna(-1)
    merged_df['hotspot_y'] = merged_df['hotspot_y'].fillna(-1)
    merged_df['image_path'] = merged_df['image_path'].fillna('MISSING')

    num_scenarios = len(merged_df['scenario_id'].unique())
    num_timesteps = len(merged_df)
    
    valid_deep_temp = merged_df['deep_core_temperature'].dropna()
    max_deep_temp = valid_deep_temp.max() if not valid_deep_temp.empty else 0.0
    
    valid_surf_temp = merged_df['surface_temperature_max'].replace(-1, np.nan).dropna()
    max_surf_temp = valid_surf_temp.max() if not valid_surf_temp.empty else 0.0
    
    valid_co = merged_df['CO'].replace(-1, np.nan).dropna()
    peak_co = valid_co.max() if not valid_co.empty else 0.0

    critical_count = len(merged_df[merged_df['risk_class'].isin(['HIGH', 'CRITICAL'])])

    scenarios_meta = {}
    if not scenarios_df.empty:
        for _, row in scenarios_df.iterrows():
            scenarios_meta[row['scenario_id']] = {
                'id': row['scenario_id'],
                'type': row['scenario_type'],
                'heating_pattern': row.get('heating_pattern', 'N/A'),
                'notes': row.get('notes', ''),
                'mitigation_start_min': row.get('mitigation_start_min', None) if pd.notna(row.get('mitigation_start_min')) else None
            }

    scenario_payloads = {}
    for sid in sorted(merged_df['scenario_id'].unique()):
        sdf = merged_df[merged_df['scenario_id'] == sid].sort_values('timestamp_min')
        
        scenario_payloads[sid] = {
            'info': scenarios_meta.get(sid, {'id': sid, 'type': sid, 'notes': ''}),
            'timestamps': sdf['timestamp_min'].tolist(),
            'deep_core_temp': [round(v, 2) if pd.notna(v) else None for v in sdf['deep_core_temperature']],
            'middle_core_temp': [round(v, 2) if pd.notna(v) else None for v in sdf['middle_core_temperature']],
            'near_surface_temp': [round(v, 2) if pd.notna(v) else None for v in sdf['near_surface_temperature']],
            'surf_temp_mean': [round(v, 2) if v != -1 and pd.notna(v) else None for v in sdf['surface_temperature_mean']],
            'surf_temp_max': [round(v, 2) if v != -1 and pd.notna(v) else None for v in sdf['surface_temperature_max']],
            'CO': [round(v, 2) if v != -1 and pd.notna(v) else None for v in sdf['CO']],
            'CO2': [round(v, 2) if pd.notna(v) else None for v in sdf['CO2']],
            'O2': [round(v, 2) if pd.notna(v) else None for v in sdf['O2']],
            'oxidation_state': [round(v, 4) if pd.notna(v) else 0.0 for v in sdf['oxidation_state']],
            'hotspot_area': [int(v) if pd.notna(v) else 0 for v in sdf['hotspot_area']],
            'risk_score': [round(v, 2) if pd.notna(v) else 0.0 for v in sdf['risk_score']],
            'risk_class': sdf['risk_class'].tolist(),
            'mitigation_status': sdf['mitigation_status'].tolist(),
            'image_paths': sdf['image_path'].tolist()
        }

    data_json = json.dumps(scenario_payloads)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CoalSafe-Sim v1 — Multimodal Early Risk & Predictive Warning Center</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0b0f19;
            --panel-bg: #131b2e;
            --panel-border: #1e293b;
            --accent-blue: #38bdf8;
            --accent-indigo: #6366f1;
            --normal-green: #10b981;
            --low-yellow: #f59e0b;
            --medium-orange: #f97316;
            --high-red: #ef4444;
            --critical-purple: #a855f7;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            padding: 20px;
            min-height: 100vh;
        }}

        /* Header Layout */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 16px 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .brand-icon {{
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #ef4444, #6366f1);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 20px;
            color: white;
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
        }}

        .brand-text h1 {{
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}

        .brand-text p {{
            font-size: 12px;
            color: var(--text-secondary);
        }}

        .kpi-row {{
            display: flex;
            gap: 20px;
        }}

        .kpi-card {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--panel-border);
            padding: 10px 18px;
            border-radius: 8px;
            text-align: center;
        }}

        .kpi-title {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}

        .kpi-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 18px;
            font-weight: 700;
            color: var(--accent-blue);
        }}

        /* Scenario Selector */
        .scenario-nav {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            overflow-x: auto;
            padding-bottom: 4px;
        }}

        .scenario-btn {{
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            color: var(--text-secondary);
            padding: 12px 18px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            font-size: 13px;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
        }}

        .scenario-btn:hover {{
            border-color: var(--accent-blue);
            color: var(--text-primary);
        }}

        .scenario-btn.active {{
            background: linear-gradient(135deg, #1e293b, #334155);
            border-color: var(--accent-blue);
            color: var(--accent-blue);
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
        }}

        .badge {{
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 700;
        }}

        /* Main Grid Layout */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }}

        @media (max-width: 1200px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .panel {{
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--panel-border);
        }}

        .panel-title {{
            font-size: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .panel-tag {{
            font-size: 10px;
            background: rgba(99, 102, 241, 0.2);
            color: var(--accent-indigo);
            border: 1px solid rgba(99, 102, 241, 0.4);
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 600;
        }}

        /* Architecture Section Highlights */
        .architecture-flow {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.6);
            border: 1px dashed var(--panel-border);
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 16px;
            font-size: 12px;
        }}

        .flow-step {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }}

        .flow-step-num {{
            font-size: 10px;
            font-weight: 700;
            color: var(--text-muted);
        }}

        .flow-step-name {{
            font-weight: 600;
            color: var(--accent-blue);
        }}

        .flow-arrow {{
            color: var(--text-muted);
            font-size: 14px;
        }}

        /* Early-Risk & Predictive Warning Panel (Architecture Diagram Layers 5 & 6) */
        .risk-warning-container {{
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 16px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 16px;
            margin-top: 16px;
        }}

        .gauge-box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
        }}

        .gauge-center-text {{
            position: absolute;
            top: 52%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }}

        .gauge-score {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 26px;
            font-weight: 800;
        }}

        .gauge-label {{
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .warning-card {{
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 12px;
        }}

        .alert-banner {{
            padding: 12px 16px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 700;
        }}

        .alert-banner.CRITICAL {{
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid var(--high-red);
            color: #fca5a5;
        }}

        .alert-banner.HIGH {{
            background: rgba(249, 115, 22, 0.2);
            border: 1px solid var(--medium-orange);
            color: #fdba74;
        }}

        .alert-banner.MEDIUM {{
            background: rgba(245, 158, 11, 0.2);
            border: 1px solid var(--low-yellow);
            color: #fde047;
        }}

        .alert-banner.LOW {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--normal-green);
            color: #6ee7b7;
        }}

        .alert-banner.NORMAL {{
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid var(--accent-blue);
            color: #93c5fd;
        }}

        .mitigation-system-box {{
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 12px;
        }}

        .mitigation-title {{
            font-size: 11px;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }}

        .mitigation-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }}

        .mit-item {{
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--panel-border);
            padding: 8px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
        }}

        .mit-item.active {{
            border-color: var(--normal-green);
            background: rgba(16, 185, 129, 0.1);
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-muted);
        }}

        .status-dot.active {{
            background: var(--normal-green);
            box-shadow: 0 0 8px var(--normal-green);
        }}

        /* Thermal Scrubber Component */
        .thermal-scrubber-box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
        }}

        #thermalCanvas {{
            width: 100%;
            max-width: 320px;
            height: 320px;
            border-radius: 12px;
            border: 2px solid var(--panel-border);
            background: #000;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            image-rendering: pixelated;
        }}

        .scrubber-controls {{
            width: 100%;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .scrubber-slider {{
            flex: 1;
            accent-color: var(--accent-blue);
        }}

        .time-display {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 700;
            background: rgba(15, 23, 42, 0.8);
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid var(--panel-border);
        }}

        .play-btn {{
            background: var(--accent-blue);
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .play-btn:hover {{
            background: #7dd3fc;
        }}

        .chart-container {{
            position: relative;
            height: 260px;
            width: 100%;
        }}
    </style>
</head>
<body>

    <!-- Header Section -->
    <header>
        <div class="brand">
            <div class="brand-icon">CS</div>
            <div class="brand-text">
                <h1>CoalSafe-Sim Version 1</h1>
                <p>Physics-Informed Multimodal Early Warning & Predictive Risk Dashboard</p>
            </div>
        </div>
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">Stockpiles Monitored</div>
                <div class="kpi-value">{num_scenarios}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Max Sub-Surface Temp</div>
                <div class="kpi-value" style="color: var(--high-red);">{max_deep_temp:.1f}°C</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Max Surface Temp</div>
                <div class="kpi-value" style="color: var(--medium-orange);">{max_surf_temp:.1f}°C</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Peak CO</div>
                <div class="kpi-value" style="color: var(--low-yellow);">{peak_co:.0f} ppm</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Critical Warnings</div>
                <div class="kpi-value" style="color: var(--critical-purple);">{critical_count}</div>
            </div>
        </div>
    </header>

    <!-- Architecture System Flow Highlights -->
    <div class="architecture-flow">
        <div class="flow-step">
            <span class="flow-step-num">LAYER 1</span>
            <span class="flow-step-name">Hidden Sub-Surface State</span>
        </div>
        <span class="flow-arrow">➔</span>
        <div class="flow-step">
            <span class="flow-step-num">LAYER 2</span>
            <span class="flow-step-name">Physical Progression</span>
        </div>
        <span class="flow-arrow">➔</span>
        <div class="flow-step">
            <span class="flow-step-num">LAYER 3</span>
            <span class="flow-step-name">Observable Signatures</span>
        </div>
        <span class="flow-arrow">➔</span>
        <div class="flow-step">
            <span class="flow-step-num">LAYER 4</span>
            <span class="flow-step-name">Multimodal AI Engine</span>
        </div>
        <span class="flow-arrow">➔</span>
        <div class="flow-step">
            <span class="flow-step-num">LAYER 5</span>
            <span class="flow-step-name">Early-Risk & Warning</span>
        </div>
        <span class="flow-arrow">➔</span>
        <div class="flow-step">
            <span class="flow-step-num">LAYER 6</span>
            <span class="flow-step-name">Automated Mitigation</span>
        </div>
    </div>

    <!-- Scenario Selector Navigation -->
    <div class="scenario-nav" id="scenarioNav"></div>

    <!-- Main Grid -->
    <div class="dashboard-grid">
        
        <!-- Left Column: Time-Series Graphs -->
        <div class="left-col">
            
            <!-- Panel 1: 3-Depth Layer Sub-Surface Thermal Core Migration -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>🔥 3-Depth Sub-Surface vs Surface Heat Migration</span>
                    </div>
                    <div class="panel-tag">LAYER 1 & 2: EARLY WARNING DELAY GAP</div>
                </div>
                <div class="chart-container">
                    <canvas id="depthTempChart"></canvas>
                </div>
            </div>

            <!-- Panel 2: Observable Gas Sensors (CO, CO2, O2) -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>💨 Observable Gas Sensors (CO, CO₂, O₂)</span>
                    </div>
                    <div class="panel-tag">LAYER 3: MULTISENSOR TRENDS</div>
                </div>
                <div class="chart-container">
                    <canvas id="gasesChart"></canvas>
                </div>
            </div>

            <!-- Panel 3: Risk Score & Hotspot Growth -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>📈 Risk Trajectory & Spatial Hotspot Area</span>
                    </div>
                    <div class="panel-tag">SYNTHETIC TARGET BANDS</div>
                </div>
                <div class="chart-container">
                    <canvas id="riskHotspotChart"></canvas>
                </div>
            </div>

        </div>

        <!-- Right Column: Early Risk Score & Predictive Warning Widget (Architecture Layer 5 & 6) -->
        <div class="right-col">
            
            <!-- Layer 5 & 6 Widget Panel -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>⚠️ Early-Risk Score & Predictive Warning</span>
                    </div>
                    <div class="panel-tag">LAYER 5 & 6 DECISION ENGINE</div>
                </div>

                <!-- Risk Score Donut Gauge & Warning Card -->
                <div class="risk-warning-container">
                    <div class="gauge-box">
                        <div class="chart-container" style="height: 140px; width: 140px;">
                            <canvas id="gaugeChart"></canvas>
                        </div>
                        <div class="gauge-center-text">
                            <div class="gauge-score" id="currentRiskScore">0%</div>
                            <div class="gauge-label" id="currentRiskClass">NORMAL</div>
                        </div>
                    </div>

                    <div class="warning-card">
                        <div class="alert-banner NORMAL" id="alertBanner">
                            <span id="alertIcon">🛡️</span>
                            <div>
                                <div style="font-size: 13px;" id="alertStatusText">NORMAL STATE</div>
                                <div style="font-size: 11px; opacity: 0.8;" id="actionRecommended">Action: Routine Monitoring</div>
                            </div>
                        </div>

                        <!-- Mitigation Status Indicator -->
                        <div class="mitigation-system-box">
                            <div class="mitigation-title">
                                <span>AUTOMATED MITIGATION SYSTEM</span>
                                <span id="mitStatusBadge" style="color: var(--text-muted);">INACTIVE</span>
                            </div>
                            <div class="mitigation-grid">
                                <div class="mit-item" id="mitWater">
                                    <div class="status-dot" id="dotWater"></div>
                                    <span>Water/Foam Spray</span>
                                </div>
                                <div class="mit-item" id="mitAir">
                                    <div class="status-dot" id="dotAir"></div>
                                    <span>Airflow Mgmt</span>
                                </div>
                                <div class="mit-item" id="mitStack">
                                    <div class="status-dot" id="dotStack"></div>
                                    <span>Stack Reconfig</span>
                                </div>
                                <div class="mit-item" id="mitConveyor">
                                    <div class="status-dot" id="dotConveyor"></div>
                                    <span>Conveyor Control</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Predictive Trajectory Forecast (+1h, +2h, +4h, +6h) -->
                <div style="margin-top: 20px;">
                    <div style="font-size: 12px; font-weight: 700; color: var(--text-secondary); margin-bottom: 8px; display: flex; justify-content: space-between;">
                        <span>RISK TRAJECTORY FORECAST</span>
                        <span style="color: var(--accent-blue);" id="forecastDelta">+0.0% / hr</span>
                    </div>
                    <div class="chart-container" style="height: 150px;">
                        <canvas id="forecastChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Synchronized Spatial Thermal Heatmap Scrubber -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>📸 50×50 Surface Thermal Heatmap</span>
                    </div>
                    <div class="panel-tag">SPATIAL SYNCHRONIZATION</div>
                </div>

                <div class="thermal-scrubber-box">
                    <canvas id="thermalCanvas" width="50" height="50"></canvas>
                    
                    <div class="scrubber-controls">
                        <button class="play-btn" id="playBtn">▶ Play</button>
                        <input type="range" min="0" max="120" value="0" class="scrubber-slider" id="timeSlider">
                        <div class="time-display" id="timeDisplay">0 min</div>
                    </div>

                    <div style="font-size: 11px; color: var(--text-muted); text-align: center;">
                        Move the slider or play to scrub through thermal frames synchronized with sub-surface temperature & gas signals.
                    </div>
                </div>
            </div>

        </div>

    </div>

    <!-- Embedded JSON Data & JS Engine -->
    <script>
        const SCENARIO_DATA = {data_json};

        let currentSid = 'S01';
        let currentTimestep = 0;
        let isPlaying = false;
        let playInterval = null;

        // Chart instances
        let depthTempChart = null;
        let gasesChart = null;
        let riskHotspotChart = null;
        let gaugeChart = null;
        let forecastChart = null;

        // Color helpers
        function getRiskColor(label) {{
            switch(label) {{
                case 'CRITICAL': return '#a855f7';
                case 'HIGH': return '#ef4444';
                case 'MEDIUM': return '#f97316';
                case 'LOW': return '#f59e0b';
                default: return '#10b981';
            }}
        }}

        // Initialize Navigation
        function initNav() {{
            const nav = document.getElementById('scenarioNav');
            nav.innerHTML = '';
            
            Object.keys(SCENARIO_DATA).forEach(sid => {{
                const sc = SCENARIO_DATA[sid];
                const btn = document.createElement('button');
                btn.className = `scenario-btn ${{sid === currentSid ? 'active' : ''}}`;
                btn.innerHTML = `
                    <span>${{sid}}</span>
                    <span style="font-size: 11px; opacity: 0.8;">${{sc.info.type || sid}}</span>
                `;
                btn.onclick = () => selectScenario(sid);
                nav.appendChild(btn);
            }});
        }}

        function selectScenario(sid) {{
            currentSid = sid;
            document.querySelectorAll('.scenario-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.innerText.includes(sid));
            }});
            currentTimestep = 0;
            document.getElementById('timeSlider').value = 0;
            updateDashboard();
        }}

        // Initialize Charts
        function initCharts() {{
            // 1. Depth Temperatures Chart
            const ctx1 = document.getElementById('depthTempChart').getContext('2d');
            depthTempChart = new Chart(ctx1, {{
                type: 'line',
                data: {{ labels: [], datasets: [] }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {{ mode: 'index', intersect: false }},
                    scales: {{
                        x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
                        y: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }}, title: {{ display: true, text: 'Temp (°C)', color: '#94a3b8' }} }}
                    }},
                    plugins: {{ legend: {{ labels: {{ color: '#f8fafc', font: {{ size: 11 }} }} }} }}
                }}
            }});

            // 2. Gas Sensors Chart
            const ctx2 = document.getElementById('gasesChart').getContext('2d');
            gasesChart = new Chart(ctx2, {{
                type: 'line',
                data: {{ labels: [], datasets: [] }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {{ mode: 'index', intersect: false }},
                    scales: {{
                        x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
                        y: {{ id: 'y1', type: 'linear', position: 'left', grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }}, title: {{ display: true, text: 'CO / CO2 (ppm)', color: '#94a3b8' }} }},
                        y2: {{ id: 'y2', type: 'linear', position: 'right', grid: {{ drawOnChartArea: false }}, ticks: {{ color: '#38bdf8' }}, title: {{ display: true, text: 'O2 (%)', color: '#38bdf8' }} }}
                    }},
                    plugins: {{ legend: {{ labels: {{ color: '#f8fafc', font: {{ size: 11 }} }} }} }}
                }}
            }});

            // 3. Risk & Hotspot Area Chart
            const ctx3 = document.getElementById('riskHotspotChart').getContext('2d');
            riskHotspotChart = new Chart(ctx3, {{
                type: 'line',
                data: {{ labels: [], datasets: [] }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {{ mode: 'index', intersect: false }},
                    scales: {{
                        x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
                        y: {{ id: 'y1', position: 'left', max: 100, min: 0, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }}, title: {{ display: true, text: 'Risk Score (0-100)', color: '#94a3b8' }} }},
                        y2: {{ id: 'y2', position: 'right', grid: {{ drawOnChartArea: false }}, ticks: {{ color: '#f97316' }}, title: {{ display: true, text: 'Hotspot Area (cells)', color: '#f97316' }} }}
                    }},
                    plugins: {{ legend: {{ labels: {{ color: '#f8fafc', font: {{ size: 11 }} }} }} }}
                }}
            }});

            // 4. Gauge Chart
            const ctx4 = document.getElementById('gaugeChart').getContext('2d');
            gaugeChart = new Chart(ctx4, {{
                type: 'doughnut',
                data: {{
                    datasets: [{{
                        data: [0, 100],
                        backgroundColor: ['#10b981', '#1e293b'],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '78%',
                    plugins: {{ tooltip: {{ enabled: false }}, legend: {{ display: false }} }}
                }}
            }});

            // 5. Risk Trajectory Forecast Chart (+1h, +2h, +4h, +6h)
            const ctx5 = document.getElementById('forecastChart').getContext('2d');
            forecastChart = new Chart(ctx5, {{
                type: 'line',
                data: {{
                    labels: ['Now', '+1h', '+2h', '+4h', '+6h'],
                    datasets: [{{
                        label: 'Risk Forecast (%)',
                        data: [0, 0, 0, 0, 0],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.15)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {{
                        x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
                        y: {{ min: 0, max: 100, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }}
                    }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});
        }}

        // Update Dashboard Views
        function updateDashboard() {{
            const data = SCENARIO_DATA[currentSid];
            if (!data) return;

            const t = currentTimestep;
            document.getElementById('timeDisplay').innerText = `${{t}} min`;

            // Update Charts Data
            depthTempChart.data.labels = data.timestamps;
            depthTempChart.data.datasets = [
                {{ label: 'Deep Core Temp (Layer 0)', data: data.deep_core_temp, borderColor: '#ef4444', borderWidth: 2, pointRadius: 0 }},
                {{ label: 'Middle Core Temp (Layer 1)', data: data.middle_core_temp, borderColor: '#f97316', borderWidth: 2, pointRadius: 0 }},
                {{ label: 'Near Surface Temp (Layer 2)', data: data.near_surface_temp, borderColor: '#f59e0b', borderWidth: 2, pointRadius: 0 }},
                {{ label: 'Surface Temp Max (Observable)', data: data.surf_temp_max, borderColor: '#38bdf8', borderWidth: 2, borderDash: [4, 4], pointRadius: 0 }}
            ];
            depthTempChart.update();

            gasesChart.data.labels = data.timestamps;
            gasesChart.data.datasets = [
                {{ label: 'CO (ppm)', data: data.CO, borderColor: '#ef4444', yAxisID: 'y1', borderWidth: 2, pointRadius: 0 }},
                {{ label: 'CO2 (ppm)', data: data.CO2, borderColor: '#f59e0b', yAxisID: 'y1', borderWidth: 2, pointRadius: 0 }},
                {{ label: 'O2 (%)', data: data.O2, borderColor: '#38bdf8', yAxisID: 'y2', borderWidth: 2, pointRadius: 0 }}
            ];
            gasesChart.update();

            riskHotspotChart.data.labels = data.timestamps;
            riskHotspotChart.data.datasets = [
                {{ label: 'Risk Score', data: data.risk_score, borderColor: '#a855f7', yAxisID: 'y1', borderWidth: 2, pointRadius: 0 }},
                {{ label: 'Hotspot Area', data: data.hotspot_area, borderColor: '#f97316', yAxisID: 'y2', borderWidth: 1.5, pointRadius: 0 }}
            ];
            riskHotspotChart.update();

            // Update Current Timestep Widgets
            const currentRisk = data.risk_score[t] || 0;
            const currentClass = data.risk_class[t] || 'NORMAL';
            const isMitigating = data.mitigation_status[t] === 1;

            document.getElementById('currentRiskScore').innerText = `${{currentRisk.toFixed(0)}}%`;
            document.getElementById('currentRiskClass').innerText = currentClass;
            document.getElementById('currentRiskClass').style.color = getRiskColor(currentClass);

            // Donut Gauge Update
            gaugeChart.data.datasets[0].data = [currentRisk, 100 - currentRisk];
            gaugeChart.data.datasets[0].backgroundColor = [getRiskColor(currentClass), '#1e293b'];
            gaugeChart.update();

            // Alert Banner Update
            const banner = document.getElementById('alertBanner');
            banner.className = `alert-banner ${{currentClass}}`;
            
            const statusText = document.getElementById('alertStatusText');
            const actionText = document.getElementById('actionRecommended');
            
            if (currentClass === 'CRITICAL') {{
                statusText.innerText = 'CRITICAL RUNAWAY RISK';
                actionText.innerText = 'Action: Emergency Water/Foam Spray & Stack Reconfiguration';
            }} else if (currentClass === 'HIGH') {{
                statusText.innerText = 'HIGH RISK WARNING';
                actionText.innerText = 'Action: Automated Preventive Water Mitigation Triggered';
            }} else if (currentClass === 'MEDIUM') {{
                statusText.innerText = 'MEDIUM RISK DETECTED';
                actionText.innerText = 'Action: Increase Airflow & Monitor Hotspot Growth';
            }} else if (currentClass === 'LOW') {{
                statusText.innerText = 'LOW RISK ELEVATION';
                actionText.innerText = 'Action: Inspect Sub-Surface Sensor Telemetry';
            }} else {{
                statusText.innerText = 'NORMAL STATE';
                actionText.innerText = 'Action: Routine Monitoring Loop';
            }}

            // Update Mitigation Status System Controls
            const mitBadge = document.getElementById('mitStatusBadge');
            const dotWater = document.getElementById('dotWater');
            const dotAir = document.getElementById('dotAir');
            const dotStack = document.getElementById('dotStack');
            const dotConveyor = document.getElementById('dotConveyor');

            if (isMitigating || currentRisk >= 60) {{
                mitBadge.innerText = 'ACTIVE';
                mitBadge.style.color = 'var(--normal-green)';
                dotWater.className = 'status-dot active';
                dotAir.className = 'status-dot active';
                dotStack.className = 'status-dot active';
                dotConveyor.className = 'status-dot active';
            }} else {{
                mitBadge.innerText = 'INACTIVE';
                mitBadge.style.color = 'var(--text-muted)';
                dotWater.className = 'status-dot';
                dotAir.className = 'status-dot';
                dotStack.className = 'status-dot';
                dotConveyor.className = 'status-dot';
            }}

            // Risk Trajectory Forecast (+1h, +2h, +4h, +6h) Calculation using dataset future progression & rate of rise
            const idx1h = Math.min(120, t + 30);  // +30 min in sim scale (~1h operational time scale)
            const idx2h = Math.min(120, t + 60);  // +60 min in sim scale (~2h operational)
            const idx4h = Math.min(120, t + 90);  // +90 min
            const idx6h = Math.min(120, t + 120); // +120 min

            const f1h = data.risk_score[idx1h] !== undefined ? data.risk_score[idx1h] : currentRisk;
            const f2h = data.risk_score[idx2h] !== undefined ? data.risk_score[idx2h] : f1h;
            const f4h = data.risk_score[idx4h] !== undefined ? data.risk_score[idx4h] : f2h;
            const f6h = data.risk_score[idx6h] !== undefined ? data.risk_score[idx6h] : f4h;

            forecastChart.data.datasets[0].data = [currentRisk, f1h, f2h, f4h, f6h];
            forecastChart.update();

            // Hourly rate calculation
            const futureDelta = f1h - currentRisk;
            const ratePerHour = futureDelta * 2; // rate over next operational hour
            document.getElementById('forecastDelta').innerText = `${{ratePerHour >= 0 ? '+' : ''}}${{ratePerHour.toFixed(1)}}% / hr`;

            // Draw Spatial Thermal Canvas Heatmap
            renderThermalHeatmap(t);
        }}

        // Render Synthetic Spatial Heatmap on HTML5 Canvas
        function renderThermalHeatmap(t) {{
            const canvas = document.getElementById('thermalCanvas');
            const ctx = canvas.getContext('2d');
            const data = SCENARIO_DATA[currentSid];
            if (!data) return;

            const surfMax = data.surf_temp_max[t] || 30;
            const nearSurf = data.near_surface_temp[t] || 30;
            const oxState = data.oxidation_state[t] || 0;

            const img = ctx.createImageData(50, 50);
            const cx = 25;
            const cy = 25;
            const radius = Math.min(3 + oxState * 8, 12);
            const strength = Math.max(0, nearSurf - 30);

            for (let y = 0; y < 50; y++) {{
                for (let x = 0; x < 50; x++) {{
                    const distSq = (x - cx) ** 2 + (y - cy) ** 2;
                    const heat = strength * Math.exp(-distSq / (2 * radius * radius));
                    const temp = 30 + heat;

                    // Inferno colormap mock
                    const norm = Math.min(1, Math.max(0, (temp - 30) / 40));
                    const idx = (y * 50 + x) * 4;

                    img.data[idx] = Math.floor(norm * 255);             // Red
                    img.data[idx + 1] = Math.floor(Math.pow(norm, 2) * 200); // Green
                    img.data[idx + 2] = Math.floor(Math.pow(1 - norm, 2) * 120); // Blue
                    img.data[idx + 3] = 255;                            // Alpha
                }}
            }}

            ctx.putImageData(img, 0, 0);
        }}

        // Controls Event Listeners
        document.getElementById('timeSlider').addEventListener('input', (e) => {{
            currentTimestep = parseInt(e.target.value);
            updateDashboard();
        }});

        document.getElementById('playBtn').addEventListener('click', () => {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('playBtn');
            if (isPlaying) {{
                btn.innerText = '⏸ Pause';
                playInterval = setInterval(() => {{
                    currentTimestep = (currentTimestep + 1) % 121;
                    document.getElementById('timeSlider').value = currentTimestep;
                    updateDashboard();
                }}, 150);
            }} else {{
                btn.innerText = '▶ Play';
                clearInterval(playInterval);
            }}
        }});

        // Window Load
        window.addEventListener('DOMContentLoaded', () => {{
            initNav();
            initCharts();
            updateDashboard();
        }});
    </script>
</body>
</html>"""

    out_html = DASHBOARD_DIR / "index.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  [SUCCESS] Compiled HTML dashboard at: {out_html}")
    print("=" * 65)

if __name__ == "__main__":
    main()
