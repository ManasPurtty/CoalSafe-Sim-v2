"""
verify_cross_modal.py — Multimodal Consensus & Decision Gating System.

Implements Cross-Modal Verification between Tabular Gas/Sensor Telemetry (Modality 1)
and Spatial Thermal Image Heatmaps (Modality 2) to prevent false-alarm automated spray triggers.
"""

import pandas as pd
import numpy as np
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_TABULAR = ROOT / "data" / "tabular" / "sensor_data.csv"
DATA_THERMAL = ROOT / "data" / "metadata" / "thermal_metadata.csv"

def evaluate_cross_modal_decision(tabular_row, thermal_row):
    """
    Evaluates multimodal consensus before triggering Layer 6 Automated Mitigation.
    
    Returns decision status:
      - 'TRIGGER_MITIGATION': Both modalities confirm HIGH/CRITICAL risk.
      - 'MODALITY_CONFLICT_SUPPRESS': One modality reports high risk while the other reports normal/low.
      - 'ROUTINE_MONITORING': Both modalities report normal/low risk.
    """
    # Modality 1: Tabular Sensor Telemetry Risk (Gas CO/CO2 + Surface Probe)
    tab_risk = tabular_row["risk_score"]
    tab_high = tab_risk >= 60.0  # HIGH or CRITICAL threshold
    
    # Modality 2: Thermal Image Spatial Anomaly Risk (Hotspot cell coverage & peak temp)
    hotspot_area = thermal_row.get("hotspot_area_cells", 0)
    max_pixel_temp = thermal_row.get("max_pixel_temp", tabular_row.get("surface_temperature_max", 30.0))
    
    thermal_high = (hotspot_area >= 20) or (max_pixel_temp >= 40.0)
    
    if tab_high and thermal_high:
        return "TRIGGER_MITIGATION", "Consensus Match: Both Tabular Sensors and Thermal Image confirm HIGH RISK. Automated Spray Triggered."
    elif tab_high and not thermal_high:
        return "MODALITY_CONFLICT_SUPPRESS", "Conflict Mismatch: Tabular sensors report HIGH RISK but Thermal Image shows NO HOTSPOT. Action Suppressed (Possible Sensor Fault)."
    elif not tab_high and thermal_high:
        return "MODALITY_CONFLICT_SUPPRESS", "Conflict Mismatch: Thermal Image shows HOTSPOT but Tabular Gas sensors report NORMAL. Action Suppressed (Possible Camera Reflection)."
    else:
        return "ROUTINE_MONITORING", "Both modalities report NORMAL/LOW risk. Routine Monitoring."

def run_verification_demo():
    print("=" * 70)
    print("  CoalSafe-Sim v2  |  Multimodal Decision Gating & Cross-Verification")
    print("=" * 70)
    
    if not DATA_TABULAR.exists() or not DATA_THERMAL.exists():
        print("  [ERROR] Data files missing. Run run_all.py first.")
        return
        
    df_tab = pd.read_csv(DATA_TABULAR)
    df_therm = pd.read_csv(DATA_THERMAL)
    
    # Simulate 3 test cases:
    # Case 1: Normal Consensus (S01)
    # Case 2: High Risk Consensus (S05 at t=90)
    # Case 3: Injected Sensor Conflict (Faulty Gas Sensor)
    
    print("\n[TEST CASE 1] Normal Baseline (S01, Minute 50)")
    r_tab1 = df_tab[(df_tab["scenario_id"] == "S01") & (df_tab["timestamp_min"] == 50)].iloc[0]
    r_th1  = df_therm[(df_therm["scenario_id"] == "S01") & (df_therm["timestamp_min"] == 50)].iloc[0]
    dec1, msg1 = evaluate_cross_modal_decision(r_tab1, r_th1)
    print(f"  Decision : {dec1}\n  Details  : {msg1}")
    
    print("\n[TEST CASE 2] Confirmed High Risk Event (S05, Minute 90)")
    r_tab2 = df_tab[(df_tab["scenario_id"] == "S05") & (df_tab["timestamp_min"] == 90)].iloc[0]
    r_th2  = df_therm[(df_therm["scenario_id"] == "S05") & (df_therm["timestamp_min"] == 90)].iloc[0]
    dec2, msg2 = evaluate_cross_modal_decision(r_tab2, r_th2)
    print(f"  Decision : {dec2}\n  Details  : {msg2}")
    
    print("\n[TEST CASE 3] Injected Conflict Test (Faulty CO Sensor = 250 ppm on Normal S01)")
    r_tab3 = r_tab1.copy()
    r_tab3["risk_score"] = 85.0 # Falsely inflated tabular risk due to glitch
    dec3, msg3 = evaluate_cross_modal_decision(r_tab3, r_th1)
    print(f"  Decision : {dec3}\n  Details  : {msg3}")
    
    print("\n" + "=" * 70)
    print("  [SUCCESS] Cross-Modal Decision Gating Verification Complete.")
    print("=" * 70)

if __name__ == "__main__":
    run_verification_demo()
