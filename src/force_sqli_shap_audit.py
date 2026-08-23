"""
Task 2 — Live SQLi SHAP Audit (Track B Confirmation)
Confirms whether Track B's app-layer feature restoration propagated correctly to the live
XGB-Twin-v2 model, and whether SHAP attributions for SQL_injection are domain-appropriate.

Expected features for SQL_injection (from Track B analysis):
    http.content_length, tcp.len, dev_tcp.len, http.response, dev_http.content_length

Pass condition: at least 2 of these 5 appear in the top-5 SHAP features across SQLi samples.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.twin_model import DigitalTwin
from src.deviation_engine import DeviationEngine

SQLI_EXPECTED_FEATURES = {
    "http.content_length",
    "tcp.len",
    "dev_tcp.len",
    "http.response",
    "dev_http.content_length"
}

ICMP_EXPECTED_FEATURES = {
    "icmp.checksum",
    "icmp.seq_le",
    "dev_icmp.checksum",
    "dev_icmp.seq_le"
}


def run_sqli_shap_audit(
    data_csv: str = "data/sampled_dataset.csv",
    model_dir: str = "models",
    n_sqli_samples: int = 30,
    n_control_samples: int = 20,
    output_png: str = "results/sqli_shap_audit_panel.png",
    random_state: int = 42
):
    print("=" * 80)
    print("  TASK 2 — SQL_injection SHAP AUDIT (Track B Confirmation)")
    print("  Verifies app-layer feature dominance in SHAP for SQLi vs. ICMP control group")
    print("=" * 80)

    try:
        import shap
    except ImportError:
        print("[ERROR] shap not installed. Run: pip install shap")
        sys.exit(1)

    df = pd.read_csv(data_csv)
    df_sqli = df[df["Attack_type"] == "SQL_injection"].sample(
        min(n_sqli_samples, len(df[df["Attack_type"] == "SQL_injection"])),
        random_state=random_state
    ).reset_index(drop=True)
    df_icmp = df[df["Attack_type"] == "DDoS_ICMP"].sample(
        min(n_control_samples, len(df[df["Attack_type"] == "DDoS_ICMP"])),
        random_state=random_state
    ).reset_index(drop=True)

    print(f"\nAuditing {len(df_sqli)} SQL_injection samples + {len(df_icmp)} DDoS_ICMP control samples")

    twin = DigitalTwin.load(model_dir)
    dev_engine = DeviationEngine(twin=twin, model_dir=model_dir)
    ids_model = joblib.load(os.path.join(model_dir, "xgb_fused.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    raw_feature_cols = joblib.load(os.path.join(model_dir, "raw_features.pkl"))
    dev_feature_cols = joblib.load(os.path.join(model_dir, "dev_features.pkl"))

    results = {}
    for group_name, group_df in [("SQL_injection", df_sqli), ("DDoS_ICMP (control)", df_icmp)]:
        dev_df = dev_engine.compute_deviations(group_df)
        X_raw = group_df[raw_feature_cols].values
        X_dev = dev_df[dev_feature_cols].values
        X_fused = np.hstack([X_raw, X_dev])
        all_feature_names = raw_feature_cols + dev_feature_cols

        explainer = shap.TreeExplainer(ids_model)
        shap_values = explainer.shap_values(X_fused)

        probs = ids_model.predict_proba(X_fused)
        preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))

        print(f"\n  {group_name}: Predictions = {set(preds)}")

        # Aggregate mean |SHAP| per feature — handle multi-class list/3D array
        shap_arr = np.array(shap_values)
        # If 3D (n_classes, n_samples, n_features), average over classes first
        if shap_arr.ndim == 3:
            mean_shap_abs = np.mean(np.abs(shap_arr), axis=(0, 1))  # (n_features,)
        elif shap_arr.ndim == 2:
            mean_shap_abs = np.mean(np.abs(shap_arr), axis=0)       # (n_features,)
        else:
            mean_shap_abs = np.abs(shap_arr)

        feature_importance = {f: float(v) for f, v in zip(all_feature_names, mean_shap_abs)}
        top5 = sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
        results[group_name] = {
            "top5": top5,
            "feature_importance": feature_importance,
            "preds": preds
        }

    # ── Assertion check ───────────────────────────────────────────────────────────
    sqli_top5_features = {f for f, _ in results["SQL_injection"]["top5"]}
    sqli_overlap = sqli_top5_features & SQLI_EXPECTED_FEATURES
    sqli_pass = len(sqli_overlap) >= 2

    icmp_top5_features = {f for f, _ in results["DDoS_ICMP (control)"]["top5"]}
    icmp_overlap = icmp_top5_features & ICMP_EXPECTED_FEATURES
    icmp_pass = len(icmp_overlap) >= 2

    print("\n" + "=" * 80)
    print("  SHAP AUDIT RESULTS")
    print("=" * 80)

    for group_name, group_results in results.items():
        expected_set = SQLI_EXPECTED_FEATURES if "SQL" in group_name else ICMP_EXPECTED_FEATURES
        top5 = group_results["top5"]
        overlap = {f for f, _ in top5} & expected_set

        print(f"\n  [{group_name}]")
        print(f"  {'Feature':<35} {'Mean |SHAP|':>12}  {'Expected?':<10}")
        print("  " + "─" * 60)
        for feat, val in top5:
            expected_marker = "✓ YES" if feat in expected_set else "  no"
            print(f"  {feat:<35} {val:>12.6f}  {expected_marker:<10}")
        print(f"\n  Domain overlap: {overlap}")

    print()
    print("─" * 80)
    sqli_status = "[PASS]" if sqli_pass else "[FAIL]"
    icmp_status = "[PASS]" if icmp_pass else "[FAIL]"
    print(f"  {sqli_status} SQL_injection SHAP domain check: {len(sqli_overlap)}/5 expected features in top-5")
    print(f"  {icmp_status} DDoS_ICMP control SHAP check   : {len(icmp_overlap)}/4 expected features in top-5")

    overall = "PASS" if sqli_pass and icmp_pass else ("PARTIAL" if sqli_pass or icmp_pass else "FAIL")
    print(f"\n  OVERALL TRACK B VERDICT: {overall}")
    if overall == "PASS":
        print("  App-layer feature restoration confirmed. SQLi and ICMP SHAP features are domain-appropriate.")
    elif overall == "PARTIAL":
        print("  One group passed. Investigate the failing group's feature pipeline.")
    else:
        print("  Both groups failed. App-layer feature restoration may not have reached the deployed model.")
    print("=" * 80)

    # ── Visual panel ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0a0f1e")
    titles = ["SQL_injection (Target)", "DDoS_ICMP (Control)"]
    expected_sets = [SQLI_EXPECTED_FEATURES, ICMP_EXPECTED_FEATURES]
    palette_hit  = "#38bdf8"
    palette_miss = "#64748b"

    for ax, group_name, title, expected_set in zip(
        axes, results.keys(), titles, expected_sets
    ):
        top10 = sorted(results[group_name]["feature_importance"].items(), key=lambda x: -x[1])[:10]
        feats = [f for f, _ in top10]
        vals  = [v for _, v in top10]
        colors = [palette_hit if f in expected_set else palette_miss for f in feats]
        bars = ax.barh(feats[::-1], vals[::-1], color=colors[::-1], height=0.65)
        ax.set_facecolor("#0d1526")
        ax.set_title(title, color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Mean |SHAP Value|", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")
        hit_patch  = mpatches.Patch(color=palette_hit,  label="Domain-Expected Feature")
        miss_patch = mpatches.Patch(color=palette_miss, label="Other Feature")
        ax.legend(handles=[hit_patch, miss_patch], facecolor="#0d1526", labelcolor="#94a3b8", fontsize=8)

    plt.suptitle("Track B — SHAP Domain Audit: SQL_injection vs. DDoS_ICMP",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="#0a0f1e")
    plt.close()
    print(f"\n[SAVED] SHAP audit panel → {output_png}")

    return {
        "sqli_overlap": sqli_overlap,
        "sqli_pass": sqli_pass,
        "icmp_pass": icmp_pass,
        "overall": overall,
        "top5_sqli": results["SQL_injection"]["top5"],
        "top5_icmp": results["DDoS_ICMP (control)"]["top5"]
    }


def append_to_track_b_report(results: dict, report_path: str = "results/track_b_shap_app_layer_report.md"):
    sqli_rows = "\n".join(
        f"| {f} | {v:.6f} | {'✓ Expected' if f in SQLI_EXPECTED_FEATURES else 'Other'} |"
        for f, v in results["top5_sqli"]
    )
    icmp_rows = "\n".join(
        f"| {f} | {v:.6f} | {'✓ Expected' if f in ICMP_EXPECTED_FEATURES else 'Other'} |"
        for f, v in results["top5_icmp"]
    )

    section = f"""

---

## 4. Live Pipeline SHAP Confirmation (Task 2 — Track B Gate Check)

**SQL_injection Top-5 SHAP Features:**

| Feature | Mean |SHAP| | Domain Match |
|---|---:|---|
{sqli_rows}

**Domain overlap (SQLi):** `{results["sqli_overlap"]}` — {len(results["sqli_overlap"])}/5 expected features in top-5

**DDoS_ICMP Control Top-5 SHAP Features:**

| Feature | Mean |SHAP| | Domain Match |
|---|---:|---|
{icmp_rows}

**Overall Track B Verdict:** `{results["overall"]}`

{"✅ App-layer feature restoration confirmed. SHAP attributions for SQLi are domain-appropriate." if results["overall"] == "PASS" else "⚠️ Partial or failing SHAP domain alignment. See audit panel for details."}

![SHAP Audit Panel](sqli_shap_audit_panel.png)
"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"[APPENDED] Track B confirmation → {report_path}")


if __name__ == "__main__":
    results = run_sqli_shap_audit()
    append_to_track_b_report(results)
