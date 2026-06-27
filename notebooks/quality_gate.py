"""
quality_gate.py - CI Quality Gate
Pass/Fail + SQLite logging
"""

import sqlite3
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
PRODUCTS_DB = BASE_DIR / "data" / "products.db"

# ── Thresholds ─────────────────────────────────────────────────
THRESHOLDS = {
    "precision@5"     : 0.30,
    "recall@5"        : 0.70,
    "f1@5"            : 0.40,
    "mrr"             : 0.70,
    "ndcg@5"          : 0.70,
    "faithfulness"    : 0.70,
    "answer_relevancy": 0.70,
    "context_recall"  : 0.60,
}


def setup_eval_table():
    conn = sqlite3.connect(PRODUCTS_DB)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date  TEXT,
            metric    TEXT,
            score     REAL,
            threshold REAL,
            passed    INTEGER,
            run_id    TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_to_sqlite(all_scores, run_id):
    conn     = sqlite3.connect(PRODUCTS_DB)
    cur      = conn.cursor()
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for metric, score in all_scores.items():
        threshold = THRESHOLDS.get(metric, 0.5)
        passed    = 1 if score >= threshold else 0
        cur.execute("""
            INSERT INTO eval_runs (run_date, metric, score, threshold, passed, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_date, metric, score, threshold, passed, run_id))

    conn.commit()
    conn.close()
    print(f"[SQLite]  Logged to eval_runs — run_id: {run_id}")


def check_production_gate(retrieval_scores, ragas_scores):
    print("=" * 60)
    print("  CI QUALITY GATE")
    print("=" * 60)

    all_scores = {**retrieval_scores, **ragas_scores}
    run_id     = datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
    passed     = []
    failed     = []

    print(f"\n  Run ID : {run_id}\n")
    print(f"  {'Metric':<22} {'Score':>8} {'Threshold':>10} {'Status':>8}")
    print(f"  {'-'*52}")

    for metric, threshold in THRESHOLDS.items():
        score = all_scores.get(metric, 0.0)
        if score >= threshold:
            status = "✅ PASS"
            passed.append(metric)
        else:
            status = "❌ FAIL"
            failed.append(metric)
        print(f"  {metric:<22} {score:>8.4f} {threshold:>10.4f} {status:>8}")

    # Log to SQLite
    setup_eval_table()
    log_to_sqlite(all_scores, run_id)

    # Final verdict
    print("\n" + "=" * 60)
    if failed:
        print("  🚨 DEPLOYMENT BLOCKED")
        print(f"  {len(failed)} metric(s) below threshold:")
        for m in failed:
            print(f"    ❌ {m} = {all_scores.get(m, 0):.4f}  (need {THRESHOLDS[m]})")
        print("=" * 60)
        return False
    else:
        print("  ✅ DEPLOYMENT APPROVED")
        print(f"  All {len(passed)} metrics passed!")
        print("=" * 60)
        return True


def show_eval_history():
    conn = sqlite3.connect(PRODUCTS_DB)
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT run_id, metric, score, passed
            FROM eval_runs
            ORDER BY run_date DESC
            LIMIT 30
        """)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            print("No history yet.")
            return

        print("\n" + "=" * 60)
        print("  EVAL HISTORY")
        print("=" * 60)
        print(f"  {'Run ID':<22} {'Metric':<22} {'Score':>7} {'Pass':>5}")
        print(f"  {'-'*55}")
        for run_id, metric, score, passed in rows:
            status = "✅" if passed else "❌"
            print(f"  {run_id:<22} {metric:<22} {score:>7.4f} {status:>5}")

    except Exception as e:
        print(f"Error: {e}")
        conn.close()


if __name__ == "__main__":
    # Actual scores from our eval runs
    retrieval_scores = {
        "precision@5" : 0.3500,
        "recall@5"    : 0.9167,
        "f1@5"        : 0.5000,
        "mrr"         : 0.9750,
        "ndcg@5"      : 0.8739,
    }

    ragas_scores = {
        "faithfulness"    : 1.0000,
        "answer_relevancy": 1.0000,
        "context_recall"  : 0.7000,
    }

    result = check_production_gate(retrieval_scores, ragas_scores)
    show_eval_history()