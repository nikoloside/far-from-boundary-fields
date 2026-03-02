#!/usr/bin/env python3
"""
実験結果を集約するスクリプト。
data/results/ 以下の metrics.csv / metrics.json を読み、
一覧テーブル (CSV / Markdown) を出力。
"""

import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "results"
OUTPUT_FILE = DATA_ROOT / "aggregated_table.json"


def load_metrics(exp_dir: Path) -> dict | None:
    """実験フォルダから metrics を読み込む"""
    for name in ("metrics.json", "metrics.csv"):
        p = exp_dir / name
        if p.exists():
            if p.suffix == ".json":
                return json.loads(p.read_text())
            # CSV は簡易対応（必要なら拡張）
            return {"source": str(p), "raw": p.read_text()}
    return None


def main():
    rows = []
    for exp in sorted(DATA_ROOT.iterdir()):
        if not exp.is_dir() or exp.name.startswith("."):
            continue
        m = load_metrics(exp)
        if m:
            rows.append({"experiment": exp.name, **m})
    # 簡易出力（実際の metrics に合わせて調整）
    if rows:
        output = OUTPUT_FILE
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2))
        print(f"Wrote {output}")
    else:
        print("No metrics found. Add metrics.json to each experiment folder.")


if __name__ == "__main__":
    main()
