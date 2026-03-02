# 4. MFCD 対称版の定義

## 定義

- **MFCD (一方向)**: volume-weighted, Pred→GT（フラグメントカバレッジ重視）
- **SymMFCD**: ½(MFCD(A→B) + MFCD(B→A))

##  deliverables

- 全主要テーブルで MFCD と SymMFCD を併記
- トイ例（2D または 3D）：
  - Case 1: 大片 OK・小片欠落 → MFCD 高、SymMFCD 中
  - Case 2: 全体ずれ・全片あり → 両方高

## 結果出力先

`data/results/mfcd_definition/`
