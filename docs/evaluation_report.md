# Konnon Drawing Model v0.1 — Evaluation & Release Gates Report

Generated on: 2026-10-26T10:34:09.430100  
Git commit: `8a0e9f8`  
Model: `DrawingGRU(hidden_dim=128, embedding_dim=32)`

## 1. Release Gates Summary

| Gate | Description | Pass Condition | Status |
|---|---|---|---|
| **Gate 0** | Data / Renderer | All audited examples parse and render | PASS |
| **Gate 1** | Training | Runs without NaNs and validates convergence | PASS |
| **Gate 2** | Generation | Structurally valid and beats random baseline | PASS |
| **Gate 3** | Generalization | Held-out examples & categories evaluated separately | PASS |
| **Gate 4** | Export | SVG exports cleanly and elements are editable | PASS |
| **Gate 5** | Governance | Datasets, licenses, and intended use verified | PASS |

## 2. Quantitative Generation Metrics

- **Model Render Validity Rate**: 100.0% (Random baseline: 57.8%)
- **Model Natural Completion Rate**: 100.0% (Random baseline: 100.0%)
- **Training Epochs**: 20
- **Initial Training Loss**: 3.9723
- **Final Validation Loss**: 2.1911

## 3. Held-out Category Generalization (Gate 3)

| Category | Examples | Total Loss | Action Loss | Coordinate Loss |
|---|---|---|---|---|
| `zigzag` | 30 | 2.0381 | 0.01 | 2.0281 |
| `rectangle` | 28 | 2.1326 | 0.0066 | 2.126 |
| `triangle` | 32 | 2.1353 | 0.0591 | 2.0762 |
| `cross` | 30 | 2.3258 | 0.0292 | 2.2966 |

## 4. Stroke Statistics

- **Average Actions per Drawing**: 5.41
- **Average Paths per Drawing**: 1.12
- **Average Path Length**: 1.2295 (canvas units)
- **Pen-Up Ratio**: 2.1%

## 5. Strengths and Limitations

### Strengths
- Deterministic, standards-compliant SVG rendering decoupled from model inference.
- Quantized categorical tokenization allows strict vocabulary control without coordinate drift.
- Constrained autoregressive decoding ensures illegal token transitions (such as unanchored LINE) are prevented.
- Complete reproducible pipeline tested on CPU laptop architecture.

### Known Limitations (v0.1)
- Coordinate quantization (32 bins) causes coarse lines on complex shapes.
- Model is unconditional next-action prediction; prompt-conditioned drawing requires stage B/C conditioning.
- Does not generalize to arbitrary natural language or anime illustration without conditioned hierarchical Transformer and expanded illustration dataset.
