# Konnon Drawing Model v0.1 — Evaluation & Release Gates Report

Generated on: 2026-10-26T10:33:03.083762  
Git commit: `unknown`  
Model: `DrawingGRU(hidden_dim=128, embedding_dim=32)`

## 1. Release Gates Summary

| Gate | Description | Pass Condition | Status |
|---|---|---|---|
| **Gate 0** | Data / Renderer | All audited examples parse and render | PASS |
| **Gate 1** | Training | Runs without NaNs and validates convergence | PASS |
| **Gate 2** | Generation | Structurally valid and beats random baseline | PASS |
| **Gate 3** | Generalization | Generalizes to held-out validation drawings | FAIL |
| **Gate 4** | Export | SVG exports cleanly and elements are editable | PASS |
| **Gate 5** | Governance | Datasets, licenses, and intended use verified | PASS |

## 2. Quantitative Generation Metrics

- **Model Render Validity Rate**: 100.0% (Random baseline: 57.8%)
- **Model Natural Completion Rate**: 100.0% (Random baseline: 100.0%)
- **Training Epochs**: 20
- **Initial Training Loss**: 3.9723
- **Final Validation Loss**: 2.1911

## 3. Stroke Statistics

- **Average Actions per Drawing**: 5.41
- **Average Paths per Drawing**: 1.12
- **Average Path Length**: 1.2295 (canvas units)
- **Pen-Up Ratio**: 2.1%

## 4. Strengths and Limitations

### Strengths
- Deterministic, standards-compliant SVG rendering decoupled from model inference.
- Quantized categorical tokenization allows strict vocabulary control without coordinate drift.
- Constrained autoregressive decoding ensures illegal token transitions (such as unanchored LINE) are prevented.
- Complete reproducible pipeline tested on CPU laptop architecture.

### Known Limitations (v0.1)
- Coordinate quantization (32 bins) causes coarse lines on complex shapes.
- Model is unconditional next-action prediction; prompt-conditioned drawing requires stage B/C conditioning.
- Does not generalize to arbitrary natural language or anime illustration without conditioned hierarchical Transformer and expanded illustration dataset.
