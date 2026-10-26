# A/B Test Report: Plain Discrete Decoding vs. Sub-bin Continuous Coordinates

- Model Checkpoint: `outputs_qd/checkpoints/best.pt`
- Samples Evaluated: 96 (16 per class across 6 classes)
- Temperature: 0.4 (Identical seeds per pair)

| Metric | Plain Discrete | Sub-bin Continuous | Delta |
|---|---|---|---|
| Validity Rate | 1.000 | 1.000 | 0.000 |
| Natural Completion Rate | 0.833 | 0.833 | 0.000 |
| kNN Recognizability | 0.823 | 0.854 | +0.031 |
| Mean Path Length | 4.645 | 4.596 | -0.049 |
| Sub-bin Resolution Rate | 0.0% | 99.8% | +99.8% |

Contact Sheet: `ab_comparison_contact_sheet.png`
