# Konnon Drawing Model - evaluation

- checkpoint: outputs_qd_skip/checkpoints/best.pt (epoch 9), temperature 0.5, 16 samples/class
- render validity: 1.000 (random baseline 0.156)
- natural completion: 0.885
- kNN recognizability: generated 0.760 vs held-out real 0.853

| category | valid | recognized | diversity (ref) | NN dist |
|---|---|---|---|---|
| bed | 1.00 | 0.88 | 12.03 (12.16) | 8.49 |
| bicycle | 1.00 | 0.69 | 12.47 (12.58) | 9.64 |
| car | 1.00 | 0.69 | 12.48 (11.32) | 9.02 |
| chair | 1.00 | 1.00 | 12.95 (11.74) | 8.62 |
| house | 1.00 | 0.69 | 13.85 (12.69) | 9.59 |
| table | 1.00 | 0.62 | 12.51 (11.74) | 8.50 |

## Gates
- 0_data_renderer: 18000/18000 examples parse and render (run pytest for unit tests)
- 1_training: no NaN in log
- 2_generation: PASS
- 3_generalization: held-out split: val loss, kNN reference and NN distance reported; unseen categories not tested
- 4_export: 96/96 saved JSON re-render to identical SVG
- 5_governance: manual review required before any public release

Note: kNN recognizability = nearest-neighbour vote of rasterised samples against training drawings; a proxy for 'does it look like the requested category', not a human rating.
Contact sheet: contact_sheet.png (red = invalid, orange = length-limit fallback).