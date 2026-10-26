# Konnon Drawing Model - evaluation

- checkpoint: outputs_qd/checkpoints/best.pt (epoch 9), temperature 0.3, 16 samples/class
- render validity: 1.000 (random baseline 0.156)
- natural completion: 0.635
- kNN recognizability: generated 0.802 vs held-out real 0.853

| category | valid | recognized | diversity (ref) | NN dist |
|---|---|---|---|---|
| bed | 1.00 | 0.81 | 10.94 (12.16) | 7.39 |
| bicycle | 1.00 | 0.88 | 11.52 (12.58) | 8.38 |
| car | 1.00 | 0.88 | 10.77 (11.32) | 7.98 |
| chair | 1.00 | 0.94 | 10.50 (11.74) | 6.98 |
| house | 1.00 | 0.56 | 12.91 (12.69) | 9.39 |
| table | 1.00 | 0.75 | 11.70 (11.74) | 7.86 |

## Gates
- 0_data_renderer: 18000/18000 examples parse and render (run pytest for unit tests)
- 1_training: no NaN in log
- 2_generation: PASS
- 3_generalization: held-out split: val loss, kNN reference and NN distance reported; unseen categories not tested
- 4_export: 96/96 saved JSON re-render to identical SVG
- 5_governance: manual review required before any public release

Note: kNN recognizability = nearest-neighbour vote of rasterised samples against training drawings; a proxy for 'does it look like the requested category', not a human rating.
Contact sheet: contact_sheet.png (red = invalid, orange = length-limit fallback).