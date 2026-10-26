# Konnon Drawing Model - evaluation

- checkpoint: outputs_qd_h384/checkpoints/best.pt (epoch 5), temperature 0.5, 16 samples/class
- render validity: 1.000 (random baseline 0.156)
- natural completion: 0.781
- kNN recognizability: generated 0.771 vs held-out real 0.853

| category | valid | recognized | diversity (ref) | NN dist |
|---|---|---|---|---|
| bed | 1.00 | 0.56 | 12.34 (12.16) | 8.54 |
| bicycle | 1.00 | 0.81 | 12.09 (12.58) | 8.91 |
| car | 1.00 | 1.00 | 12.05 (11.32) | 8.84 |
| chair | 1.00 | 1.00 | 12.87 (11.74) | 9.30 |
| house | 1.00 | 0.56 | 14.14 (12.69) | 10.36 |
| table | 1.00 | 0.69 | 12.55 (11.74) | 8.69 |

## Gates
- 0_data_renderer: 18000/18000 examples parse and render (run pytest for unit tests)
- 1_training: no NaN in log
- 2_generation: PASS
- 3_generalization: held-out split: val loss, kNN reference and NN distance reported; unseen categories not tested
- 4_export: 96/96 saved JSON re-render to identical SVG
- 5_governance: manual review required before any public release

Note: kNN recognizability = nearest-neighbour vote of rasterised samples against training drawings; a proxy for 'does it look like the requested category', not a human rating.
Contact sheet: contact_sheet.png (red = invalid, orange = length-limit fallback).