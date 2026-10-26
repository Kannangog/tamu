# Konnon Drawing Model - evaluation

- checkpoint: outputs/checkpoints/best.pt (epoch 30), temperature 0.4, 16 samples/class
- render validity: 1.000 (random baseline 0.156)
- natural completion: 1.000
- kNN recognizability: generated 1.000 vs held-out real 1.000

| category | valid | recognized | diversity (ref) | NN dist |
|---|---|---|---|---|
| bed | 1.00 | 1.00 | 8.22 (9.09) | 3.20 |
| bicycle | 1.00 | 1.00 | 10.59 (11.17) | 7.02 |
| car | 1.00 | 1.00 | 9.01 (8.27) | 5.69 |
| chair | 1.00 | 1.00 | 8.77 (9.27) | 3.62 |
| house | 1.00 | 1.00 | 8.34 (9.78) | 4.40 |
| table | 1.00 | 1.00 | 7.65 (8.55) | 3.32 |

## Gates
- 0_data_renderer: 4800/4800 examples parse and render (run pytest for unit tests)
- 1_training: no NaN in log
- 2_generation: PASS
- 3_generalization: held-out split: val loss, kNN reference and NN distance reported; unseen categories not tested
- 4_export: 96/96 saved JSON re-render to identical SVG
- 5_governance: manual review required before any public release

Note: kNN recognizability = nearest-neighbour vote of rasterised samples against training drawings; a proxy for 'does it look like the requested category', not a human rating.
Contact sheet: contact_sheet.png (red = invalid, orange = length-limit fallback).