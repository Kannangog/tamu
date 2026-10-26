# Honest Failures & Unmet Targets Log

Documents real measured weaknesses, benchmark failures, edge-case bottlenecks, and quality limitations with quantitative evidence.

---

### FAILURE-000: Initial Benchmark Environment Module
- **Issue**: `ModuleNotFoundError: No module named 'psutil'` during initial environment benchmark.
- **Root Cause**: Virtualenv lacked psutil package.
- **Resolution**: Installed `psutil==7.2.2` via `uv pip install psutil --python .venv/Scripts/python.exe`.
- **Evidence**: Verified memory inspection operational (`Total: 7.33 GB, Available: 2.20 GB`).

---

### FAILURE-001: General Model Scribble Prior (Previous Iteration)
- **Issue**: Model outputs dense horizontal scribbles rather than sharp contours on open-ended text prompts.
- **Root Cause**: With only 3.88M parameters trained for 8 epochs on CPU on 1,400 samples, the model converged to the mean spatial distribution of human doodles without resolving high-frequency structural contours.
- **Evidence**: `outputs_general/eval/benchmark_contact_sheet.png` shows center-weighted scribble clusters.
- **Planned Resolution in v0**: Staged generation (layout $\to$ guide $\to$ sketch $\to$ ink $\to$ shade) with canvas conditioning so each stage conditions on the rasterized state of the previous stage.
