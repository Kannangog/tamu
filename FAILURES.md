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

---

### FAILURE-002: Naive PIL/NumPy Boundary in 1000-Stroke Render
- **Issue**: Initial 1000-stroke render took 10.52s, narrowly exceeding the 10.0s target.
- **Root Cause**: Converting the canvas back and forth between PIL and NumPy on every stroke added 8.7s of array memory copies.
- **Resolution**: Implemented lazy synchronization (`ensure_pil` and `sync_pil_to_canvas`), caching the PIL Drawing context across consecutive strokes of the same layer. Reduced render time to 1.82s.

---

### FAILURE-003: Brush Swatch Visual Weaknesses (Honest Inspection of outputs/hand/swatches.png)
- **Weakness 1 (Low-Pressure Faintness)**: At lowest pressure ($p=0.2$), `pencil` and `colored_pencil` strokes appear almost invisible on light backgrounds. While physically realistic for light graphite touches, it hurts legibility on standard sRGB displays.
- **Weakness 2 (Charcoal Grain Directionality)**: Charcoal particles are scattered with isotropic 2D Gaussian jitter rather than directional shearing along the stroke normal.
- **Weakness 3 (Marker Nib Angle)**: Marker stroke caps are rounded rather than angled chisel slant marks.

