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

---

### FAILURE-004: Gate 2 Scripted Demos Honest Assessment (`outputs/hand_demos/gate2_contact_sheet.png`)
- **Piece 1 (Articulated Mannequin)**:
  - *What works*: 3 distinct 3D poses (stand, reach, run) rendered in 3 separate stages (layout wireframe, guide capsules, ink contours) with camera perspective and depth ordering.
  - *Weakness*: Limb cylinders are rendered as parallel boundary pairs without closed joint end-caps, leaving small visual gaps at elbow and knee intersections.
- **Piece 2 (Loomis Head Rotations)**:
  - *What works*: Accurate geometric foreshortening across 3 rotations (front yaw 0°, 3/4 yaw 35°, profile yaw 80°); jaw line and nose wedge turn convincingly with perspective.
  - *Weakness*: Cranium circle and brow guide lines are drawn in thin pencil lines that are low-contrast on high-DPI displays.
- **Piece 3 (Dynamic Pose Sequence)**:
  - *What works*: Proportions are strictly preserved across 4 kinematic frames (stand $\to$ walk $\to$ run $\to$ crouch) using the adult 7.5-head archetype.
  - *Weakness*: Limbs terminate at circular ankle/wrist joints; missing articulated foot and hand silhouette blocks.
- **Piece 4 (Pencil Still Life)**:
  - *What works*: Unified single light direction (top-left) correctly drives ground cast shadow projections and directional hatching across sphere, cylinder, and cube.
  - *Weakness*: Sphere core shadow boundary uses a planar quadrilateral polygon rather than wrapping around the 3D spherical curvature.

---

### FAILURE-005: Phase 4 Staged Model EOS Density & High-Frequency Detailing
- **Issue**: In all 8 evaluated prompts (`outputs_v2/eval/staged_evaluation_sheet.png`), each stage generated exactly 160 tokens without emitting an early `EOS_TOKEN`, hitting the generation limit.
- **Root Cause**: In 5 epochs on 180 synthetic samples, cross-entropy loss converged smoothly from 53.1 down to 6.43, and the model learned token syntax (drawing valid commands without parse crashes). However, the probability mass for `EOS_TOKEN` did not overcome continuation tokens during autoregressive sampling before `max_tokens=160`.
- **Visual Artifacts**:
  - The model produces abstract, flowing stroke gestalts that adapt between stages (conditioning is active), but lacks crisp, recognizable outlines (e.g., recognizable bicycle spokes or realistic facial features).
  - High-frequency details remain noisy/scribbled at this training scale ($4.6\text{M}$ params, 5 epochs, CPU synthetic gym).
- **Resolution Path**:
  - Requires longer pretraining (100k+ transitions, e.g. using `configs/staged_v0_full_gpu.yaml` on rented GPU).
  - Explicit EOS logit bias or length penalty during inference sampling.



