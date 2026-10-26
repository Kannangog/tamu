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

---

### FAILURE-006: KV-Cache Step Zero Duplicate BOS Sampling Bug
- **Issue**: Initial greedy decoding during Step 1 overfit sanity achieved only 21.88% exact match rate despite train loss reaching 0.002.
- **Root Cause**: In `sample_stage_tokens`, the initial forward pass on `[BOS]` at `start_pos=0` updated the KV cache but discarded logits, and the generation loop at `step=1` re-fed `[BOS]` a second time. This presented the causal decoder with an illegal `[BOS, BOS]` sequence prefix.
- **Resolution**: Fixed `sample_stage_tokens` to start sampling directly at step 0 from initial BOS logits, feeding each emitted token strictly once into the KV cache.
- **Evidence**: Greedy decoding exact token match rate immediately jumped from 21.88% to 100.00% (32/32 transitions bit-for-bit identical, pixel MAE 0.000000).

---

### FAILURE-007: EOS Class Imbalance & Suppression in Unweighted Cross-Entropy
- **Issue**: In sequences of 100-160 tokens, EOS accounts for $< 0.7\%$ of targets. Standard cross-entropy pushed EOS probability at true termination boundaries down to only 1.44% (mean logit rank 7.4), causing all uncalibrated greedy rollouts to hit `max_seq_len`.
- **Resolution**: Implemented class weighting $w_{\text{EOS}} = 4.0$ in cross-entropy loss and added an explicit length-prior logit bias ($\alpha = 0.04$ after 16 tokens).
- **Evidence**: Audit in `outputs_v2/audit/loss_eos_audit.json` showed uncalibrated EOS probability rising from 0.00% to 3.87% and natural termination restored.

---

### FAILURE-008: Web Studio Drawing Unresponsiveness & Model Loading Mismatch
- **Issue**: In the Live Drawing Studio web interface (`http://127.0.0.1:8080`), attempts to draw produced nothing on canvas. Category clicks / initial load threw unhandled 500 errors and closed the HTTP connection.
- **Root Cause**:
  1. `start_webview.py` was started with `--checkpoint outputs_v2/laptop/checkpoints/best.pt`. Because v2 staged transformer checkpoints lack `"model_config"`, `DrawingStudioHandler.model` remained `None`. Any call to `/api/generate` threw `AttributeError: 'NoneType' object has no attribute 'eval'`.
  2. Freehand interactive drawing on the HTML5 canvas was blocked by `if (!isSketchMode) return;` in `mousedown`. Pointer event listeners (`pointerdown`/`pointermove`) recorded points for the HUD telemetry bar but never rendered strokes to the 2D canvas context.
- **Resolution**:
  1. Implemented `DrawingStudioHandler.get_gru_model()` with auto-discovery and fallback loading of the v1 GRU checkpoint (`outputs_qd/checkpoints/best.pt`). Both v1 and v2 models remain loaded simultaneously so all endpoints (`/api/generate`, `/api/generate_staged`, `/api/generate_text`, `/api/record_stylus_stroke`) function reliably.
  2. Replaced divided mouse/pointer listeners in `app.js` with a unified Pointer Events drawing engine (`pointerdown`, `pointermove`, `pointerup`, `pointercancel`) supporting mouse, stylus (with pressure & tilt), and touch with `touch-action: none;`. Strokes render live on the canvas and automatically populate user prefix actions with one-click "✨ Complete" AI continuation.
- **Evidence**: Verified with automated endpoint test script returning HTTP 200 across all 6 API routes, clean JS syntax validation via Node.js, and 123 green unit tests.




