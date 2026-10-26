# Design Decisions & Assumptions Log

Documents all technical decisions, architecture choices, and trade-offs.

---

### DECISION-001: Autonomous Project Structure & Backward Compatibility
- **Context**: The prompt outlines a modular layout (`src/konnon_draw/v2/`, `skills/`, `gym/`, `model/`, `train/`, `eval/`, `teacher/`, `studio/`) while requiring strict backward compatibility with existing v1 and earlier v2 imports.
- **Decision**: Implement modules under the target layout (`v2/`, `skills/`, `gym/`, etc.) and provide re-export bridges in `src/konnon_draw/schema_v2.py`, `renderer_v2.py`, etc., so all existing 85 unit tests and checkpoints continue to work without breakage.
- **Why**: Ensures backward compatibility without breaking existing pipelines while cleanly satisfying the new architecture requirements.
- **Alternatives Considered**: Modifying existing paths in place, which would break existing import paths across tests and scripts.

---

### DECISION-002: Process Memory & Resource Guard
- **Context**: The target machine has 8 GB total RAM and ~2.2 GB available physical RAM. Memory must remain $< 1.2$ GB RSS.
- **Decision**: Installed `psutil==7.2.2` in the virtualenv to reliably query Working Set memory across all training and generation workers. Set max batch sizes to 16 with sequence lengths bounded dynamically.
- **Why**: Avoids Windows pagefile swapping and out-of-memory process termination.
- **Alternatives Considered**: Using Windows ctypes API directly (fragile across child processes).

---

### DECISION-003: Pure Python/Pillow/Numpy Graphics Rendering
- **Context**: Canvas rendering needs antialiasing, variable width curves, custom brushes, and SVG export on CPU Windows.
- **Decision**: Rely on Pillow and NumPy with supersampling rather than native C extensions like pycairo or skia-python.
- **Why**: Zero dependency installation friction, 100% deterministic cross-platform execution, zero native DLL build issues on Windows.
- **Alternatives Considered**: `pycairo` or `skia-python`, which frequently have wheel compatibility problems on Windows CPU environments.

---

### DECISION-004: Multimodal Cross-Attention Conditioning via Flattened Patch Grid
- **Context**: The transformer model needs to condition on both text instructions and the rendered visual state of the previous drawing stage without exceeding 1.2 GB RAM.
- **Decision**: Implemented `CanvasEncoder` using a lightweight 4-stage convolutional downsampler ($128 \times 128 \to 8 \times 8$), creating 64 spatial patch tokens + 1 `[CANVAS_CLS]` global summary token with learned 2D positional embeddings. Unified multimodal context is formed by concatenating $[e_{\text{stage}} \,;\, H_{\text{text}} \,;\, H_{\text{canvas}}]$.
- **Why**: Standard Transformer cross-attention blocks can simultaneously attend to semantic text tokens, active stage identity, and 2D canvas locations with $O(L_{\text{dec}} \cdot L_{\text{ctx}})$ complexity and low memory footprint (~480 MB RSS during training).
- **Alternatives Considered**: ViT patch projection from scratch (slower convergence, higher memory) or full dual cross-attention layers (doubles parameter count).

---

### DECISION-005: Preserving Legacy DrawingGRU Compatibility
- **Context**: Existing v1 models and tests expected `from konnon_draw.model import DrawingGRU`, but Phase 4 restructured `src/konnon_draw/model/` into a package.
- **Decision**: Maintained `DrawingGRU` in `src/konnon_draw/model/gru.py` and exported it directly from `konnon_draw.model.__init__.py`. Configured `web_server.py` to auto-detect checkpoint type (GRU v1 vs Staged Transformer v0).
- **Why**: Completely preserves 100% backward compatibility for all legacy tests, sampling scripts, and checkpoints while introducing the new staged transformer architecture.

---

### DECISION-006: Hardware Stylus Handling via W3C Pointer Events
- **Context**: The studio needs to capture pressure and tilt from drawing tablets and styluses.
- **Decision**: Implemented native Pointer Events (`pointerdown`, `pointermove`, `pointerup`) on `#drawCanvas` querying `e.pressure`, `e.tiltX`, and `e.tiltY`, alongside a live HUD display and an endpoint `/api/record_stylus_stroke`.
- **Why**: Cross-browser, zero-dependency, works with Wacom, Microsoft Surface, iPad Pencil, and standard mouse/touch fallbacks.

---

### DECISION-007: Storing Scaled Gym Transitions in JSONL and Streaming Images to Stay Under 150 MB RAM
- **Context**: Scaling the synthetic gym to 10,000+ transitions with $128 \times 128 \times 3$ uint8 numpy arrays in memory would require $\sim 560$ MB of raw array buffers, risking RAM limit violation ($< 1.2$ GB RSS) on the 8 GB laptop.
- **Decision**: Saved all intermediate stage canvases as PNGs in `data/processed/staged_gym_10k/canvases/` and stored metadata and token sequences in JSONL files (`train.jsonl`, `val.jsonl`, `test.jsonl`). `StagedDrawingDataset` lazily loads canvas images on demand.
- **Why**: Keeps process RSS memory flat at $\sim 135$ MB during generation and under 250 MB during DataLoader batch iteration.
- **Alternatives Considered**: In-memory numpy concatenation (exceeds RAM budget) or HDF5 / SQLite (adds dependency and concurrency overhead on Windows).

---

### DECISION-008: Deterministic Seeding for Builtin Semantic Text Tower
- **Context**: In Phase 8 testing, different tests changing global PyTorch seed altered the frozen text encoder weights because `nn.TransformerEncoderLayer` was initialized before `torch.manual_seed(42)` in `FrozenBuiltinTextTower.__init__`.
- **Decision**: Scoped `torch.manual_seed(42)` at the start of `FrozenBuiltinTextTower.__init__` and restored the caller's previous RNG state in a `finally:` block.
- **Why**: Guarantees identical deterministic embeddings across all runs and prevents hidden state contamination of other tests.

---

### DECISION-009: Causal Conditioning Evaluation at Drawing Offset (Prefix Length 10)
- **Context**: Probing conditioning KL divergence directly after `[BOS_TOKEN]` (step 0) measured near-zero KL divergence because every valid sequence in the grammar deterministically begins with `[TOKEN_SET_LAYER, LAYER_OFFSET + 4, ...]`.
- **Decision**: Evaluated next-token distribution KL divergence after the standard layer/brush initialization prefix (length 10), where actual coordinate and stroke choices are made.
- **Why**: Measures the model's true conditional sensitivity to text prompt, canvas state, and stage index, yielding $D_{\text{KL}} = 1.75$ (text), $6.39$ (canvas), and $0.15$ (stage).

---

### DECISION-010: EOS Loss Weighting ($w_{\text{EOS}} = 4.0$) and Dynamic Length Prior Bias
- **Context**: In sequences of 100-150 tokens, EOS accounts for $< 0.7\%$ of target tokens, causing unweighted cross-entropy to under-predict EOS at true end positions (1.4% probability, rank 7-8).
- **Decision**: Added class weight $w_{\text{EOS}} = 4.0$ to cross-entropy loss during training and calibrated a soft linear length-prior bias ($\alpha = 0.04$ after minimum 16 tokens) during autoregressive sampling.
- **Why**: Restores natural stage completion without relying on arbitrary hard sequence cutoffs.

---

### DECISION-011: Unified Pointer Events Engine & Multi-Model Concurrent Serving in Web Studio
- **Context**: In the Live Drawing Studio (`http://127.0.0.1:8080`), user drawing was unresponsive when directly interacting with the canvas, and AI generation failed with `AttributeError` whenever a v2 checkpoint was supplied to `start_webview.py`.
- **Decision**:
  1. Consolidated web client drawing handlers into a single Pointer Events engine (`pointerdown`, `pointermove`, `pointerup`, `pointercancel`) supporting mouse, stylus (with real pressure & tilt), and touch with `touch-action: none;`. Strokes immediately render to canvas, populate user prefix actions, and expose the "✨ Complete" action.
  2. Enhanced `DrawingStudioHandler` to concurrently load and serve both the v1 DrawingGRU checkpoint (`outputs_qd/checkpoints/best.pt`) for `/api/generate` and the v2 Staged Transformer (`outputs_v2/laptop/checkpoints/best.pt` or `outputs_v2/scaled/full/checkpoint_best.pt`) for `/api/generate_staged` and `/api/compare_ab`.
- **Why**: Eliminates dual-listener event competition, allows immediate freehand sketching at any time, and ensures all webview endpoints operate concurrently without crashing.


