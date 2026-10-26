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

