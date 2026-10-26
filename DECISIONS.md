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
