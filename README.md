# Konnon Drawing Model v0.2

A category-conditioned GRU that draws everyday objects as pen strokes (MOVE / LINE / PEN_UP / END on a 64x64 grid)
and exports editable SVG. CPU-only, no image-generation API. Categories in the bundled run:
**bicycle, bed, chair, table, house, car**.

## Quick start (Windows PowerShell, project root)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pillow matplotlib pytest
# PyTorch: CPU command from https://pytorch.org/get-started/locally/
pip install -e .
.\scripts\run_all.ps1
```
Draw something with the trained model:
```powershell
python -m konnon_draw.sampling --category bicycle --n 8 --temperature 0.4 --out outputs/samples/mine
python -m konnon_draw.sampling --category all --n 4 --seed 5
```
Each sample is saved as `.json` (actions) and `.svg` (open in a browser / Inkscape and edit the paths).
Temperature: 0.3-0.5 gives clean lines; 0.8+ gets wobbly (wheels especially). `--top-k 3` also tightens output.

## Two data sources
1. **Procedural objects (`objects_gen.py`)** - self-made, no licence issues, used for the bundled run.
   Parametric strokes with random proportions, mirroring, scale and shift. The model learns clean
   line-drawing structure, but only in the style of these templates.
2. **Quick, Draw! (`quickdraw.py`)** - real human doodles, for variety. Needs internet:
   ```powershell
   python -m konnon_draw.quickdraw --categories bicycle bed chair table house car --per-class 3000
   python -m konnon_draw.training --data data/processed/quickdraw.jsonl --out-dir outputs_qd --log-dir logs_qd
   python -m konnon_draw.evaluation --data data/processed/quickdraw.jsonl --checkpoint outputs_qd/checkpoints/best.pt --out outputs_qd/eval --log logs_qd/train_log.csv
   ```
   Only the first lines of each category are streamed (not the whole multi-hundred-MB file). A manifest with source,
   licence (CC BY 4.0 - attribution required, verify current terms), access date and transforms is written to `data/manifests/`.
   This path was tested on synthetic NDJSON only (the build sandbox could not reach Google's servers), not on the real download.
   Human doodles are far messier than the templates: expect rougher results and consider `--hidden-dim 384`, more epochs and a lower `--lr`.

## What changed since v0.1
- Category conditioning (embedding fed at every step and into the initial state).
- 64 coordinate bins (was 32); Gaussian-smoothed coordinate targets so near-misses cost less; cosine LR schedule.
- Hidden size 256, max length 160, batch 32 (about 390k parameters).
- Per-category evaluation incl. kNN "recognized as requested" score, contact sheet grouped by category.
- Memory-safe distance computation in evaluation (a naive version needs >11 GB RAM).

## Reading the results honestly
- Validity ~100% is mostly by construction (constrained decoder). Judge quality from `outputs/eval/contact_sheet.png`.
- kNN recognizability compares samples with training drawings, so it favours copies of the template style; it is a proxy, not a human rating.
- Chairs, tables, houses and beds come out clean; bicycle wheels and car wheels are the weak spot (closing a 20-point circle from memory).
- It draws only the categories it was trained on. Free-text prompts ("a red chair next to a lamp") are not supported.
- Anime or detailed art is a separate, much larger milestone.

## Layout
`schema.py` actions + validator | `renderer.py` SVG | `objects_gen.py`, `data_gen.py` (toy shapes), `quickdraw.py` data |
`encoding.py` | `model.py` | `training.py` | `sampling.py` | `evaluation.py` | `raster.py` | `tests/` (37 tests)

## Dataset Attribution & Licensing
- **Google Quick, Draw! Dataset**: The real stroke sequences used in `data/processed/quickdraw.jsonl` are sourced from the Quick, Draw! Dataset provided by **Google Creative Lab** under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.
  - Source: [Google Creative Lab Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
  - Raw Simplified NDJSON: `https://storage.googleapis.com/quickdraw_dataset/full/simplified/`
  - Access Date: 26 October 2026
  - Transformations & Filtering: Documented in [`data/manifests/quickdraw.manifest.json`](data/manifests/quickdraw.manifest.json).

