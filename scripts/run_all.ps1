# Run from the project root with the venv active:  .\scripts\run_all.ps1
$ErrorActionPreference = "Stop"
pytest -q
python -m konnon_draw.objects_gen --per-class 800
python -m konnon_draw.training --epochs 40        # ~15 s/epoch on a fast CPU; expect 2-3x on a laptop
python -m konnon_draw.evaluation
python -m konnon_draw.sampling --category all --n 6 --out outputs/samples/showcase
Write-Host "Done. Open outputs\eval\contact_sheet.png and outputs\eval\report.md"
