#!/usr/bin/env bash
set -euo pipefail
pytest -q
python -m konnon_draw.objects_gen --per-class 800
python -m konnon_draw.training --epochs 40
python -m konnon_draw.evaluation
python -m konnon_draw.sampling --category all --n 6 --out outputs/samples/showcase
