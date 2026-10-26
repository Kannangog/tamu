from pathlib import Path
from konnon_draw.renderer import render_svg

demo = [
    {"type": "MOVE", "x": 0.15, "y": 0.75},
    {"type": "LINE", "x": 0.5, "y": 0.15},
    {"type": "LINE", "x": 0.85, "y": 0.75},
    {"type": "PEN_UP"},
    {"type": "MOVE", "x": 0.3, "y": 0.55},
    {"type": "LINE", "x": 0.7, "y": 0.55},
    {"type": "END"},
]

Path("outputs/samples").mkdir(parents=True, exist_ok=True)
Path("outputs/samples/renderer_demo.svg").write_text(
    render_svg(demo), encoding="utf-8"
)
print("Wrote outputs/samples/renderer_demo.svg")
