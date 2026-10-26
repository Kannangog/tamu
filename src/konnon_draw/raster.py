"""Small PIL rasteriser used for evaluation (distances) and contact sheets."""
import numpy as np
from PIL import Image, ImageDraw

from .renderer import actions_to_paths


def draw_tile(actions, size=128, width=2):
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    for pts in actions_to_paths(actions):
        px = [(x * (size - 1), y * (size - 1)) for x, y in pts]
        if len(px) == 1:
            x, y = px[0]
            d.ellipse([x - width, y - width, x + width, y + width], fill=0)
        else:
            d.line(px, fill=0, width=width, joint="curve")
    return img


def rasterize(actions, size=32):
    """Flattened float vector in [0, 1]; 1 = ink. Drawn at 4x then box-downsampled."""
    big = draw_tile(actions, size=size * 4, width=6)
    arr = 1.0 - np.asarray(big.resize((size, size), Image.BOX), dtype=np.float32) / 255.0
    return arr.reshape(-1)
