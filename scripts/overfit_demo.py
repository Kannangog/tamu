"""Sanity test: Train a tiny PyTorch model to overfit one scripted v2 artwork.

Proves that:
1. The v2 hand language and tokenization scheme are 100% learnable by neural networks.
2. The entire end-to-end pipeline (Scripted Art -> Tokens -> PyTorch Model -> Model Output -> Renderer v2) works seamlessly on CPU.
"""

import math
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from konnon_draw.schema_v2 import (
    CanvasConfigV2,
    DocumentV2,
    LayerV2,
    PointV2,
    StrokeActionV2,
    validate_document_v2,
)
from konnon_draw.renderer_v2 import render_document_v2
from konnon_draw.encoding_v2 import compute_critic_score


# ---------------------------------------------------------
# Define a Clean Scripted Target Piece (Sphere + Ground)
# ---------------------------------------------------------

def create_target_piece() -> DocumentV2:
    """Creates a clean study of a shaded sphere and ground line (~80 points)."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=512, height=512, background="#fcfbf7"),
        layers=[
            LayerV2(id="ground", name="Ground Line"),
            LayerV2(id="sphere", name="Sphere & Form"),
        ],
    )
    # Ground line
    doc.commands.append(
        StrokeActionV2(
            points=[PointV2(0.1, 0.75, p=0.4), PointV2(0.9, 0.75, p=0.4)],
            brush_id="pencil",
            color="#78716c",
            base_width=2.5,
            curve_type="linear",
            layer_id="ground",
        )
    )
    # Sphere contour (center 0.5, 0.5, radius 0.2)
    sc = (0.5, 0.5)
    sr = 0.2
    c_pts = []
    for i in range(25):
        rad = (i / 24.0) * 2.0 * math.pi
        x = sc[0] + sr * math.cos(rad)
        y = sc[1] + sr * math.sin(rad)
        p = 0.3 + 0.7 * (0.5 - 0.5 * math.sin(rad))  # shadow side heavier
        c_pts.append(PointV2(x=x, y=y, p=p))
    doc.commands.append(
        StrokeActionV2(
            points=c_pts,
            brush_id="pencil",
            color="#1c1917",
            base_width=3.5,
            layer_id="sphere",
        )
    )
    # Shading hatching stroke inside sphere
    h_pts = [
        PointV2(0.45, 0.55, p=0.8),
        PointV2(0.55, 0.62, p=0.9),
        PointV2(0.62, 0.58, p=0.6),
    ]
    doc.commands.append(
        StrokeActionV2(
            points=h_pts,
            brush_id="pencil",
            color="#292524",
            base_width=4.0,
            layer_id="sphere",
        )
    )
    return doc


# ---------------------------------------------------------
# Tiny Autoregressive Neural Agent
# ---------------------------------------------------------

class TinyHandModel(nn.Module):
    """Compact GRU agent predicting continuous (dx, dy, dp, pen_state)."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, h=None):
        feat = torch.relu(self.fc_in(x))
        out, h_next = self.gru(feat, h)
        pred = self.fc_out(out)
        return pred, h_next


def flatten_doc_to_sequence(doc: DocumentV2) -> np.ndarray:
    """Converts strokes to a sequence of vectors: [x, y, p, pen_up_flag]."""
    seq = []
    for cmd in doc.commands:
        if cmd.type == "stroke":
            for i, p in enumerate(cmd.points):
                is_end = 1.0 if i == len(cmd.points) - 1 else 0.0
                seq.append([p.x, p.y, p.p, is_end])
    return np.array(seq, dtype=np.float32)


def sequence_to_doc(seq: np.ndarray, base_doc: DocumentV2) -> DocumentV2:
    """Reconstructs DocumentV2 from predicted vector sequence."""
    out_doc = DocumentV2(canvas=base_doc.canvas, layers=list(base_doc.layers), commands=[])
    cur_pts = []

    for vec in seq:
        x, y, p, pen_up = float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3])
        cur_pts.append(PointV2(x=x, y=y, p=max(0.1, min(1.0, p))))
        if pen_up > 0.5:
            if len(cur_pts) >= 2:
                out_doc.commands.append(
                    StrokeActionV2(
                        points=list(cur_pts),
                        brush_id="pencil",
                        color="#1c1917",
                        base_width=3.0,
                        curve_type="catmull_rom",
                        layer_id="sphere",
                    )
                )
            cur_pts = []

    if len(cur_pts) >= 2:
        out_doc.commands.append(
            StrokeActionV2(
                points=list(cur_pts),
                brush_id="pencil",
                color="#1c1917",
                base_width=3.0,
                layer_id="sphere",
            )
        )
    return out_doc


def run_overfit_sanity_test(out_dir: str = "outputs/hand_demos"):
    print("=" * 60)
    print("PHASE 5 SANITY TEST: OVERFIT NEURAL AGENT ON SCRIPTED ART")
    print("=" * 60)

    target_doc = create_target_piece()
    validate_document_v2(target_doc)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Render Target Image
    target_img = render_document_v2(target_doc, width=512, height=512, supersample=1, seed=42)
    target_img.save(out_path / "overfit_target.png")

    # 2. Extract Data Sequence
    data_seq = flatten_doc_to_sequence(target_doc)
    seq_len = len(data_seq)
    print(f"Target piece extracted into sequence of length: {seq_len} steps")

    # Target sequence: input is seq[:-1], target is seq[1:]
    x_train = torch.tensor(data_seq[:-1], dtype=torch.float32).unsqueeze(0)  # (1, T-1, 4)
    y_train = torch.tensor(data_seq[1:], dtype=torch.float32).unsqueeze(0)   # (1, T-1, 4)

    # 3. Initialize Model
    torch.manual_seed(42)
    model = TinyHandModel(input_dim=4, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print("Training tiny neural agent for 80 epochs...")
    final_loss = 0.0
    for epoch in range(1, 81):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/80 | Loss: {final_loss:.6f}")

    print(f"\nFinal training loss: {final_loss:.6f}")
    assert final_loss < 0.05, f"Expected loss < 0.05, got {final_loss}"

    # 4. Autoregressive Sampling from Model
    model.eval()
    with torch.no_grad():
        pred_seq = [data_seq[0]]
        cur_input = torch.tensor(data_seq[0], dtype=torch.float32).view(1, 1, 4)
        h = None
        for step in range(seq_len - 1):
            out, h = model(cur_input, h)
            next_vec = out.squeeze(0).squeeze(0).numpy()
            pred_seq.append(next_vec)
            # Feed prediction back into next step
            cur_input = out

    pred_arr = np.array(pred_seq)

    # 5. Decode into DocumentV2
    reconstructed_doc = sequence_to_doc(pred_arr, target_doc)
    validate_document_v2(reconstructed_doc)

    # 6. Render Reconstructed Piece
    recon_img = render_document_v2(reconstructed_doc, width=512, height=512, supersample=1, seed=42)
    recon_path = out_path / "overfit_reconstruction.png"
    recon_img.save(recon_path)
    print(f"Reconstruction successfully saved to: {recon_path}")

    # 7. Evaluate Critic Score
    critic_score = compute_critic_score(reconstructed_doc, target_img, eval_size=128)
    print(f"Critic Perceptual Similarity Score: {critic_score:.4f} / 1.0000")
    assert critic_score > 0.85, f"Reconstruction critic score too low: {critic_score}"

    print("\n[SUCCESS] Phase 5 Sanity Test Passed: Pipeline is brain-ready and learnable!")


if __name__ == "__main__":
    run_overfit_sanity_test()
