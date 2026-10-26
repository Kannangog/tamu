"""STEP 1: Overfit Sanity Check for Konnon Draw Staged Transformer (v0).

Takes 32 gym transitions, trains the model on only those until train loss is near zero (cap 300 epochs),
then decodes greedily with teacher forcing OFF and measures token-exact match rate and pixel error.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from konnon_draw.gym.dataset import StagedDrawingDataset, generate_staged_gym_data
from konnon_draw.model.stroke_transformer import KonnonStagedTransformer
from konnon_draw.model.text_encoder import FrozenTextEncoder
from konnon_draw.model.sampling import sample_stage_tokens
from konnon_draw.v2.renderer import render_document_v2
from konnon_draw.v2.tokens import (
    VOCAB_SIZE,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    token_sequence_to_doc,
)


def run_overfit_sanity(
    num_transitions: int = 32,
    max_epochs: int = 200,
    target_loss: float = 0.0005,
    out_dir: str = "outputs_v2/overfit",
    seed: int = 42,
) -> Dict[str, Any]:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"

    print("=" * 70)
    print(f"STEP 1: OVERFIT SANITY CHECK ({num_transitions} transitions, cap {max_epochs} epochs)")
    print("=" * 70)

    # 1. Generate 32 transitions (11 drawings * 3 stages ~ 33 transitions, slice 32)
    raw_items = generate_staged_gym_data(num_samples=12, canvas_res=128, seed=seed)
    items = raw_items[:num_transitions]
    print(f"Loaded {len(items)} training transitions for overfit verification.")

    canvas_res = 128
    max_seq_len = 192
    dataset = StagedDrawingDataset(items, canvas_res=canvas_res, max_tokens=max_seq_len, augment_stroke_order=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. Initialize Model
    d_model = 128
    text_encoder = FrozenTextEncoder(model_type="builtin", d_model=d_model, device=device)
    model = KonnonStagedTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_layers=3,
        n_heads=4,
        d_ff=384,
        max_seq_len=max_seq_len,
        canvas_res=canvas_res,
        dropout=0.0,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # 3. Overfit Training Loop
    t0 = time.time()
    best_loss = float("inf")
    epoch_reached = 0

    print("\nEpoch | Train Loss | LR       | Elapsed (s)")
    print("-" * 46)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            captions = batch["caption"]
            canvases = batch["canvas"].to(device)
            tokens = batch["tokens"].to(device)
            stages = batch["stage_idx"].to(device)

            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            optimizer.zero_grad()
            with torch.no_grad():
                text_emb = text_encoder.encode_text(captions, device=device)

            context = model.build_multimodal_context(
                text_embeddings=text_emb,
                canvas=canvases,
                stage_idx=stages,
            )

            logits, _, _ = model(inputs, context=context)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, total_batches)
        epoch_reached = epoch
        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 25 == 0 or avg_loss <= target_loss or epoch == 1:
            curr_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"{epoch:5d} | {avg_loss:10.5f} | {curr_lr:8.2e} | {elapsed:6.1f}s", flush=True)

        if avg_loss <= target_loss:
            print(f"--> Reached target loss ({avg_loss:.5f} <= {target_loss}) at epoch {epoch}!")
            break

    # Save overfit checkpoint
    ckpt_path = out_p / "model_overfit.pt"
    torch.save({
        "epoch": epoch_reached,
        "loss": best_loss,
        "model_state_dict": model.state_dict(),
    }, ckpt_path)
    print(f"Saved overfit model to {ckpt_path}")

    # 4. Greedy Decoding Evaluation (Teacher Forcing OFF)
    print("\n" + "=" * 70)
    print("EVALUATING GREEDY DECODING (TEACHER FORCING OFF)")
    print("=" * 70)
    model.eval()

    token_matches = 0
    total_token_dist = 0
    pixel_mae_list = []
    pixel_mse_list = []
    eval_records = []

    for idx, item in enumerate(items):
        caption = item["caption"]
        stage_idx = int(item["stage_idx"])
        canvas_arr = item["canvas_arr"]
        
        # Ground truth clean tokens exactly as supervised in dataset (up to first EOS)
        ds_toks = dataset[idx]["tokens"].tolist()
        gt_tokens = [t for t in ds_toks if t != PAD_TOKEN]

        canvas_t = torch.from_numpy(canvas_arr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        stage_t = torch.tensor([stage_idx], dtype=torch.long, device=device)

        with torch.no_grad():
            text_emb = text_encoder.encode_text([caption], device=device)
            context = model.build_multimodal_context(
                text_embeddings=text_emb,
                canvas=canvas_t,
                stage_idx=stage_t,
            )

        # Greedy decoding (temperature=0.0)
        pred_tokens = sample_stage_tokens(
            model=model,
            context=context,
            max_tokens=max_seq_len,
            temperature=0.0,
            top_k=0,
            device=device,
            seed=seed + idx,
        )

        # Check exact token match
        is_exact_match = (pred_tokens == gt_tokens)
        if is_exact_match:
            token_matches += 1

        # Render ground truth vs predicted
        gt_doc = token_sequence_to_doc(gt_tokens)
        pred_doc = token_sequence_to_doc(pred_tokens)

        gt_img = render_document_v2(gt_doc, width=128, height=128, supersample=1, seed=42)
        pred_img = render_document_v2(pred_doc, width=128, height=128, supersample=1, seed=42)

        gt_arr = np.array(gt_img, dtype=np.float32) / 255.0
        pred_arr = np.array(pred_img, dtype=np.float32) / 255.0

        mae = float(np.mean(np.abs(pred_arr - gt_arr)))
        mse = float(np.mean((pred_arr - gt_arr) ** 2))
        pixel_mae_list.append(mae)
        pixel_mse_list.append(mse)

        eval_records.append({
            "idx": idx,
            "category": item["category"],
            "stage_idx": stage_idx,
            "gt_tokens_len": len(gt_tokens),
            "pred_tokens_len": len(pred_tokens),
            "exact_match": is_exact_match,
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "gt_img": gt_img,
            "pred_img": pred_img,
        })

        if idx < 6 or is_exact_match:
            status_str = "MATCH" if is_exact_match else "DIFF"
            print(f"[{idx:02d}] {status_str} | GT Len: {len(gt_tokens):3d} | Pred Len: {len(pred_tokens):3d} | MAE: {mae:.5f} | MSE: {mse:.5f}")

    exact_match_rate = token_matches / len(items)
    mean_mae = float(np.mean(pixel_mae_list))
    mean_mse = float(np.mean(pixel_mse_list))

    print("\n" + "-" * 70)
    print(f"OVERFIT RESULTS (32 transitions):")
    print(f"  Exact Token Match Rate : {exact_match_rate * 100:.2f}% ({token_matches}/{len(items)})")
    print(f"  Mean Pixel Error (MAE) : {mean_mae:.6f} (normalized [0, 1])")
    print(f"  Mean Pixel Error (MSE) : {mean_mse:.6f}")
    print(f"  Final Train Loss       : {best_loss:.5f}")
    print(f"  Epochs Required        : {epoch_reached}")
    print("-" * 70)

    # 5. Save Overfit Contact Sheet (GT vs Pred)
    contact_w = 4 * 128
    contact_h = 8 * (128 + 24)
    sheet = Image.new("RGB", (contact_w, contact_h), (240, 240, 245))
    draw = ImageDraw.Draw(sheet)

    # Display 8 pairs (GT left, Pred right)
    for row in range(min(8, len(eval_records))):
        rec = eval_records[row]
        y = row * (128 + 24)
        sheet.paste(rec["gt_img"], (20, y))
        sheet.paste(rec["pred_img"], (160, y))
        diff_arr = np.abs(np.array(rec["pred_img"]).astype(np.int32) - np.array(rec["gt_img"]).astype(np.int32))
        diff_img = Image.fromarray(np.clip(diff_arr * 4, 0, 255).astype(np.uint8))
        sheet.paste(diff_img, (300, y))

        lbl = f"[{rec['idx']}] GT / Pred / Diff | MAE: {rec['mae']:.4f} | {'EXACT' if rec['exact_match'] else 'DIFF'}"
        draw.text((20, y + 128 + 4), lbl, fill=(20, 20, 30))

    sheet_path = out_p / "overfit_contact_sheet.png"
    sheet.save(sheet_path)
    print(f"Saved contact sheet to {sheet_path}")

    # 6. Save JSON Evidence
    json_data = {
        "num_transitions": num_transitions,
        "epochs_trained": epoch_reached,
        "final_loss": best_loss,
        "exact_match_rate": exact_match_rate,
        "exact_matches": token_matches,
        "mean_pixel_mae": mean_mae,
        "mean_pixel_mse": mean_mse,
        "records": [
            {
                "idx": r["idx"],
                "category": r["category"],
                "stage_idx": r["stage_idx"],
                "exact_match": r["exact_match"],
                "mae": r["mae"],
                "mse": r["mse"],
            }
            for r in eval_records
        ],
    }
    metrics_path = out_p / "overfit_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved overfit metrics evidence to {metrics_path}")

    return json_data


if __name__ == "__main__":
    run_overfit_sanity()
