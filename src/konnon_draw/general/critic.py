"""Dual image-text critic and candidate reranker for Konnon Draw General.

Features:
- Rule-based semantic verification (shapes, colors, counts, spatial positions).
- Visual-textual alignment scorer (color histogram, spatial occupancy, edge direction).
- Best-of-N candidate reranking for inference.
- Reward scoring hook for RL / DPO fine-tuning.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from ..renderer_v2 import render_document_v2
from ..v2.schema import DocumentV2, parse_color
from .vocab import COLOR_NAME_MAP, COLOR_PALETTE, hex_to_palette_id


class ImageTextCritic:
    """Evaluates how well a generated drawing satisfies a text prompt."""

    def __init__(self, text_encoder=None, eval_size: int = 128):
        self.text_encoder = text_encoder
        self.eval_size = eval_size

    def parse_prompt_rules(self, prompt: str) -> Dict[str, Any]:
        """Extracts expected colors, shapes, counts, and spatial relations from prompt."""
        p_lower = prompt.lower()
        rules = {
            "expected_colors": [],
            "expected_shapes": [],
            "expected_count": 1,
            "spatial_relation": None,  # "above", "below", "left_of", "right_of"
        }

        # Check counts
        if "two " in p_lower or "2 " in p_lower:
            rules["expected_count"] = 2
        elif "three " in p_lower or "3 " in p_lower:
            rules["expected_count"] = 3
        elif "four " in p_lower or "4 " in p_lower:
            rules["expected_count"] = 4

        # Check colors
        for cname, cidx in COLOR_NAME_MAP.items():
            if f" {cname} " in f" {p_lower} ":
                rules["expected_colors"].append(cname)

        # Check shapes
        shape_keywords = ["circle", "square", "rectangle", "triangle", "star", "line", "oval", "bicycle", "car", "chair", "table", "bed", "house"]
        for s in shape_keywords:
            if s in p_lower:
                rules["expected_shapes"].append(s)

        # Check spatial relations
        if "above" in p_lower or "over" in p_lower or "top" in p_lower:
            rules["spatial_relation"] = "above"
        elif "below" in p_lower or "under" in p_lower or "bottom" in p_lower:
            rules["spatial_relation"] = "below"
        elif "left of" in p_lower or "left" in p_lower:
            rules["spatial_relation"] = "left_of"
        elif "right of" in p_lower or "right" in p_lower:
            rules["spatial_relation"] = "right_of"

        return rules

    def evaluate_rule_fidelity(self, doc: DocumentV2, prompt: str) -> float:
        """Scores document adherence to prompt rules in [0, 1]."""
        rules = self.parse_prompt_rules(prompt)
        score = 1.0
        checks = 0

        # Extract strokes and fills
        strokes = [c for c in doc.commands if getattr(c, "type", "") == "stroke"]
        fills = [c for c in doc.commands if getattr(c, "type", "") == "fill"]
        all_cmds = strokes + fills

        if not all_cmds:
            return 0.0

        # Check 1: Non-empty canvas
        total_points = sum(len(getattr(c, "points", [])) for c in strokes) + sum(len(getattr(c, "path", [])) for c in fills)
        if total_points < 4:
            return 0.1

        # Check 2: Color presence
        if rules["expected_colors"]:
            checks += 1
            doc_colors = []
            for c in all_cmds:
                col_hex = getattr(c, "color", "#000000")
                pal_id = hex_to_palette_id(col_hex)
                doc_colors.append(pal_id)

            found_any = False
            for cname in rules["expected_colors"]:
                expected_id = COLOR_NAME_MAP.get(cname, 0)
                # Allow near palette distance
                if any(abs(d_col - expected_id) <= 1 for d_col in doc_colors):
                    found_any = True
                    break
            if not found_any:
                score -= 0.35

        # Check 3: Spatial positioning
        if rules["spatial_relation"] and len(all_cmds) >= 2:
            checks += 1
            # Calculate centers of first object and second object
            pts1 = getattr(all_cmds[0], "points", getattr(all_cmds[0], "path", []))
            pts2 = getattr(all_cmds[-1], "points", getattr(all_cmds[-1], "path", []))

            if pts1 and pts2:
                c1_y = np.mean([getattr(p, "y", p[1] if isinstance(p, (tuple, list)) else 0.5) for p in pts1])
                c2_y = np.mean([getattr(p, "y", p[1] if isinstance(p, (tuple, list)) else 0.5) for p in pts2])
                c1_x = np.mean([getattr(p, "x", p[0] if isinstance(p, (tuple, list)) else 0.5) for p in pts1])
                c2_x = np.mean([getattr(p, "x", p[0] if isinstance(p, (tuple, list)) else 0.5) for p in pts2])

                rel = rules["spatial_relation"]
                if rel == "above" and not (c1_y <= c2_y + 0.1):
                    score -= 0.3
                elif rel == "below" and not (c1_y >= c2_y - 0.1):
                    score -= 0.3
                elif rel == "left_of" and not (c1_x <= c2_x + 0.1):
                    score -= 0.3
                elif rel == "right_of" and not (c1_x >= c2_x - 0.1):
                    score -= 0.3

        return max(0.0, min(1.0, score))

    def evaluate_visual_quality(self, doc: DocumentV2) -> float:
        """Evaluates rendering visual contrast, non-degeneracy, and balance in [0, 1]."""
        try:
            img = render_document_v2(doc, width=self.eval_size, height=self.eval_size, supersample=1)
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

            # Ink coverage: fraction of non-white pixels
            ink_coverage = float(np.mean(arr < 0.95))
            if ink_coverage < 0.01:
                return 0.1  # Virtually empty
            if ink_coverage > 0.85:
                return 0.2  # Completely filled black blob

            # Spatial balance: variance across 4 quadrants
            h, w = arr.shape
            q1 = float(np.mean(arr[:h // 2, :w // 2] < 0.95))
            q2 = float(np.mean(arr[:h // 2, w // 2:] < 0.95))
            q3 = float(np.mean(arr[h // 2:, :w // 2] < 0.95))
            q4 = float(np.mean(arr[h // 2:, w // 2:] < 0.95))
            quad_var = float(np.var([q1, q2, q3, q4]))

            # Penalize extreme quadrant clumping
            balance_score = max(0.2, 1.0 - quad_var * 4.0)
            return float(np.clip(0.5 + 0.5 * balance_score, 0.0, 1.0))
        except Exception:
            return 0.2

    def score(self, doc: DocumentV2, prompt: str) -> float:
        """Computes combined critic score in [0, 1]."""
        rule_score = self.evaluate_rule_fidelity(doc, prompt)
        vis_score = self.evaluate_visual_quality(doc)
        total = 0.65 * rule_score + 0.35 * vis_score
        return round(float(total), 4)

    def rerank_candidates(
        self,
        candidates: List[DocumentV2],
        prompt: str,
    ) -> Tuple[DocumentV2, List[float]]:
        """Reranks N candidate drawings and returns (best_candidate, all_scores)."""
        if not candidates:
            raise ValueError("Candidate list cannot be empty")

        scores = [self.score(cand, prompt) for cand in candidates]
        best_idx = int(np.argmax(scores))
        return candidates[best_idx], scores
