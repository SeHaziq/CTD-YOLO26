"""Importable CTD/YOLO26 detection, with no project-specific paths or file output."""
import json
import os
from pathlib import Path
import numpy as np
from .weights import resolve_weights_dir, verify_weights


class MangaDetector:
    """Reusable detector. Input is uint8 BGR; returned masks have original size.

    Instantiate once per worker. Concurrent calls on one instance are not supported.
    No OCR, translation, inpainting, training, or automatic downloads are performed.
    """

    def __init__(self, weights_dir=None, *, mode="hybrid", device="auto", verify=True):
        if mode not in {"hybrid", "ctd"}:
            raise ValueError("mode must be 'hybrid' or 'ctd'")
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")
        directory = resolve_weights_dir(weights_dir)
        verify_weights(directory, mode=mode, check_hash=verify)
        import torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use device='auto' or device='cpu'")
        self.mode = mode
        cfg = json.loads(Path(__file__).with_name("config.json").read_text())
        cfg.update(device=self.device, ctd_weights=str(directory / "ctd.pt"),
                   yolo_weights=str(directory / "yolo26n-manga.pt"))
        from ._adapter import HybridDetector
        self._detector = HybridDetector(cfg, load_yolo=mode == "hybrid")

    def detect(self, image):
        """Return grouped_blocks, raw detections, raw_mask and refined_mask.

        Raw masks are uint8 probabilities (0..255); refined masks are binary
        uint8 (0/255). Grouped lines may include synthetic rectangle fallbacks.
        Raw detector scores are preserved; grouped confidence is not calibrated.
        """
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
            raise ValueError("Expected a nonempty uint8 BGR image with three channels")
        result = self._detector(image, backend="yolo26" if self.mode == "hybrid" else "ctd")
        return {
            "version": "1.0.0", "mode": self.mode, "device": self.device,
            "width": image.shape[1], "height": image.shape[0],
            "grouped_blocks": [b.to_dict() for b in result["grouped"]],
            "raw_blocks": result["raw_blocks"],
            "raw_block_confidence": result["raw_block_confidence"],
            "raw_block_detector_sources": result["raw_block_detector_sources"],
            "raw_block_language_sources": result["raw_block_language_sources"],
            "raw_lines": result["raw_lines"], "raw_line_confidence": result["raw_line_confidence"],
            "raw_mask": result["mask"], "refined_mask": result["refined"],
            "note": result["note"],
        }

    __call__ = detect
