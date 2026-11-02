# Manga Detector

A reusable Python library for manga text-region detection and text masks. It combines the selected manga-fine-tuned YOLO26n model with CTD's line detector, U-Net, grouping and mask refinement. It performs no OCR, translation, inpainting, training or UI work.

## Install in your translator project

```bash
pip install "git+https://github.com/SeHaziq/CTD-YOLO26.git@v1.0.0"
```

Use Python 3.10–3.12. Install the appropriate PyTorch/Torchvision CUDA build for your machine first. The tested environment was Python 3.11, Torch 2.7.1+cu128, Torchvision 0.22.1 and Ultralytics 8.4.198. CPU inference is also supported. For local development: `pip install -e .`.

## Use

```python
import cv2
from manga_detector import MangaDetector

detector = MangaDetector(weights_dir="/path/to/weights", device="auto")
result = detector.detect(cv2.imread("page.jpg"))  # uint8 BGR, H x W x 3

blocks = result["grouped_blocks"]
mask = result["refined_mask"]  # original-size uint8 binary mask, 0/255
```

Create one detector per processing worker and reuse it. Concurrent calls on one instance are not supported. Input pages are not changed and the library writes no image files. `mode="ctd"` selects CTD-only inference without loading YOLO or requiring its weight file. `device` accepts `auto`, `cuda` or `cpu`. The library does not reset your root logger, temporary directory or import path. Ultralytics may apply its own OpenCV/CPU-thread settings; configure threading in the host application if necessary.

Other outputs include `raw_mask` (uint8 ink probability), `raw_blocks`, raw detector confidence/provenance, `raw_lines` and line confidence. Grouped lines may include synthesized rectangle fallbacks. The model is one-class: it does not semantically distinguish ordinary text and SFX. Grouped confidence is not calibrated.

## Model weights

The two selected files remain in the local `weights/` folder, but are intentionally excluded from Git and wheels:

| File | SHA-256 |
|---|---|
| `ctd.pt` | `1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb` |
| `yolo26n-manga.pt` | `f0dca2b99fbba0fe06d94d7cb24225277b67f5b9e870f88ce23a9692fc946a7b` |

Supply `weights_dir`, set `MANGA_DETECTOR_WEIGHTS`, or use the default user cache. Checkouts with a populated local `weights/` folder use it automatically. Inference never downloads weights automatically; checksums are verified before loading.

After a permitted GitHub Release is published, upload the fine-tuned `yolo26n-manga.pt` there and use:

```python
from manga_detector import download_weights

weights = download_weights(
    release_base_url="https://github.com/SeHaziq/CTD-YOLO26/releases/download/v1.0.0"
)
```

CTD is downloaded from the original upstream release; the fine-tuned YOLO file comes from your release. Both downloads are checked against the pinned hashes. You can also use `manga-detector-download --release-base-url URL`. The URL above works only after the fine-tuned weight asset has been published; the source tag alone does not contain model binaries. Until then, supply your local weights directory.

## Scope and publishing

The selected hybrid reached 89.41% block F1 at IoU .5 on a frozen ten-page new-title test, versus CTD's 84.66%, recovering 6/15 SFX blocks versus zero. This small test does not establish broad manga accuracy; difficult SFX and grouping errors remain. Paired GPU median latency was approximately 67 ms/page hybrid versus 47 ms/page CTD, excluding file I/O.

No datasets, annotations, experiments, tests, generated previews, training scripts, launchers or virtual environments belong in this repository. The vendored `_ctd` modules are required runtime model/postprocessing code.

**License:** project integration code is AGPL-3.0; vendored CTD remains GPL-3.0. This is not an MIT/proprietary component. Preserve the included licenses and notices when integrating or redistributing. Source publication does not itself settle upstream checkpoint or training-image rights; confirm redistribution rights before publishing the fine-tuned weights. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
