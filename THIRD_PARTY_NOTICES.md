# Third-party notices

## CTD

Source: https://github.com/dmMaze/comic-text-detector

Commit: `440b978563c71b758e31aaa315d100faba1efa2f`. License: GPL-3.0, retained in `LICENSES/GPL-3.0.txt`. Vendored YOLOv5 files retain their upstream Ultralytics notices.

The vendored runtime was modified to namespace imports under `manga_detector._ctd`, remove training/export/demo entrypoints and dependencies, use `packaging.version` instead of deprecated `pkg_resources`, and avoid global logger/warning changes. The numeric inference architecture, preprocessing and grouping/refinement algorithms are retained. The original CTD checkpoint is unchanged.

Official checkpoint origin: https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1

The source license is not an independently verified grant for all underlying checkpoint/training data. Confirm applicable model redistribution terms before publishing mirrored weights.

## YOLO26

Runtime: Ultralytics 8.4.198, https://github.com/ultralytics/ultralytics, AGPL-3.0 / separately available Enterprise terms. The integration code uses AGPL-3.0; CTD's GPL notices remain in force. Ultralytics' license documentation: https://www.ultralytics.com/license

The selected YOLO26n checkpoint was initialized from https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt and fine-tuned privately on 12 manga pages. Original asset hash: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`. Selected fine-tune hash: `f0dca2b99fbba0fe06d94d7cb24225277b67f5b9e870f88ce23a9692fc946a7b`.

Training images and annotations are not distributed. Their redistribution/training rights and the upstream model terms must be resolved before distributing the fine-tuned weights publicly. Keeping binaries out of Git does not establish permission to publish them as Release assets.

Other installed dependencies retain their own licenses. No DBNet++, U-Net++, research data or experimental checkpoints are included.
