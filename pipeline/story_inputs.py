"""Adapt local slices using AniFlow's stitcher and YOLO, never touching originals."""
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from pipeline.story_bible import StoryError, atomic_json, digest


def reading_regions(image, target_height=1800):
    """Cover the complete strip, including text/quiet beats missed by YOLO.

    White gutters are preferred boundaries. Long continuous artwork is split
    with overlap and explicitly flagged, so a reviewer can repair boundaries.
    No content is filtered as blank or advertising during input preparation.
    """
    import numpy as np
    from pipeline.cropper import _find_gap_regions
    thumb = np.asarray(image.resize((80, image.height)).convert("RGB"))
    white = (thumb.mean(axis=(1, 2)) > 245) & (thumb.std(axis=1).max(axis=1) < 8)
    gaps = _find_gap_regions(white, 12)
    boundaries = [0]
    for start, end in gaps:
        midpoint = (start + end) // 2
        if midpoint - boundaries[-1] >= 240:
            boundaries.append(midpoint)
    if len(boundaries) > 1 and image.height - boundaries[-1] < 240:
        boundaries.pop()
    boundaries.append(image.height)
    regions = []
    for start, end in zip(boundaries, boundaries[1:]):
        y = start
        while y < end:
            bottom = min(end, y + target_height)
            uncertain = end - start > target_height
            regions.append({"x": 0, "y": max(start, y - (160 if y > start else 0)),
                            "w": image.width, "h": bottom - max(start, y - (160 if y > start else 0)),
                            "boundary_uncertain": uncertain})
            y = bottom
    return regions


def prepare_panels(store, progress=None):
    # Lazy imports keep text review usable without the video/ML dependency stack.
    from pipeline.stitcher import stitch_images
    from pipeline.panel_extractor_yolo import get_global_extractor

    imported = json.loads((store.root / "inputs.json").read_text(encoding="utf-8"))
    if imported["input_kind"] != "raw_slices":
        raise StoryError("当前输入已经是分镜，无需再次检测")
    key = digest({"version": 2, "inputs": imported})
    root = store.root / "preparation" / key
    root.mkdir(parents=True, exist_ok=True)
    stitched = stitch_images([p["path"] for p in imported["panels"]],
                             SimpleNamespace(stitch_dir=str(root)), progress)
    detector = get_global_extractor()
    boxes = []
    with Image.open(stitched) as image:
        width, height = image.size
        # Sliding windows avoid YOLO shrinking an entire 100k-pixel strip to 1024.
        window_height, overlap = 1800, 600
        starts = list(range(0, height, window_height - overlap))
        for i, y in enumerate(starts):
            bottom = min(height, y + window_height)
            path = root / f"window_{i:04d}.png"
            image.crop((0, y, width, bottom)).save(path)
            detected = detector.detect_image_boxes(str(path))
            for box in detected:
                # Keep complete boxes where possible; a neighbouring overlapping
                # window supplies panels cut by a window boundary.
                if (box["y"] < 8 and y > 0) or (box["y"] + box["h"] > bottom - y - 8 and bottom < height):
                    continue
                boxes.append({**box, "y": box["y"] + y})
            if progress:
                progress(i + 1, len(starts), f"已检测阅读窗口 {i + 1}/{len(starts)}")
        boxes = sorted(detector._suppress_overlaps(boxes), key=lambda b: (b["y"], b["x"]))
        if not boxes:
            raise StoryError("没有检测到分镜，请使用 AniFlow 手工调整框后重新导入")
        panels = []
        regions = reading_regions(image)
        for index, box in enumerate(regions):
            path = root / f"panel_{index + 1:05d}.png"
            image.crop((box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"])).save(path)
            panels.append({"panel_id": str(index + 1), "path": str(path), "raw_path": str(path),
                           "source_page": stitched, "x": box["x"], "y": box["y"],
                           "width": box["w"], "height": box["h"],
                           "boundary_uncertain": box["boundary_uncertain"],
                           "detected_boxes": [b for b in boxes if b["y"] < box["y"] + box["h"] and b["y"] + b["h"] > box["y"]]})
    atomic_json(root / "detection.json", {"source": imported, "stitched": stitched, "panels": panels,
                                          "detected_boxes": boxes, "coverage": "complete_vertical_strip"})
    result = {**imported, "input_kind": "detected_panels", "raw_inputs": imported["panels"],
              "panels": panels, "detection_manifest": str(root / "detection.json")}
    atomic_json(store.root / "inputs.json", result)
    return result
