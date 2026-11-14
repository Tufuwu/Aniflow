"""Adapt local slices using AniFlow's stitcher and YOLO, never touching originals."""
import json
import hashlib
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


def presentation_regions(image, boxes):
    """Keep detected artwork whole, then group small neighbouring reading beats.

    White-gutter boundaries are removed when a complete detected panel crosses
    them. Unlike sliding analysis windows, these are never cut at a fixed height.
    Every source row is retained in the provenance intervals.
    """
    import numpy as np
    from pipeline.cropper import _find_gap_regions
    thumb = np.asarray(image.resize((80,image.height)).convert('RGB'))
    white = (thumb.min(axis=(1,2)) > 248)
    boundaries = [0] + [(a+b)//2 for a,b in _find_gap_regions(white,12)] + [image.height]
    boundaries = sorted(set(y for y in boundaries if y in {0,image.height} or
                            not any(b['y'] < y < b['y']+b['h'] for b in boxes)))
    atoms = [(a,b) for a,b in zip(boundaries,boundaries[1:]) if b > a]
    regions, start, end, content = [], None, None, 0
    def append(a,b):
        related = [box for box in boxes if box['y'] < b and box['y']+box['h'] > a]
        regions.append({'x':0,'y':a,'w':image.width,'h':b-a,
                        'boundary_uncertain':len(related) != 1,
                        'boundary_method':'natural_gutters_and_complete_detector_boxes'})
    for i,(a,b) in enumerate(atoms):
        occupied = int((~white[a:b]).sum()) + 24
        boundary_card = i < 3 or i >= len(atoms)-2
        if start is not None and (boundary_card or content+occupied > 2400):
            append(start,end); start,end,content = None,None,0
        if boundary_card:
            append(a,b)
            continue
        start = a if start is None else start
        end,content = b,content+occupied
        if content >= 1050:
            append(start,end); start,end,content = None,None,0
    if start is not None:
        append(start,end)
    return regions


def compact_white_gutters(image, max_gap=28):
    """Remove only completely white rows, retaining a reversible source row map."""
    import numpy as np
    arr = np.asarray(image.convert('RGB'))
    white = (arr == 255).all(axis=(1,2))
    spans, i = [], 0
    while i < len(white):
        j = i+1
        while j < len(white) and white[j] == white[i]:
            j += 1
        if not white[i] or j-i <= max_gap:
            spans.append([i,j])
        else:
            spans.append([i,i+max_gap//2])
            spans.append([j-max_gap//2,j])
        i = j
    output = Image.fromarray(np.concatenate([arr[a:b] for a,b in spans]))
    return output, spans


def prepare_panels(store, progress=None):
    # Lazy imports keep text review usable without the video/ML dependency stack.
    from pipeline.stitcher import stitch_images
    from pipeline.panel_extractor_yolo import get_global_extractor

    imported = json.loads((store.root / "inputs.json").read_text(encoding="utf-8"))
    if imported["input_kind"] != "raw_slices":
        raise StoryError("当前输入已经是分镜，无需再次检测")
    key = digest({"version": 3, "inputs": imported})
    root = store.root / "preparation" / key
    root.mkdir(parents=True, exist_ok=True)
    stitched = stitch_images([p["path"] for p in imported["panels"]],
                             SimpleNamespace(stitch_dir=str(root)), progress)
    detector = get_global_extractor()
    boxes = []
    with Image.open(stitched) as image:
        width, height = image.size
        # Sliding windows avoid YOLO shrinking an entire 100k-pixel strip to 1024.
        window_height, overlap = 3600, 1200
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
        regions = presentation_regions(image, boxes)
        for index, box in enumerate(regions):
            path = root / f"panel_{index + 1:05d}.png"
            original = image.crop((box["x"],box["y"],box["x"]+box["w"],box["y"]+box["h"]))
            original_path = root / f"source_{index+1:05d}.png"
            original.save(original_path)
            prepared, row_map = compact_white_gutters(original)
            prepared.save(path)
            panels.append({"panel_id": str(index + 1), "path": str(path), "raw_path": str(path),
                           "source_page": stitched, "x": box["x"], "y": box["y"],
                           "width": box["w"], "height": box["h"],
                           "boundary_uncertain": box["boundary_uncertain"],
                           "derivation":{"kind":"white_gutters_only","original_crop":str(original_path),
                                         "original_sha256":hashlib.sha256(original_path.read_bytes()).hexdigest(),
                                         "retained_rows":row_map,"source_bbox":box},
                           "detected_boxes": [b for b in boxes if b["y"] < box["y"] + box["h"] and b["y"] + b["h"] > box["y"]]})
    atomic_json(root / "detection.json", {"source": imported, "stitched": stitched, "panels": panels,
                                          "detected_boxes": boxes, "coverage": "complete_vertical_strip"})
    result = {**imported, "input_kind": "detected_panels", "raw_inputs": imported["panels"],
              "panels": panels, "detection_manifest": str(root / "detection.json")}
    atomic_json(store.root / "inputs.json", result)
    return result
