"""Upload-only comparison UI for local quality enhancement."""
import hashlib
from io import BytesIO
import json
import re
from tempfile import TemporaryDirectory
from pathlib import Path

import cv2
from flask import Blueprint, jsonify, render_template, request, send_from_directory, url_for
from PIL import Image

from pipeline.quality_enhancement import (
    CACHE_DIR, EnhancementOptions, EnhancementService, atomic_bytes, decode_image, required_resolution,
)


def create_quality_blueprint(cache_dir=CACHE_DIR):
    bp = Blueprint("quality", __name__)
    root = Path(cache_dir).resolve()

    @bp.get("/quality-enhancement")
    def page():
        return render_template("quality_enhancement.html")

    @bp.post("/api/quality-enhancement/preview")
    def preview():
        try:
            options = EnhancementOptions.parse(json.loads(request.form.get("options", '{"enabled":true}')))
            target = (int(request.form["width"]), int(request.form["height"]))
            required_resolution(target, options.max_zoom)
            upload = request.files["image"]
            data = upload.stream.read(24 * 1024**2 + 1)
            if len(data) > 24 * 1024**2:
                return jsonify(error="图片超过 24 MiB"), 413
            with Image.open(BytesIO(data)) as probe:
                if probe.format not in {"JPEG", "PNG", "WEBP", "BMP"} or probe.width * probe.height > 12_000_000:
                    raise ValueError("请上传不超过 1200 万像素的 JPEG、PNG、WebP 或 BMP")
            with TemporaryDirectory() as folder:
                source = Path(folder) / "input"
                source.write_bytes(data)
                item = EnhancementService(root).image(source, target, options)
            key = hashlib.sha256(data).hexdigest()
            before = root / "previews" / (key + ".png")
            if not before.is_file():
                ok, encoded = cv2.imencode(".png", decode_image(data))
                if not ok:
                    raise ValueError("原图预览编码失败")
                atomic_bytes(before, encoded.tobytes())
            item["before_url"] = url_for(".asset", kind="previews", key=key)
            item["after_url"] = (url_for(".asset", kind="images", key=Path(item["path"]).stem)
                                 if item["enhanced"] else item["before_url"])
            del item["path"]
            return jsonify(item)
        except (ValueError, OSError, KeyError, Image.DecompressionBombError, cv2.error) as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503

    @bp.get("/api/quality-enhancement/assets/<kind>/<key>.png")
    def asset(kind, key):
        if kind not in {"images", "previews"} or not re.fullmatch("[a-f0-9]{64}", key):
            return jsonify(error="预览不存在"), 404
        return send_from_directory(root / kind, key + ".png", mimetype="image/png", max_age=86400)

    return bp
