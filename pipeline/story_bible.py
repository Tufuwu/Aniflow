"""Immutable chapter revisions, original-image snapshots and model-call traces."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import threading
from datetime import datetime, timezone
from uuid import uuid4
from filelock import FileLock


class StoryError(ValueError):
    pass


class StoryConflict(StoryError):
    pass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class StoryBible:
    """One store per chapter; callers hold lock throughout read/modify/save.

    current.json is only a pointer. Every revision and reviewed paragraph remains
    available even when a new analysis or style is requested.
    """
    _locks: dict[str, FileLock] = {}
    _guard = threading.Lock()

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self._guard:
            self.lock = self._locks.setdefault(str(self.root), FileLock(self.root / ".chapter.lock"))

    def read(self, revision=None) -> dict:
        if revision is None:
            pointer = self.root / "current.json"
            if not pointer.exists():
                return {"schema_version": 1, "revision": None, "bible": None,
                        "plan": None, "script": None}
            revision = json.loads(pointer.read_text(encoding="utf-8"))["revision"]
        if not isinstance(revision, str) or len(revision) != 32 or any(
                c not in "0123456789abcdef" for c in revision):
            raise StoryError("无效的历史版本")
        path = self.root / "revisions" / f"{revision}.json"
        if not path.is_file():
            raise StoryError("历史版本不存在")
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, state: dict, action: str, expected_revision=None) -> dict:
        with self.lock:
            current = self.read()
            if current["revision"] != expected_revision:
                raise StoryConflict("内容已变化，请刷新后重试")
            result = copy.deepcopy(state)
            result.update(revision=uuid4().hex, parent_revision=current["revision"],
                          action=action, updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(self.root / "revisions" / f"{result['revision']}.json", result)
            atomic_json(self.root / "current.json", {"revision": result["revision"]})
            return result

    def snapshot(self, path) -> dict:
        path = Path(path).resolve()
        if not path.is_file():
            raise StoryError(f"原始分镜不存在：{path.name}")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        relative = Path("evidence") / f"{sha}{path.suffix.lower()}"
        target = self.root / relative
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            # Concurrent writers can only publish the exact same content.
            tmp = target.with_name(f".{uuid4().hex}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        return {"original_path": str(path), "snapshot_path": relative.as_posix(), "sha256": sha}

    def evidence_path(self, panel: dict) -> Path:
        path = (self.root / panel["snapshot_path"]).resolve()
        if not path.is_relative_to(self.root / "evidence") or not path.is_file():
            raise StoryError("原图证据不存在")
        if hashlib.sha256(path.read_bytes()).hexdigest() != panel["sha256"]:
            raise StoryError("证据校验失败：图片已变化")
        return path

    def history(self) -> list[dict]:
        # Follow the committed chain; an interrupted save may leave an orphan.
        out, state = [], self.read()
        while state.get("revision"):
            out.append({k: state.get(k) for k in
                        ("revision", "parent_revision", "action", "updated_at")})
            state = self.read(state["parent_revision"]) if state.get("parent_revision") else {}
        return out
