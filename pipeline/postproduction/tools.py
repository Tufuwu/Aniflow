"""File-based tool entry points for the CLI and a future local MCP adapter."""

import json
from pathlib import Path

from .director import model_advisor, plan
from .render import render
from .timeline import asset_paths, load_project, save_project, validate


def plan_project(storyboard: str, project: str, width: int = 1920, height: int = 1080,
                 fps: int = 30, vision_model: str | None = None, replace: bool = False) -> dict:
    """Create or explicitly update an edit project from a storyboard JSON array."""
    source = Path(storyboard).resolve()
    shots = json.loads(source.read_text(encoding='utf-8-sig'))
    advisor = model_advisor(vision_model) if vision_model else None
    timeline = plan(shots, base=source.parent, width=width, height=height, fps=fps, advisor=advisor)
    path = save_project(timeline, project, replace=replace)
    return {'project': str(path), 'frames': validate(timeline), 'decisions': timeline['decisions']}


def validate_project(project: str) -> dict:
    """Check the Timeline contract and whether its referenced media files exist."""
    source = Path(project).resolve()
    timeline = load_project(source)
    asset_paths(timeline, source.parent)
    return {'project': str(source), 'frames': validate(timeline), 'canvas': timeline['canvas']}


def render_project(project: str, output: str, preview: bool = False) -> dict:
    """Render a new MP4 and return per-clip diagnostic data."""
    source = Path(project).resolve()
    return render(load_project(source), output, base=source.parent, preview=preview)
