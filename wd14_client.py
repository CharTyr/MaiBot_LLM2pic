"""WD14 tagger reverse-tag client.

Calls a Modal-hosted WD14 tagger endpoint to extract Danbooru tags
from an image (base64). Used to support "引用图片 → 反推 tag → 融合用户文字 → 出图" workflow.
"""

from typing import Any, Optional
import asyncio
import base64
import json
import urllib.request
import urllib.error

from src.common.logger import get_logger

logger = get_logger("MaiBot_LLM2pic")

DEFAULT_ENDPOINT = "https://seckchiho--wd14-tagger-web-tag.modal.run"


class WD14Result:
    """Parsed WD14 reverse-tag result."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.prompt: str = str(raw.get("prompt") or "").strip()
        self.general: dict[str, float] = {}
        general_raw = raw.get("general")
        if isinstance(general_raw, dict):
            self.general = {k: float(v) for k, v in general_raw.items() if isinstance(v, (int, float))}
        elif isinstance(general_raw, list):
            for item in general_raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    self.general[str(item[0])] = float(item[1])
        self.character: dict[str, float] = {}
        char_raw = raw.get("character")
        if isinstance(char_raw, dict):
            self.character = {k: float(v) for k, v in char_raw.items() if isinstance(v, (int, float))}
        self.rating: dict[str, float] = {}
        rating_raw = raw.get("rating")
        if isinstance(rating_raw, dict):
            self.rating = {k: float(v) for k, v in rating_raw.items() if isinstance(v, (int, float))}

    @property
    def success(self) -> bool:
        return bool(self.prompt)

    def filtered_tags(self, threshold: float = 0.35, exclude_categories: tuple[str, ...] = ("rating",)) -> str:
        """Return a comma-separated tag string filtered by confidence threshold."""
        parts: list[str] = []
        if self.prompt:
            for tag in self.prompt.split(","):
                tag = tag.strip()
                if tag:
                    parts.append(tag)
        if not parts and self.general:
            parts = [
                tag for tag, conf in sorted(self.general.items(), key=lambda x: -x[1])
                if conf >= threshold
            ]
        return ", ".join(parts)


def _call_wd14_endpoint(image_base64: str, endpoint: str, threshold: float, timeout: float) -> dict[str, Any]:
    """Synchronous HTTP POST to the WD14 tagger endpoint."""
    payload = json.dumps({
        "image_base64": image_base64,
        "general_threshold": threshold,
    }).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


async def reverse_tag_image(
    image_base64: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    threshold: float = 0.35,
    timeout: float = 60.0,
) -> Optional[WD14Result]:
    """Reverse-tag an image (base64) via the WD14 tagger endpoint.

    Returns WD14Result on success, None on failure.
    """
    if not image_base64:
        return None
    try:
        raw = await asyncio.to_thread(_call_wd14_endpoint, image_base64, endpoint, threshold, timeout)
        result = WD14Result(raw)
        if result.success:
            logger.info("[WD14] reverse-tag OK: %d general tags, prompt len=%d", len(result.general), len(result.prompt))
            return result
        logger.warning("[WD14] reverse-tag returned empty prompt: %s", str(raw)[:200])
        return None
    except urllib.error.URLError as exc:
        logger.error("[WD14] reverse-tag request failed: %s", exc)
        return None
    except Exception as exc:
        logger.error("[WD14] reverse-tag unexpected error: %s", exc, exc_info=True)
        return None