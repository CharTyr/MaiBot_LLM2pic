"""WD14 tagger reverse-tag client.

Calls a Modal-hosted WD14 tagger endpoint to extract Danbooru tags
from an image (base64). Used to support "引用图片 → 反推 tag → 融合用户文字 → 出图" workflow.
"""

from typing import Any, Optional
import asyncio
import base64
import hashlib
import json
import time
import urllib.request
import urllib.error
from collections import OrderedDict

from src.common.logger import get_logger

logger = get_logger("MaiBot_LLM2pic")

DEFAULT_ENDPOINT = "https://seckchiho--wd14-tagger-web-tag.modal.run"

# ---- LRU cache for WD14 results ----
_WD14_CACHE: "OrderedDict[str, tuple[Optional['WD14Result'], float]]" = OrderedDict()
_WD14_CACHE_MAX = 20
_WD14_CACHE_TTL = 600.0  # 10 minutes


def _image_hash(image_base64: str) -> str:
    try:
        raw = base64.b64decode(image_base64, validate=False)
        return hashlib.md5(raw).hexdigest()
    except Exception:
        return hashlib.md5(image_base64.encode()).hexdigest()


def _wd14_cache_get(h: str) -> Optional["WD14Result"]:
    entry = _WD14_CACHE.get(h)
    if entry is None:
        return None
    result, ts = entry
    if time.time() - ts > _WD14_CACHE_TTL:
        _WD14_CACHE.pop(h, None)
        return None
    _WD14_CACHE.move_to_end(h)
    return result


def _wd14_cache_set(h: str, result: Optional["WD14Result"]) -> None:
    _WD14_CACHE[h] = (result, time.time())
    _WD14_CACHE.move_to_end(h)
    while len(_WD14_CACHE) > _WD14_CACHE_MAX:
        _WD14_CACHE.popitem(last=False)


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

    def format_for_llm(self) -> str:
        """Return a structured, LLM-friendly text summary of the WD14 reverse-tag result.

        Groups general tags by confidence bucket, lists known characters and rating.
        Omit sections that have no data. Return empty string when general is empty.
        """
        if not self.general:
            return ""

        sorted_general = sorted(self.general.items(), key=lambda x: -x[1])

        high_tags = [(tag, conf) for tag, conf in sorted_general if conf >= 0.6]
        mid_tags = [(tag, conf) for tag, conf in sorted_general if 0.35 <= conf < 0.6]
        low_tags = [(tag, conf) for tag, conf in sorted_general if conf < 0.35]

        def _fmt(items: list[tuple[str, float]]) -> str:
            return ", ".join(f"{tag} ({conf:.2f})" for tag, conf in items)

        lines: list[str] = ["## 参考图 WD14 反推结果"]

        if high_tags:
            lines.append("")
            lines.append("### 高置信度 tag（confidence ≥ 0.6）")
            lines.append(_fmt(high_tags))

        if mid_tags:
            lines.append("")
            lines.append("### 中置信度 tag（0.35 ≤ confidence < 0.6）")
            lines.append(_fmt(mid_tags))

        if self.character:
            char_items = sorted(self.character.items(), key=lambda x: -x[1])
            lines.append("")
            lines.append("### 已知角色（如保留此角色，不要补充外貌 tag）")
            lines.append(_fmt(char_items))

        if self.rating:
            top_rating = max(self.rating.items(), key=lambda x: x[1])
            lines.append("")
            lines.append("### 安全等级")
            lines.append(f"{top_rating[0]} ({top_rating[1]:.2f})")

        if low_tags:
            lines.append("")
            lines.append("### 低置信度 tag（confidence < 0.35，参考用，不建议直接使用）")
            lines.append(_fmt(low_tags))

        return "\n".join(lines).strip()


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
    max_retries: int = 2,
) -> Optional[WD14Result]:
    """Reverse-tag an image (base64) via the WD14 tagger endpoint.

    Returns WD14Result on success, None on failure.
    Caches results by image hash (LRU, TTL 10min). Retries on transient errors.
    """
    if not image_base64:
        return None

    h = _image_hash(image_base64)
    cached = _wd14_cache_get(h)
    if cached is not None:
        logger.info("[WD14] cache hit (hash=%s...)", h[:12])
        return cached
    # None cached means "checked but failed" — don't re-hit endpoint repeatedly
    # However we can't distinguish "never checked" from "checked and failed",
    # so we only cache successes. Failures always retry.

    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = await asyncio.to_thread(_call_wd14_endpoint, image_base64, endpoint, threshold, timeout)
            result = WD14Result(raw)
            if result.success:
                logger.info(
                    "[WD14] reverse-tag OK (attempt %s/%s): %d general tags, prompt len=%d",
                    attempt, max_retries, len(result.general), len(result.prompt),
                )
                _wd14_cache_set(h, result)
                return result
            logger.warning("[WD14] reverse-tag returned empty prompt (attempt %s/%s): %s", attempt, max_retries, str(raw)[:200])
            last_error = "empty prompt"
        except (urllib.error.URLError, asyncio.TimeoutError, OSError) as exc:
            last_error = str(exc)
            logger.warning("[WD14] reverse-tag request failed (attempt %s/%s): %s", attempt, max_retries, exc)
        except Exception as exc:
            last_error = str(exc)
            logger.error("[WD14] reverse-tag unexpected error (attempt %s/%s): %s", attempt, max_retries, exc, exc_info=True)

        if attempt < max_retries:
            backoff = 2.0 * attempt
            logger.info("[WD14] retrying in %.1fs ...", backoff)
            await asyncio.sleep(backoff)

    logger.error("[WD14] reverse-tag exhausted %s retries: %s", max_retries, last_error)
    return None