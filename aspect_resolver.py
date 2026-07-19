"""Content-aware aspect ratio selection for LLM2Pic."""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Optional

from .core.utils.prompt_output_parser import normalize_aspect


_ASPECTS = ("portrait", "landscape", "square")
_MAX_REFERENCE_ENCODED_LENGTH = 32 * 1024 * 1024
_MAX_REFERENCE_BYTES = 24 * 1024 * 1024
_MAX_REFERENCE_PIXELS = 25_000_000

_EXPLICIT_TOKENS = {
    "square": (
        "方图", "正方形", "1:1", "square", "头像", "表情包", "贴纸",
        "icon", "sticker", "大头照", "脸部特写", "头像图",
    ),
    "landscape": (
        "横图", "横屏", "横版", "宽屏", "宽幅", "横向", "横向构图", "16:9",
        "landscape", "horizontal", "wide shot", "panorama", "电脑壁纸", "桌面壁纸",
    ),
    "portrait": (
        "竖图", "竖屏", "竖版", "窄屏", "竖向", "竖向构图", "portrait",
        "立绘", "全身立绘", "手机壁纸",
    ),
}

_LANDSCAPE_TOKENS = (
    ("landscape", 4), ("horizontal", 4), ("wide shot", 4), ("panorama", 4),
    ("scenery", 3), ("cityscape", 3), ("skyline", 3), ("building", 2),
    ("architecture", 2), ("vehicle", 3), ("car", 3), ("road", 2),
    ("train", 3), ("bicycle", 2), ("motorcycle", 2), ("mountain", 2),
    ("beach", 2), ("ocean", 2), ("field", 2), ("群像", 4), ("多人", 4),
    ("双人", 4), ("风景", 3), ("建筑", 2), ("车辆", 3), ("横图", 4),
    ("横屏", 4), ("宽幅", 4),
)

_PORTRAIT_TOKENS = (
    ("full body", 2), ("character sheet", 3), ("standing", 1),
    ("portrait", 2), ("selfie", 2), ("立绘", 3), ("全身", 1),
    ("竖图", 4), ("竖屏", 4),
)

_SQUARE_TOKENS = (
    ("close-up", 4), ("close up", 4), ("face focus", 4), ("headshot", 4),
    ("avatar", 4), ("icon", 3), ("sticker", 3), ("表情包", 4),
    ("头像", 4), ("近景", 3), ("脸部特写", 4), ("方图", 4),
)


@dataclass(frozen=True)
class AspectDecision:
    aspect: str
    source: str
    scores: dict[str, int]
    llm_aspect: Optional[str] = None
    reference_aspect: Optional[str] = None
    explicit_token: Optional[str] = None



def _is_negated(text: str, index: int) -> bool:
    prefix = text[max(0, index - 3):index]
    return "不要" in prefix or "别" in prefix or "不想要" in prefix


def _find_explicit_aspect(user_request: str) -> tuple[Optional[str], Optional[str]]:
    text = str(user_request or "").strip().lower()
    matches: list[tuple[int, str, str]] = []
    for aspect, tokens in _EXPLICIT_TOKENS.items():
        for token in tokens:
            for match in re.finditer(re.escape(token.lower()), text):
                start = match.start()
                if not _is_negated(text, start):
                    matches.append((start, aspect, token))
    if not matches:
        return None, None
    _, aspect, token = max(matches, key=lambda item: item[0])
    return aspect, token


def _score_tokens(text: str, tokens: tuple[tuple[str, int], ...]) -> int:
    return sum(weight for token, weight in tokens if token in text)


def _has_multiple_subjects(text: str) -> bool:
    """Detect multi-person composition, not multi-angle character sheets.

    Important: do NOT match bare substring ``multiple`` — ``multiple views``
    is a common solo character-sheet tag and was wrongly boosting landscape.
    """
    if re.search(r"(?<!\d)[2-9]\s*(?:girls?|boys?)\b", text):
        return True
    multi_person = (
        "multiple girls",
        "multiple boys",
        "multiple people",
        "group of",
        "group photo",
        "2girls",
        "2boys",
        "3girls",
        "3boys",
        "多人",
        "群像",
        "双人",
    )
    if any(token in text for token in multi_person):
        return True
    if re.search(r"\bgroup\b", text) and any(
        x in text for x in ("people", "girls", "boys", "人物", "合影")
    ):
        return True
    return False

def _strip_data_uri(image_base64: str) -> str:
    encoded = str(image_base64 or "").strip()
    if encoded.lower().startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    return "".join(encoded.split())


def decode_reference_image(image_base64: str):
    """安全解码参考图，统一处理 data URI、大小和透明通道。"""
    encoded = _strip_data_uri(image_base64)
    if not encoded or len(encoded) > _MAX_REFERENCE_ENCODED_LENGTH:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > _MAX_REFERENCE_BYTES:
            return None
        from PIL import Image
        image = Image.open(io.BytesIO(raw))
        if image.width * image.height > _MAX_REFERENCE_PIXELS:
            return None
        image.load()
        if image.mode not in {"1", "L", "P", "RGB", "RGBA", "LA"}:
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            image = image.convert("RGBA" if has_alpha else "RGB")
    except Exception:
        return None
    if min(image.width, image.height) < 256:
        return None
    return image


def detect_reference_aspect(image_base64: str) -> Optional[str]:
    """把参考图归入三个 NAI 安全尺寸档位；无法解码时返回 None。"""
    image = decode_reference_image(image_base64)
    if image is None:
        return None
    ratio = image.width / image.height
    if ratio >= 1.15:
        return "landscape"
    if ratio <= 0.87:
        return "portrait"
    return "square"


def resolve_aspect(
    *,
    user_request: str,
    generated_prompt: str,
    llm_aspect: Optional[str] = None,
    has_characters: bool = False,
    selfie_mode: bool = False,
    reference_image_base64: str = "",
) -> AspectDecision:
    """按明确意图、场景内容、参考图和 LLM 建议裁决最终画幅。"""
    llm = normalize_aspect(llm_aspect)
    explicit, explicit_token = _find_explicit_aspect(user_request)
    text = f"{user_request} {generated_prompt}".lower()

    scores = {
        "portrait": _score_tokens(text, _PORTRAIT_TOKENS),
        "landscape": _score_tokens(text, _LANDSCAPE_TOKENS),
        "square": _score_tokens(text, _SQUARE_TOKENS),
    }
    if has_characters or _has_multiple_subjects(text):
        scores["landscape"] += 4
    if selfie_mode:
        scores["portrait"] += 2

    reference = detect_reference_aspect(reference_image_base64)

    if explicit:
        return AspectDecision(explicit, "explicit", scores, llm, reference, explicit_token)
    selfie_requested = "自拍" in str(user_request or "") or "selfie" in str(user_request or "").lower()
    if selfie_mode or selfie_requested:
        return AspectDecision("portrait", "selfie", scores, llm, reference)

    ordered = sorted(_ASPECTS, key=lambda item: (scores[item], item), reverse=True)
    best, second = ordered[0], scores[ordered[1]]
    if scores[best] >= 4 and scores[best] >= second + 1:
        return AspectDecision(best, "content", scores, llm, reference)

    # 没有强烈构图信号时，参考图只作为布局保真度的 tie-breaker。
    if reference and max(scores.values()) <= 2:
        return AspectDecision(reference, "reference", scores, llm, reference)

    if llm:
        return AspectDecision(llm, "llm", scores, llm, reference)
    if reference:
        return AspectDecision(reference, "reference", scores, llm, reference)
    return AspectDecision("portrait", "default", scores, llm, reference)
