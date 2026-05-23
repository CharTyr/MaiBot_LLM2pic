"""
图片工具函数：下载、编码、压缩、裁切、格式检测等
"""

import base64
import urllib.request
import urllib.parse
from typing import Any, Tuple

from src.common.logger import get_logger

logger = get_logger("MaiBot_LLM2pic")


def download_image_to_base64(url: str, timeout: int = 60) -> Tuple[bool, str]:
    """
    下载图片并转换为 base64，容忍 IncompleteRead

    Args:
        url: 图片 URL
        timeout: 超时时间（秒）

    Returns:
        Tuple[bool, str]: (是否成功, base64数据或错误信息)
    """
    from http.client import IncompleteRead

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                try:
                    image_bytes = response.read()
                except IncompleteRead as e:
                    image_bytes = e.partial

                if not image_bytes:
                    return False, "下载的图片数据为空"
                base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
                return True, base64_encoded
            else:
                return False, f"下载失败 (状态: {response.status})"
    except Exception as e:
        return False, str(e)


def get_image_mime_type(base64_data: str) -> str:
    """
    根据 base64 数据判断图片 MIME 类型

    Args:
        base64_data: base64 编码的图片数据

    Returns:
        str: MIME 类型
    """
    if base64_data.startswith("iVBORw"):
        return "image/png"
    elif base64_data.startswith("/9j/"):
        return "image/jpeg"
    elif base64_data.startswith("UklGR"):
        return "image/webp"
    elif base64_data.startswith("R0lGOD"):
        return "image/gif"
    else:
        return "image/png"  # 默认


def _looks_like_image_bytes(data: bytes) -> bool:
    if not data:
        return False
    return (
        data.startswith(b"\x89PNG\r\n\x1a\n")
        or data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"RIFF") and b"WEBP" in data[:16]
        or data.startswith(b"GIF87a")
        or data.startswith(b"GIF89a")
    )


def _looks_like_image_url(url: str) -> bool:
    if not url:
        return False
    lower = url.lower()
    if any(lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]):
        return True
    if "gradio_api/file=" in lower or "/image/" in lower or "image_url" in lower:
        return True
    if lower.endswith(".css") or lower.endswith(".js") or "bootstrap" in lower:
        return False
    return False


def _probe_url_is_image(url: str, timeout: int = 20) -> bool:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "image/*,*/*;q=0.8"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if content_type.startswith("image/"):
                return True
            head = resp.read(32)
            return _looks_like_image_bytes(head)
    except Exception:
        return False


def _normalize_url_for_request(url: str) -> str:
    """
    将可能包含空格/中文/未转义字符的 URL 规范化为可请求格式。
    重点处理 query 参数，避免 InvalidURL(control characters)。
    """
    parts = urllib.parse.urlsplit(url)
    if not parts.query:
        return url

    normalized_items = []
    for item in parts.query.split("&"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            key, value = item, ""
        # 先反解再重编码，避免重复编码和非法字符
        key_decoded = urllib.parse.unquote(key)
        value_decoded = urllib.parse.unquote(value)
        key_encoded = urllib.parse.quote(key_decoded, safe="._-")
        value_encoded = urllib.parse.quote(value_decoded, safe="._-")
        normalized_items.append(f"{key_encoded}={value_encoded}")

    normalized_query = "&".join(normalized_items)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, normalized_query, parts.fragment)
    )


def _compress_image_if_needed(image_base64: str) -> str:
    """将 PNG 转为 JPEG 以减小体积，避免 QQ 发送大图失败。"""
    # 只处理 PNG（iVBORw 开头），JPEG 已经够小
    if not image_base64.startswith("iVBORw"):
        return image_base64
    try:
        from io import BytesIO
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        image_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_bytes))
        img.load()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)
        compressed = base64.b64encode(buf.getvalue()).decode("utf-8")
        logger.info(f"[LLM2pic] PNG→JPEG: {len(image_base64)//1024}KB -> {len(compressed)//1024}KB")
        return compressed
    except ImportError:
        logger.warning("[LLM2pic] Pillow 未安装，无法转换图片格式")
        return image_base64
    except Exception as exc:
        logger.warning(f"[LLM2pic] PNG→JPEG 转换失败: {exc}")
        return image_base64


def _resize_image_for_edit(image_base64: str, max_pixels: int = 4_000_000) -> str:
    """缩放图片使总像素不超过限制，避免 API 拒绝过大图片。

    Args:
        image_base64: 原始 base64 图片数据
        max_pixels: 最大像素数（默认 4MP，适合大多数图片编辑 API）

    Returns:
        str: 处理后的 base64 图片数据（JPEG 格式）
    """
    try:
        from io import BytesIO
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        image_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_bytes))
        img.load()

        w, h = img.size
        total_pixels = w * h

        if total_pixels <= max_pixels:
            # 不需要缩放，但统一转为 JPEG 减小体积
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        # 等比缩放
        scale = (max_pixels / total_pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        result = base64.b64encode(buf.getvalue()).decode("utf-8")
        logger.info(f"[LLM2pic] 图片缩放: {w}x{h} -> {new_w}x{new_h} ({len(image_base64)//1024}KB -> {len(result)//1024}KB)")
        return result
    except ImportError:
        logger.warning("[LLM2pic] Pillow 未安装，无法缩放图片")
        return image_base64
    except Exception as exc:
        logger.warning(f"[LLM2pic] 图片缩放失败: {exc}")
        return image_base64


def _peel_envelope(payload: Any) -> Any:
    """剥离 SDK/Runner 常见的 result/data 包装层。"""
    current = payload
    visited: set[int] = set()
    while isinstance(current, dict):
        current_id = id(current)
        if current_id in visited:
            break
        visited.add(current_id)

        for wrapper_key in ("result", "data"):
            nested = current.get(wrapper_key)
            if isinstance(nested, dict):
                current = nested
                break
        else:
            return current
    return current


def _normalize_bool(value: Any) -> bool:
    """兼容表单/动作参数中字符串布尔值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)
