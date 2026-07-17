"""
NewAPI NAI 图片生成客户端。

支持 NAI V4/V4.5 全部 API 能力：
- txt2img（文生图）
- i2i（图生图）
- inpaint（局部重绘）
- character_references（角色参考 / Precise Reference）
- controlnet（Vibe Transfer / 风格迁移）
- characters[]（多角色 + 5×5 网格坐标）

API 文档：见 references/api-20260527-full.md
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from src.common.logger import get_logger

from .base import (
    GenerationContext,
    GenerationResult,
    ImageClient,
    calc_max_tokens,
    validate_ref_mutex,
)

logger = get_logger("MaiBot_LLM2pic")

# ── 常量 ──

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_SEEDS_RE = re.compile(r"<!--\s*seeds:\s*(\[[^\]]*])\s*-->")
_VIBE_CACHE_RE = re.compile(r"<!--\s*vibe_cache_ids:\s*(\[.*?\])\s*-->")
_IMG_RE = re.compile(r"!\[[^\]]*\]\((data:image/[^;)]+;base64,[^)]+)\)")
_POSITION_GRID_RE = re.compile(r"^[A-E][1-5]$")
_MULTI_CHAR_MODEL_KEYWORDS = ("nai-diffusion-4",)
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef"
    r"\u2e80-\u2eff\uac00-\ud7af\u3040-\u30ff\uf900-\ufaff]+"
)

# ── CJK 清洗 ──

def _strip_cjk(segment: str) -> str:
    if not segment or not segment.strip():
        return segment
    stripped = _CJK_RE.sub("", segment)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip(" ,;")
    return stripped


def sanitize_prompt(prompt: str) -> str:
    """清洗 prompt 中的 CJK 字符（NewAPI NAI 禁止中文）。"""
    if not prompt or not _CJK_RE.search(prompt):
        return prompt
    parts = [p.strip() for p in prompt.split(",")]
    cleaned = [_strip_cjk(p) for p in parts]
    cleaned = [p for p in cleaned if p]
    result = ", ".join(cleaned)
    result = re.sub(r",\s*,+", ", ", result).strip(" ,")
    return result


# ── 尺寸解析 ──

_SIZE_MAP = {
    "portrait": [832, 1216],
    "landscape": [1216, 832],
    "square": [1024, 1024],
}


def resolve_size(size: Any) -> list[int]:
    """把 size 参数解析为 [w, h] 整数数组。"""
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return [int(size[0]), int(size[1])]
    normalized = str(size or "").strip().lower()
    if not normalized:
        return [832, 1216]
    if normalized in _SIZE_MAP:
        return _SIZE_MAP[normalized]
    if "x" in normalized:
        parts = normalized.split("x", 1)
        try:
            return [int(parts[0]), int(parts[1])]
        except (ValueError, IndexError):
            pass
    return [832, 1216]


# ── 多角色规范化 ──

def normalize_characters(characters: Optional[list[dict]]) -> list[dict[str, str]]:
    """规范化 characters[] 列表。

    - 每项必须有 prompt（非空）
    - negative_prompt 可选
    - position 必须匹配 [A-E][1-5]，否则丢弃
    - 少于 2 个有效角色时返回空列表
    - 任一角色缺 position 时全部清除 position（API 要求一致）
    """
    if not characters:
        return []

    cleaned: list[dict[str, str]] = []
    for item in characters:
        if not isinstance(item, dict):
            continue
        char_prompt = str(item.get("prompt") or "").strip()
        if not char_prompt:
            continue
        raw_position = str(item.get("position") or "").strip().upper()
        entry = {
            "prompt": sanitize_prompt(char_prompt),
            "negative_prompt": sanitize_prompt(str(item.get("negative_prompt") or "").strip()),
            "position": raw_position if _POSITION_GRID_RE.match(raw_position) else "",
        }
        cleaned.append(entry)

    if len(cleaned) < 2:
        return []

    # 任一缺 position → 全部清除（API 要求 use_coords 时全员都有）
    if any(not item["position"] for item in cleaned):
        for item in cleaned:
            item["position"] = ""

    result: list[dict[str, str]] = []
    for item in cleaned:
        entry: dict[str, str] = {"prompt": item["prompt"]}
        if item["negative_prompt"]:
            entry["negative_prompt"] = item["negative_prompt"]
        if item["position"]:
            entry["position"] = item["position"]
        result.append(entry)
    return result


def supports_multi_char(model: str) -> bool:
    """检查模型是否支持原生多角色 API。"""
    lowered = str(model or "").lower()
    return any(keyword in lowered for keyword in _MULTI_CHAR_MODEL_KEYWORDS)


# ── Token 归一化 ──

def normalize_token(api_key: str) -> str:
    token = str(api_key or "").strip()
    if token.lower().startswith("bearer "):
        return token.split(" ", 1)[1].strip()
    return token


# ── 代理模式 ──

def _make_opener(proxy_mode: str) -> Optional[urllib.request.OpenerDirector]:
    """根据 proxy_mode 创建 opener。None = 默认 urlopen。"""
    if proxy_mode == "direct":
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return None  # auto / inherit = 默认行为


def _is_retryable_url_error(exc: urllib.error.URLError) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_STATUS_CODES
    reason = str(getattr(exc, "reason", exc)).lower()
    return any(t in reason for t in ("timed out", "timeout", "connection reset", "temporarily unavailable"))


# ── 响应解析 ──

def _extract_message_content(choice: Any) -> str:
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def extract_image_b64(content: str) -> str:
    """从 message.content 中提取 base64 图片数据（不含 data: 前缀）。"""
    # 优先匹配 markdown 图片语法
    match = _IMG_RE.search(content)
    if match:
        data_uri = match.group(1)
        if "," in data_uri:
            return data_uri.split(",", 1)[1]
        return data_uri
    # 兜底：匹配 data:image/...;base64,...
    data_uri_match = re.search(r"data:image/[^;)]+;base64,([A-Za-z0-9+/=\s]+)", content)
    if data_uri_match:
        return "".join(data_uri_match.group(1).split())
    return ""


def extract_seeds(content: str) -> list:
    match = _SEEDS_RE.search(content or "")
    if not match:
        return []
    try:
        seeds = json.loads(match.group(1))
    except (TypeError, ValueError):
        return []
    return seeds if isinstance(seeds, list) else []


def extract_vibe_cache_ids(content: str) -> list[dict]:
    match = _VIBE_CACHE_RE.search(content or "")
    if not match:
        return []
    try:
        ids = json.loads(match.group(1))
    except (TypeError, ValueError):
        return []
    return ids if isinstance(ids, list) else []


def extract_error(response_data: Any) -> str:
    if not isinstance(response_data, dict):
        return ""
    error = response_data.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        code = str(error.get("code") or "").strip()
        return f"{message} (code={code})" if message and code else message or code
    if isinstance(error, str) and error.strip():
        return error.strip()
    for key in ("message", "detail"):
        value = response_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ── 客户端 ──

class NewApiNaiClient(ImageClient):
    """NewAPI NAI 绘图客户端。

    通过 OpenAI chat/completions 端点调用 NAI 绘图。
    内层 payload 作为 JSON 字符串放在 messages[0].content。
    """

    def __init__(self, base_url: str, api_key: str, log_prefix: str = "[NAI]"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.log_prefix = log_prefix

    async def generate(self, ctx: GenerationContext) -> GenerationResult:
        """执行 NAI 绘图请求。"""
        # 互斥校验
        try:
            validate_ref_mutex(ctx)
        except ValueError as exc:
            return GenerationResult(success=False, error=str(exc))

        # 构造内层 payload
        inner = self._build_inner(ctx)
        if ctx.ref_mode == "i2i":
            i2i_block = inner.get("i2i") if isinstance(inner, dict) else None
            logger.info(
                f"{self.log_prefix} i2i payload: present={bool(i2i_block)}, "
                f"keys={list(i2i_block.keys()) if isinstance(i2i_block, dict) else None}, "
                f"image_len={len(i2i_block.get('image', '')) if isinstance(i2i_block, dict) else 0}, "
                f"strength={(i2i_block or {}).get('strength') if isinstance(i2i_block, dict) else None}"
            )

        # 构造外层 OpenAI 格式
        max_tokens = calc_max_tokens(ctx)
        payload = {
            "model": ctx.model,
            "messages": [{"role": "user", "content": json.dumps(inner, ensure_ascii=False)}],
            "stream": False,
            "max_tokens": max_tokens,
        }

        # 发请求
        resp_data = self._post(payload, ctx)
        if resp_data is None:
            return GenerationResult(success=False, error="NewAPI 请求失败")

        if not isinstance(resp_data, dict):
            return GenerationResult(success=False, error="NewAPI 返回非 JSON 响应")

        # 检查 HTTP 层面错误
        if "error" in resp_data:
            err_msg = extract_error(resp_data)
            return GenerationResult(success=False, error=err_msg or "NewAPI 返回错误")

        # 解析 choices
        choices = resp_data.get("choices")
        if not choices or not isinstance(choices, list):
            err_msg = extract_error(resp_data) or "NewAPI 未返回 choices"
            return GenerationResult(success=False, error=err_msg)

        content = _extract_message_content(choices[0])
        if not content:
            err_msg = extract_error(resp_data) or "NewAPI 未返回 message.content"
            return GenerationResult(success=False, error=err_msg)

        # 提取图片
        image_b64 = extract_image_b64(content)
        if not image_b64:
            # 可能是错误信封
            try:
                err_msg = extract_error(json.loads(content.strip()))
            except (TypeError, ValueError):
                err_msg = ""
            return GenerationResult(
                success=False,
                error=err_msg or f"NewAPI 响应中未找到图片数据: {content[:200]}",
                raw_content=content,
            )

        # 提取 seed + vibe_cache_ids
        seeds = extract_seeds(content)
        seed = seeds[0] if seeds else -1
        vibe_ids = extract_vibe_cache_ids(content)

        usage = resp_data.get("usage")
        logger.info(
            f"{self.log_prefix} 出图成功: seed={seed}, "
            f"usage={usage}, ref_mode={ctx.ref_mode}"
        )

        return GenerationResult(
            success=True,
            image_base64=image_b64,
            seed=seed,
            vibe_cache_ids=vibe_ids,
            raw_content=content,
        )

    # ── payload 构造 ──

    def _build_inner(self, ctx: GenerationContext) -> dict:
        """构造内层 JSON payload（messages[0].content 解析后的对象）。"""
        inner: dict[str, Any] = {
            "prompt": sanitize_prompt(ctx.prompt),
            "size": resolve_size(ctx.size),
            "steps": min(max(int(ctx.steps), 1), 28),
            "scale": ctx.scale,
            "sampler": str(ctx.sampler or "k_euler_ancestral"),
            "n_samples": 1,
            "image_format": str(ctx.image_format or "png"),
        }

        # negative_prompt
        neg = str(ctx.negative_prompt or "").strip()
        if neg:
            sanitized_neg = sanitize_prompt(neg)
            if sanitized_neg:
                inner["negative_prompt"] = sanitized_neg

        # seed
        if ctx.seed >= 0:
            inner["seed"] = int(ctx.seed)

        # 高级参数
        if ctx.variety_boost:
            inner["variety_boost"] = True
        if ctx.cfg_rescale is not None:
            inner["cfg_rescale"] = ctx.cfg_rescale
        if ctx.noise_schedule:
            inner["noise_schedule"] = ctx.noise_schedule
        if ctx.quality_toggle:
            inner["qualityToggle"] = True
        if ctx.auto_smea:
            inner["autoSmea"] = True

        # 多角色
        normalized_chars = normalize_characters(ctx.characters)
        if normalized_chars and supports_multi_char(ctx.model):
            inner["characters"] = normalized_chars
            inner["use_coords"] = all("position" in item for item in normalized_chars)
            inner["use_order"] = True
        elif normalized_chars:
            logger.warning(
                f"{self.log_prefix} 模型 {ctx.model!r} 不支持 characters[]，已降级为单 prompt"
            )

        # ── 参考图：互斥组 A（i2i vs inpaint）──
        if ctx.ref_mode == "i2i" and ctx.i2i_image:
            inner["i2i"] = {
                "image": ctx.i2i_image,
                "strength": float(max(0.01, min(ctx.i2i_strength, 0.99))),
                "noise": float(max(0.0, min(ctx.i2i_noise, 0.99))),
            }
        elif ctx.ref_mode == "inpaint" and ctx.inpaint_image:
            inpaint: dict[str, Any] = {
                "image": ctx.inpaint_image,
                "strength": ctx.inpaint_strength,
            }
            if ctx.inpaint_mask:
                inpaint["mask"] = ctx.inpaint_mask
            inner["inpaint"] = inpaint

        # ── 参考图：互斥组 B（controlnet vs character_references）──
        if ctx.ref_mode == "vibe" and ctx.vibe_images:
            vibe_imgs = ctx.vibe_images
            if len(vibe_imgs) > 4:
                logger.warning(f"{self.log_prefix} controlnet max 4 images, truncating from {len(vibe_imgs)}")
                vibe_imgs = vibe_imgs[:4]
            inner["controlnet"] = {
                "strength": ctx.vibe_global_strength,
                "images": vibe_imgs,
            }
        elif ctx.ref_mode == "char_ref" and ctx.char_ref_image:
            inner["character_references"] = [{
                "image": ctx.char_ref_image,
                "type": ctx.char_ref_type,
                "fidelity": ctx.char_ref_fidelity,
                "strength": ctx.char_ref_strength,
            }]

        # 扩展参数（legacy 兼容）
        if ctx.extra_params and isinstance(ctx.extra_params, dict):
            for key, value in ctx.extra_params.items():
                if value not in (None, ""):
                    inner[str(key)] = value

        return inner

    # ── HTTP 请求 ──

    def _post(self, payload: dict, ctx: GenerationContext) -> Optional[dict]:
        """发送 POST 请求，带重试。返回解析后的 JSON dict 或 None。"""
        endpoint = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {normalize_token(self.api_key)}",
            "User-Agent": "Mozilla/5.0",
        }

        prompt_preview = ctx.prompt[:200] if ctx.prompt else "(empty)"
        has_i2i = bool(ctx.ref_mode == "i2i" and ctx.i2i_image)
        i2i_len = len(ctx.i2i_image) if ctx.i2i_image else 0
        logger.info(
            f"{self.log_prefix} 发起 NAI 请求: model={ctx.model}, "
            f"ref_mode={ctx.ref_mode}, has_i2i={has_i2i}, i2i_image_len={i2i_len}, "
            f"i2i_strength={getattr(ctx, 'i2i_strength', None)}, prompt={prompt_preview}..."
        )

        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        timeout = ctx.timeout
        retry_attempts = max(1, min(ctx.retry_attempts, 5))
        proxy_mode = ctx.proxy_mode

        for attempt in range(1, retry_attempts + 1):
            try:
                response = self._urlopen(req, timeout=timeout, proxy_mode=proxy_mode)
                with response as resp:
                    body = resp.read().decode("utf-8")
                    status = resp.status if hasattr(resp, "status") else resp.getcode()
                    if not 200 <= status < 300:
                        return {"error": {"message": f"HTTP {status}: {body[:300]}"}}

                return json.loads(body)

            except urllib.error.HTTPError as exc:
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8")[:300]
                except Exception:
                    pass

                if exc.code in _RETRYABLE_STATUS_CODES and attempt < retry_attempts:
                    sleep_seconds = 6.0 if exc.code == 429 else 1.5 * attempt
                    logger.warning(
                        f"{self.log_prefix} HTTP {exc.code}，{sleep_seconds:.1f}s 后重试"
                    )
                    time.sleep(sleep_seconds)
                    continue

                logger.error(f"{self.log_prefix} HTTP 错误: {exc.code} - {error_body}")
                return {"error": {"message": f"HTTP {exc.code}: {error_body}"}}

            except urllib.error.URLError as exc:
                if _is_retryable_url_error(exc) and attempt < retry_attempts:
                    sleep_seconds = 1.5 * attempt
                    logger.warning(
                        f"{self.log_prefix} 网络错误，{sleep_seconds:.1f}s 后重试: {exc}"
                    )
                    time.sleep(sleep_seconds)
                    continue
                logger.error(f"{self.log_prefix} 连接错误: {exc}")
                return {"error": {"message": f"连接错误: {getattr(exc, 'reason', exc)}"}}

            except json.JSONDecodeError as exc:
                logger.error(f"{self.log_prefix} JSON 解析失败: {exc}")
                return {"error": {"message": "NewAPI 返回了非 JSON 响应"}}

            except Exception as exc:
                logger.error(f"{self.log_prefix} 请求错误: {exc!r}", exc_info=True)
                return {"error": {"message": str(exc)}}

        return {"error": {"message": "NewAPI 请求失败（重试耗尽）"}}

    def _urlopen(self, req: urllib.request.Request, *, timeout: int, proxy_mode: str):
        """根据 proxy_mode 选择 urlopen 方式。"""
        if proxy_mode == "direct":
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return opener.open(req, timeout=timeout)

        if proxy_mode == "inherit":
            return urllib.request.urlopen(req, timeout=timeout)

        # auto: 先尝试默认（继承代理），失败后直连
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.URLError as exc:
            if isinstance(exc, urllib.error.HTTPError):
                raise
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            logger.warning(f"{self.log_prefix} 继承代理失败，尝试直连: {exc}")
            return opener.open(req, timeout=timeout)