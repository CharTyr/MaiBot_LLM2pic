"""
Draw pipeline 编排器。

将原来的四层嵌套（handle → _background → _background_inner → _run_generation_and_send）
拍平为单函数 run_draw_pipeline()。

支持参考图模式：i2i / char_ref / vibe，通过 ref_mode 参数显式触发。
"""

from __future__ import annotations

import base64
import re
import io
from dataclasses import dataclass, field
from typing import Any, Optional

from src.common.logger import get_logger

from .clients.base import GenerationContext
from .clients.factory import create_client_from_model_config
from .style_router import StyleRouter
from .generation_service import (
    generate_image,
    ImageGenerationRequest,
)
from .utils import _normalize_bool, _resize_image_for_wd14
from .vibe_cache import VibeCache
from .aspect_resolver import decode_reference_image, resolve_aspect

logger = get_logger("MaiBot_LLM2pic")

_REMOVE_ELEMENT_RE = re.compile(
    r"(不要|别要|去掉|删除|移除|别带|不要有|no\s+|without\s+|remove\s+)"
    r".{0,12}?"
    r"(对话框|对话框气泡|气泡|对话框框|speech\s*bubble|dialogue\s*box|text\s*bubble|comic\s*panel\s*text|字幕|文字框)",
    re.IGNORECASE,
)
_POSE_KEEP_RE = re.compile(r"(照这个姿势|保留姿势|按这个姿势|保持构图|照构图|same pose|keep pose)", re.IGNORECASE)
_DIALOG_NEGATIVE = (
    "speech bubble",
    "dialogue box",
    "text bubble",
    "comic panel text",
    "spoken text",
    "written text",
    "caption",
)

_SIZE_MAP = {
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "square": (1024, 1024),
}


@dataclass
class DrawPipelineContext:
    """Draw pipeline 上下文。"""

    source: str  # "direct_pic" | "draw_picture"
    user_request: str
    chat_messages: str = ""
    persona: str = ""
    selfie_mode: bool = False
    nsfw_allowed: bool = False
    manual_style: Optional[str] = None
    ref_mode: str = ""  # "" | "i2i" | "char_ref" | "vibe"
    custom_system_prompt: str = ""
    config: dict = field(default_factory=dict)
    stream_id: str = ""
    proxy: Any = None       # _ToolRuntimeProxy 或 _CommandRuntimeProxy
    plugin: Any = None      # LLM2PicPlugin 引用（用于 _ctx_extract_image_from_recent）
    session_message: Any = None
    ref_extract_error: str = ""
    attachment_source: str = ""


def _decode_reference_image(image_base64: str):
    """统一走安全的参考图解码器。"""
    return decode_reference_image(image_base64)


def _image_data_uri_for_reference(image_base64: str) -> str:
    """将 char-ref/vibe 参考图转成 data URI，但保留原始尺寸、比例和透明通道。"""
    img = _decode_reference_image(image_base64)
    if img is None:
        return ""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _resize_image_for_nai(image_base64: str, target_size: tuple[int, int]) -> str:
    """将图片按比例裁切后 resize 到 NAI 要求的精确尺寸，返回 data URI。

    NAI 的 i2i 要求参考图和目标 size 完全一致。旧逻辑直接拉伸，
    自动画幅切换时会把人物和场景压扁；现在先做居中裁切，再缩放。
    """
    img = _decode_reference_image(image_base64)
    if img is None:
        return ""

    w, h = target_size
    source_ratio = img.width / img.height
    target_ratio = w / h
    if abs(source_ratio - target_ratio) > 0.02:
        if source_ratio > target_ratio:
            crop_width = max(1, int(img.height * target_ratio))
            left = max(0, (img.width - crop_width) // 2)
            img = img.crop((left, 0, left + crop_width, img.height))
        else:
            crop_height = max(1, int(img.width / target_ratio))
            top = max(0, (img.height - crop_height) // 2)
            img = img.crop((0, top, img.width, top + crop_height))

    from PIL import Image
    img = img.resize((w, h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"



def _merge_negative_prompt(base_negative: str, extra_tags: list[str] | None) -> str:
    items: list[str] = []
    seen: set[str] = set()
    for raw in [*(str(base_negative or "").split(",")), *(extra_tags or [])]:
        tag = str(raw or "").strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(tag)
    return ", ".join(items)


def _resolve_i2i_params(
    *,
    user_request: str,
    prompt_result,
    ref_cfg: dict,
) -> tuple[float, float, list[str], str]:
    """合并配置默认值、LLM 建议与用户否定意图，返回 strength/noise/extra_negatives/source。"""
    base_strength = float(ref_cfg.get("i2i_strength", 0.7) or 0.7)
    base_noise = float(ref_cfg.get("i2i_noise", 0.0) or 0.0)
    llm_strength = getattr(prompt_result, "i2i_strength", None)
    llm_noise = getattr(prompt_result, "i2i_noise", None)
    llm_neg = list(getattr(prompt_result, "negative_tags", None) or [])

    # LLM 填了用 LLM，没填用配置。不用用户文本关键词抬/压 strength。
    _ = user_request
    if llm_strength is not None:
        strength = float(llm_strength)
        source_parts = ["llm"]
    else:
        strength = base_strength
        source_parts = ["config"]
    noise = float(llm_noise) if llm_noise is not None else base_noise
    extra_neg = list(llm_neg)

    strength = max(0.35, min(0.95, strength))
    noise = max(0.0, min(0.3, noise))
    return strength, noise, extra_neg, "+".join(source_parts)

async def _extract_attachment(ctx: DrawPipelineContext) -> Optional[str]:
    """从当前消息 / 当前消息引用 / 最近消息中提取附图 base64。

    优先级（i2i 正确性关键）：
    1. 当前消息自身 image 段（同条消息附图）
    2. 当前消息 reply_to / reply 段（用户「回复图片 + /pic i2i」）
    3. 最近 5 分钟消息里的图（仅无 ref_mode 且无显式引用时兜底）

    铁律：
    - 用户显式引用了消息时，禁止 silent recent_fallback。
    - i2i/char_ref/vibe 必须用到「当前图或引用图」；引用目标失效时直接失败，
      不要偷偷拿 bot 刚出的图冒充参考图。
    """
    try:
        strict_ref = bool(ctx.ref_mode)
        session_message = getattr(ctx.proxy, "message", None) if ctx.proxy is not None else None
        if session_message is None:
            session_message = ctx.session_message

        # 1. 当前消息自身 image 段（Command proxy 可能是 dict，也可能 object）
        if ctx.source == "direct_pic" and ctx.proxy is not None:
            img = await ctx.proxy._extract_input_image()
            if img:
                ctx.attachment_source = "current_message_image"
                logger.info("[Pipeline] 附图来源=current_message_image b64_len=%s", len(img))
                return img

        # 2. 当前命令消息的 reply_to / reply 段（显式引用）
        if ctx.plugin is not None and session_message is not None:
            ref_info = await ctx.plugin._ctx_resolve_session_reference(
                session_message, ctx.stream_id
            )
            img = ref_info.get("image")
            reply_to = ref_info.get("reply_to") or ""
            if img:
                ctx.attachment_source = "reply_to"
                logger.info(
                    "[Pipeline] 附图来源=%s reply_to=%s b64_len=%s",
                    ref_info.get("source") or "session_message_reply_or_self",
                    reply_to or "-",
                    len(img),
                )
                return img
            if reply_to:
                # 用户明确引用了某条消息，但取不到图：禁止 recent 兜底
                reason = ref_info.get("error") or "引用消息无法访问或不含图片"
                logger.error(
                    "[Pipeline] 显式引用取图失败: reply_to=%s reason=%s strict_ref=%s",
                    reply_to,
                    reason,
                    strict_ref,
                )
                if strict_ref:
                    ctx.ref_extract_error = (
                        f"引用消息已失效或取不到图片（reply_to={reply_to}）。"
                        f"请重新回复那张参考图后再 /pic {ctx.ref_mode}"
                    )
                return None

        # 3. 兜底：最近消息扫描
        # - 有 ref_mode：禁止（必须显式附图/引用，避免捞到 bot 刚出的图）
        # - 无 ref_mode：允许给 WD14 增强用
        if ctx.plugin is not None and not strict_ref:
            img = await ctx.plugin._ctx_extract_image_from_recent(ctx.stream_id)
            if img:
                ctx.attachment_source = "recent_fallback"
                logger.warning(
                    "[Pipeline] 附图来源=recent_fallback（无显式引用，可能不是用户想要的图） b64_len=%s",
                    len(img),
                )
                return img
        elif strict_ref:
            logger.warning(
                "[Pipeline] ref_mode=%s 未找到当前图/引用图，跳过 recent_fallback",
                ctx.ref_mode,
            )
    except Exception as exc:
        logger.warning("[Pipeline] 提取附图失败: %s", exc)
    return None


async def run_draw_pipeline(ctx: DrawPipelineContext) -> bool:
    """Draw pipeline 主入口。返回 True 表示成功。"""

    try:
        # ── 1. 附图检测 ──
        ctx.attachment_source = ""
        attachment_b64 = None
        if ctx.ref_mode:
            attachment_b64 = await _extract_attachment(ctx)
            if not attachment_b64:
                err = str(getattr(ctx, "ref_extract_error", "") or "").strip()
                logger.warning(
                    "[Pipeline] ref_mode=%s 但未提取到附图（current/reply 失败，已禁止 recent 兜底） err=%s",
                    ctx.ref_mode,
                    err or "-",
                )
                if err:
                    await _safe_send(ctx, err)
                else:
                    await _safe_send(
                        ctx,
                        f"指定了 {ctx.ref_mode} 但没检测到参考图。请「回复引用那张图」或同条附图后再发 /pic {ctx.ref_mode}",
                    )
                # 显式参考图模式失败：直接中止，禁止降级文生图 / recent 兜底
                return False

        # 无 ref_mode 时才扫附图（用于 WD14 tag 增强）；允许 recent_fallback
        if not ctx.ref_mode and not attachment_b64:
            attachment_b64 = await _extract_attachment(ctx)

        # ── 2. WD14 反推 ──
        # i2i / char_ref / vibe：参考图走 NAI 原生通道，不要 WD14 全量反推污染 prompt。
        # 无 ref_mode 的普通附图：仍可 WD14 做 tag 增强。
        reference_tags = ""
        reference_image_for_llm = ""
        if attachment_b64:
            wd14_config = ctx.config.get("wd14", {})
            max_size = int(wd14_config.get("max_image_size", 1024) or 1024)
            # 视觉 LLM 仍需要缩略图；与是否跑 WD14 解耦
            try:
                reference_image_for_llm = _resize_image_for_wd14(attachment_b64, max_size)
            except Exception as exc:
                logger.warning("[Pipeline] 参考图缩略失败，回退原图: %s", exc)
                reference_image_for_llm = attachment_b64

            # 显式参考图模式都吃图像素（i2i/char_ref/vibe），WD14 全量反推只会污染 prompt
            _ref = str(ctx.ref_mode or "").strip().lower().replace("-", "_")
            skip_wd14 = _ref in {"i2i", "char_ref", "vibe"}
            if skip_wd14:
                logger.info(
                    "[Pipeline] ref_mode=%s 跳过 WD14 反推（参考图走 NAI 原生通道；视觉 LLM 仍可看图）",
                    _ref,
                )
            elif _normalize_bool(wd14_config.get("enabled", True)):
                try:
                    from .wd14_client import reverse_tag_image, DEFAULT_ENDPOINT as WD14_DEFAULT
                    endpoint = str(wd14_config.get("endpoint", WD14_DEFAULT) or WD14_DEFAULT)
                    threshold = float(wd14_config.get("threshold", 0.35) or 0.35)
                    timeout = float(wd14_config.get("timeout", 60.0) or 60.0)
                    wd14_result = await reverse_tag_image(
                        reference_image_for_llm,
                        endpoint=endpoint,
                        threshold=threshold,
                        timeout=timeout,
                    )
                    if wd14_result and wd14_result.success:
                        reference_tags = wd14_result.format_for_llm()
                        logger.info("[Pipeline] WD14 反推成功: %s...", reference_tags[:120])
                except Exception as exc:
                    logger.warning("[Pipeline] WD14 反推异常: %s", exc, exc_info=True)

        # ── 3. 预先探查目标后端类型（NAI 4.5 走自然语言，其他后端走 Danbooru Tag）──
        probe_style_router = StyleRouter(ctx.config)
        _, probe_model_cfg, _ = probe_style_router.route(
            selfie_mode=ctx.selfie_mode,
            manual_style=ctx.manual_style,
        )
        target_api_type = str((probe_model_cfg or {}).get("api_type", "newapi_nai") or "newapi_nai").lower().replace("-", "_")
        _ref_probe = str(ctx.ref_mode or "").strip().lower().replace("-", "_")
        if _ref_probe in {"i2i", "char_ref", "vibe"}:
            target_api_type = "newapi_nai"

        # ── 3. Prompt 生成 ──
        prompt_result = await ctx.proxy._generate_prompt_with_style(
            user_request=ctx.user_request or "根据聊天内容生成一张合适的图片",
            chat_messages=ctx.chat_messages,
            persona=ctx.persona,
            selfie_mode=ctx.selfie_mode,
            nsfw_allowed=ctx.nsfw_allowed,
            custom_system_prompt=ctx.custom_system_prompt,
            reference_tags=reference_tags,
            reference_image_base64=reference_image_for_llm,
            api_type=target_api_type,
        )
        if not prompt_result.success:
            await _safe_send(ctx, f"提示词生成失败: {prompt_result.error[:80]}")
            return False

        await _safe_send(ctx, "prompt 生成完成，正在出图...")

        # ── 4. 风格路由 ──
        style_router = StyleRouter(ctx.config)
        selected_style, model_config, route_reason = style_router.route(
            selfie_mode=ctx.selfie_mode,
            manual_style=ctx.manual_style,
            llm_style=prompt_result.style,
        )
        api_type = str((model_config or {}).get("api_type", "openai") or "openai").lower().replace("-", "_")
        logger.info(f"[Pipeline] style={selected_style}, api_type={api_type}, model_config_keys={list(model_config.keys()) if model_config else None}")

        # SD API / legacy 端点只支持文生图；参考图模式自动切到 NewAPI NAI 原生客户端。
        # 普通文生图仍保留原来的 SD API 路由，不改全局默认后端。
        reference_mode = str(ctx.ref_mode or "").strip().lower().replace("-", "_")
        if reference_mode in {"i2i", "char_ref", "vibe"} and api_type != "newapi_nai":
            style_config = ctx.config.get(selected_style, {}) if isinstance(ctx.config, dict) else {}
            nai_config = style_config.get("newapi_nai", {}) if isinstance(style_config, dict) else {}
            if not isinstance(nai_config, dict) or not nai_config.get("base_url") or not nai_config.get("api_key"):
                logger.warning(
                    "[Pipeline] 参考图模式缺少 NewAPI NAI 配置: ref_mode=%s style=%s",
                    reference_mode,
                    selected_style,
                )
                await _safe_send(
                    ctx,
                    f"{reference_mode} 需要 NewAPI NAI 参考图端点，但当前风格没有配置。普通文生图仍可使用。",
                )
                return False

            routed_config = dict(model_config or {})
            routed_config.update({
                "api_type": "newapi_nai",
                "base_url": str(nai_config.get("base_url") or ""),
                "api_key": str(nai_config.get("api_key") or ""),
                "api_key_paid": str(nai_config.get("api_key_paid") or ""),
                "model_name": str(nai_config.get("model_name") or "nai-diffusion-4-5-full"),
                "custom_prompt_add": str(nai_config.get("custom_prompt_add") or ""),
            })
            for endpoint_key in (
                "negative_prompt", "size", "steps", "scale", "sampler", "seed",
                "image_format", "max_tokens", "timeout", "retry_attempts", "proxy_mode",
                "quality_toggle", "auto_smea", "variety_boost", "extra_params",
            ):
                if endpoint_key in nai_config:
                    routed_config[f"newapi_nai_{endpoint_key}"] = nai_config[endpoint_key]
            model_config = routed_config
            api_type = "newapi_nai"
            logger.info(
                "[Pipeline] 参考图模式自动切换 NewAPI NAI: ref_mode=%s style=%s model=%s",
                reference_mode,
                selected_style,
                routed_config.get("model_name"),
            )

        # ── 5. 确定目标尺寸 ──
        # prompt_result.aspect 只是 LLM 建议；最终由用户意图、场景 tag、
        # 参考图比例和 LLM 建议共同裁决，避免“模型一律 portrait”。
        aspect_decision = resolve_aspect(
            user_request=ctx.user_request,
            generated_prompt=str(getattr(prompt_result, "prompt", "") or ""),
            llm_aspect=getattr(prompt_result, "aspect", None),
            has_characters=bool(getattr(prompt_result, "characters", None)),
            selfie_mode=ctx.selfie_mode,
            reference_image_base64=(
                attachment_b64 or ""
                if ctx.attachment_source != "recent_fallback"
                else ""
            ),
        )
        aspect = aspect_decision.aspect
        target_size = _SIZE_MAP.get(aspect, (832, 1216))
        logger.info(
            "[Pipeline] aspect decision: aspect=%s source=%s llm=%s reference=%s scores=%s target_size=%s",
            aspect_decision.aspect,
            aspect_decision.source,
            aspect_decision.llm_aspect or "-",
            aspect_decision.reference_aspect or "-",
            aspect_decision.scores,
            target_size,
        )

        # ── 6. 参考图 resize ──
        ref_image_data_uri = ""
        if ctx.ref_mode and attachment_b64:
            # i2i 要求参考图与输出 size 完全一致；char-ref/vibe 则保留原图比例。
            if ctx.ref_mode == "i2i":
                ref_image_data_uri = _resize_image_for_nai(attachment_b64, target_size)
            else:
                ref_image_data_uri = _image_data_uri_for_reference(attachment_b64)
            if not ref_image_data_uri:
                await _safe_send(ctx, "附图质量不足（太小或损坏），跳过参考图模式")
                ctx.ref_mode = ""

        # ── 7. 出图 ──
        if api_type in ("newapi_nai", "newapi-nai"):
            success = await _generate_with_newapi_nai(
                ctx, prompt_result, model_config, target_size, ref_image_data_uri
            )
        else:
            success = await _generate_with_legacy(ctx, prompt_result, model_config, aspect)

        return success

    except Exception as exc:
        logger.error("[Pipeline] 异常: %s", exc, exc_info=True)
        await _safe_send(ctx, f"画图出错了: {str(exc)[:80]}")
        return False


async def _generate_with_newapi_nai(
    ctx: DrawPipelineContext,
    prompt_result: Any,
    model_config: Optional[dict],
    target_size: tuple[int, int],
    ref_image_data_uri: str,
) -> bool:
    """使用新的 NewApiNaiClient 出图。"""

    mc = model_config or {}

    # 多角色时用 global_prompt 作主 prompt
    base_prompt = prompt_result.prompt
    if prompt_result.characters and prompt_result.global_prompt:
        base_prompt = prompt_result.global_prompt

    # 参考图模式可以覆盖普通文生图的 custom_prompt_add。
    # Vibe 的风格来自参考图，不应叠加 anime 配置里的固定画师串。
    ref_cfg = ctx.config.get("generation", {}).get("ref_image", {})
    prompt_model_config = model_config
    if ctx.ref_mode == "vibe":
        prompt_model_config = dict(model_config or {})
        prompt_model_config["custom_prompt_add"] = str(
            ref_cfg.get("vibe_custom_prompt_add", "{{{masterpiece, best quality}}},") or ""
        )
        logger.info("[Pipeline] vibe 使用独立 prompt 前缀，跳过普通固定画师风格串")

    final_prompt = ctx.proxy._build_final_prompt(base_prompt, prompt_model_config)
    logger.info(f"[Pipeline] final_prompt (first 300): {final_prompt[:300]}")
    logger.info(f"[Pipeline] has azuma_seren: {"azuma_seren" in final_prompt}, has characters: {bool(prompt_result.characters)}")

    # 构造 GenerationContext
    gen_ctx = GenerationContext(
        prompt=final_prompt,
        negative_prompt=str(mc.get("newapi_nai_negative_prompt", "") or ""),
        size=target_size,
        model=str(mc.get("model_name", "nai-diffusion-4-5-full") or "nai-diffusion-4-5-full"),
        steps=int(mc.get("newapi_nai_steps", 28) or 28),
        scale=float(mc.get("newapi_nai_scale", 5.0) or 5.0),
        sampler=str(mc.get("newapi_nai_sampler", "k_euler_ancestral") or "k_euler_ancestral"),
        image_format=str(mc.get("newapi_nai_image_format", "png") or "png"),
        characters=prompt_result.characters,
        timeout=int(mc.get("newapi_nai_timeout", 180) or 180),
        retry_attempts=int(mc.get("newapi_nai_retry_attempts", 3) or 3),
        proxy_mode=str(mc.get("newapi_nai_proxy_mode", "auto") or "auto"),
        quality_toggle=_normalize_bool(mc.get("newapi_nai_quality_toggle", True)),
        auto_smea=_normalize_bool(mc.get("newapi_nai_auto_smea", False)),
        variety_boost=_normalize_bool(mc.get("newapi_nai_variety_boost", False)),
        ref_mode=ctx.ref_mode if ref_image_data_uri else "none",
    )

    # 参考图字段
    ref_cfg = ctx.config.get("generation", {}).get("ref_image", {})
    if ctx.ref_mode == "i2i" and ref_image_data_uri:
        strength, noise, extra_neg, strength_source = _resolve_i2i_params(
            user_request=ctx.user_request,
            prompt_result=prompt_result,
            ref_cfg=ref_cfg if isinstance(ref_cfg, dict) else {},
        )
        gen_ctx.i2i_image = ref_image_data_uri
        gen_ctx.i2i_strength = strength
        gen_ctx.i2i_noise = noise
        gen_ctx.negative_prompt = _merge_negative_prompt(gen_ctx.negative_prompt, extra_neg)
        logger.info(
            "[Pipeline] i2i 已绑定参考图: uri_len=%s strength=%s noise=%s size=%s source=%s negative_extra=%s",
            len(ref_image_data_uri),
            gen_ctx.i2i_strength,
            gen_ctx.i2i_noise,
            target_size,
            strength_source,
            ",".join(extra_neg) if extra_neg else "-",
        )
    elif getattr(prompt_result, "negative_tags", None):
        # 非 i2i 也允许 LLM 追加 negative
        gen_ctx.negative_prompt = _merge_negative_prompt(
            gen_ctx.negative_prompt,
            list(prompt_result.negative_tags or []),
        )
    elif ctx.ref_mode == "char_ref" and ref_image_data_uri:
        gen_ctx.char_ref_image = ref_image_data_uri
        gen_ctx.char_ref_type = str(ref_cfg.get("char_ref_type", "character") or "character")
        gen_ctx.char_ref_fidelity = float(ref_cfg.get("char_ref_fidelity", 1.0) or 1.0)
        gen_ctx.char_ref_strength = float(ref_cfg.get("char_ref_strength", 1.0) or 1.0)
    elif ctx.ref_mode == "vibe" and ref_image_data_uri:
        info_ext = float(ref_cfg.get("vibe_info_extracted", 0.4) or 0.4)
        strength = float(ref_cfg.get("vibe_strength", 0.3) or 0.3)
        gen_ctx.vibe_global_strength = float(ref_cfg.get("vibe_global_strength", 1.0) or 1.0)
        # Check vibe cache_id first
        _vibe_cache = VibeCache()
        _cached_id = _vibe_cache.lookup(ref_image_data_uri, gen_ctx.model, info_ext)
        if _cached_id:
            gen_ctx.vibe_images = [{"cache_id": _cached_id, "strength": strength}]
            logger.info(f"[Pipeline] vibe cache hit: {_cached_id[:8]}...")
        else:
            gen_ctx.vibe_images = [{"image": ref_image_data_uri, "info_extracted": info_ext, "strength": strength}]

    # 客户端（经 factory，统一 ImageClient 接缝）
    client = create_client_from_model_config(mc, log_prefix=f"[{ctx.source}]")
    if client is None:
        await _safe_send(ctx, "画图的 base_url 或 API 密钥没配置")
        return False

    # 调用
    result = await client.generate(gen_ctx)

    # 降级：参考图失败 → 退回 txt2img
    if not result.success and ctx.ref_mode and ctx.ref_mode != "none":
        await _safe_send(ctx, f"{ctx.ref_mode} 出图失败，尝试普通文生图...")
        gen_ctx.ref_mode = "none"
        gen_ctx.i2i_image = None
        gen_ctx.char_ref_image = None
        gen_ctx.vibe_images = None
        result = await client.generate(gen_ctx)

    if not result.success:
        await _safe_send(ctx, f"出图失败: {result.error[:80]}")
        return False

    # Store vibe cache_ids if present
    if result.success and result.vibe_cache_ids and ctx.ref_mode == "vibe" and ref_image_data_uri:
        _vc = VibeCache()
        info_ext = float(ref_cfg.get("vibe_info_extracted", 0.4) or 0.4)
        for entry in result.vibe_cache_ids:
            idx = entry.get("index", 0)
            cid = entry.get("cache_id", "")
            if cid and idx < len(gen_ctx.vibe_images or []):
                _vc.store(ref_image_data_uri, gen_ctx.model, info_ext, cid)

    # 发送
    success, message = await ctx.proxy._handle_image_result(
        result.image_base64, prompt=gen_ctx.prompt
    )
    if not success:
        await _safe_send(ctx, message)
    return success


async def _generate_with_legacy(
    ctx: DrawPipelineContext,
    prompt_result: Any,
    model_config: Optional[dict],
    aspect: str,
) -> bool:
    """非 newapi_nai 端点的旧路径回退。"""
    request = ImageGenerationRequest(
        prompt=prompt_result.prompt,
        selfie_mode=ctx.selfie_mode,
        llm_style=prompt_result.style,
        global_prompt=prompt_result.global_prompt,
        characters=prompt_result.characters,
        aspect=aspect,
    )
    generation_result = await generate_image(
        plugin_config=ctx.config,
        client=ctx.proxy,
        request=request,
    )
    if generation_result.success:
        success, message = await ctx.proxy._handle_image_result(
            generation_result.result, prompt=request.prompt
        )
        if not success:
            await _safe_send(ctx, message)
        return success
    else:
        await _safe_send(ctx, f"画图失败了: {str(generation_result.result)[:80]}")
        return False


async def _safe_send(ctx: DrawPipelineContext, text: str) -> None:
    """安全发送文本消息，不抛异常。"""
    try:
        await ctx.proxy.send_text(text)
    except Exception:
        pass