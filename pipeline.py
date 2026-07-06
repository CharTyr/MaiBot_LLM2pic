"""
Draw pipeline 编排器。

将原来的四层嵌套（handle → _background → _background_inner → _run_generation_and_send）
拍平为单函数 run_draw_pipeline()。

支持参考图模式：i2i / char_ref / vibe，通过 ref_mode 参数显式触发。
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Optional

from src.common.logger import get_logger

from .clients.base import GenerationContext, calc_max_tokens
from .clients.newapi_nai import NewApiNaiClient
from .style_router import StyleRouter
from .generation_service import (
    generate_image,
    ImageGenerationRequest,
    _normalize_aspect,
)
from .utils import _normalize_bool, _resize_image_for_wd14

logger = get_logger("MaiBot_LLM2pic")

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


def _resize_image_for_nai(image_base64: str, target_size: tuple[int, int]) -> str:
    """将图片 resize 到 NAI 要求的精确尺寸，返回 data URI。

    Returns:
        data URI 字符串，或空字符串（图片太小/损坏时）。
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    except Exception:
        return ""

    min_side = min(img.width, img.height)
    if min_side < 256:
        return ""

    w, h = target_size
    img = img.resize((w, h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


async def _extract_attachment(ctx: DrawPipelineContext) -> Optional[str]:
    """从最近消息中提取附图 base64。"""
    try:
        if ctx.source == "direct_pic" and ctx.proxy is not None:
            return await ctx.proxy._extract_input_image()
        elif ctx.plugin is not None:
            return await ctx.plugin._ctx_extract_image_from_recent(ctx.stream_id)
    except Exception as exc:
        logger.warning("[Pipeline] 提取附图失败: %s", exc)
    return None


async def run_draw_pipeline(ctx: DrawPipelineContext) -> bool:
    """Draw pipeline 主入口。返回 True 表示成功。"""

    try:
        # ── 1. 附图检测 ──
        attachment_b64 = None
        if ctx.ref_mode:
            attachment_b64 = await _extract_attachment(ctx)
            if not attachment_b64:
                await _safe_send(ctx, f"指定了 {ctx.ref_mode} 模式但没检测到附图，按普通文生图处理")
                ctx.ref_mode = ""

        # 也检测无 ref_mode 时的附图（用于 WD14 tag 增强）
        if not attachment_b64 and ctx.source == "direct_pic" and ctx.proxy:
            try:
                attachment_b64 = await ctx.proxy._extract_input_image()
            except Exception:
                pass
        if not attachment_b64 and ctx.source == "draw_picture" and ctx.plugin:
            try:
                attachment_b64 = await ctx.plugin._ctx_extract_image_from_recent(ctx.stream_id)
            except Exception:
                pass

        # ── 2. WD14 反推（有附图时都做）──
        reference_tags = ""
        reference_image_for_llm = ""
        if attachment_b64:
            wd14_config = ctx.config.get("wd14", {})
            if _normalize_bool(wd14_config.get("enabled", True)):
                try:
                    max_size = int(wd14_config.get("max_image_size", 1024) or 1024)
                    reference_image_for_llm = _resize_image_for_wd14(attachment_b64, max_size)
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
        api_type = str((model_config or {}).get("api_type", "openai") or "openai").lower()

        # ── 5. 确定目标尺寸 ──
        aspect = _normalize_aspect(prompt_result.aspect) or "portrait"
        target_size = _SIZE_MAP.get(aspect, (832, 1216))

        # ── 6. 参考图 resize ──
        ref_image_data_uri = ""
        if ctx.ref_mode and attachment_b64:
            ref_image_data_uri = _resize_image_for_nai(attachment_b64, target_size)
            if not ref_image_data_uri:
                await _safe_send(ctx, "附图质量不足（太小或损坏），跳过参考图模式")
                ctx.ref_mode = ""

        # ── 7. 出图 ──
        if api_type in ("newapi_nai", "newapi-nai"):
            success = await _generate_with_newapi_nai(
                ctx, prompt_result, model_config, target_size, ref_image_data_uri
            )
        else:
            success = await _generate_with_legacy(ctx, prompt_result, model_config)

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

    # 加 custom_prompt_add
    final_prompt = ctx.proxy._build_final_prompt(base_prompt, model_config)

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
        gen_ctx.i2i_image = ref_image_data_uri
        gen_ctx.i2i_strength = float(ref_cfg.get("i2i_strength", 0.7) or 0.7)
        gen_ctx.i2i_noise = float(ref_cfg.get("i2i_noise", 0.0) or 0.0)
    elif ctx.ref_mode == "char_ref" and ref_image_data_uri:
        gen_ctx.char_ref_image = ref_image_data_uri
        gen_ctx.char_ref_type = str(ref_cfg.get("char_ref_type", "character") or "character")
        gen_ctx.char_ref_fidelity = float(ref_cfg.get("char_ref_fidelity", 1.0) or 1.0)
        gen_ctx.char_ref_strength = float(ref_cfg.get("char_ref_strength", 1.0) or 1.0)
    elif ctx.ref_mode == "vibe" and ref_image_data_uri:
        info_ext = float(ref_cfg.get("vibe_info_extracted", 0.4) or 0.4)
        strength = float(ref_cfg.get("vibe_strength", 0.3) or 0.3)
        gen_ctx.vibe_images = [{"image": ref_image_data_uri, "info_extracted": info_ext, "strength": strength}]
        gen_ctx.vibe_global_strength = float(ref_cfg.get("vibe_global_strength", 1.0) or 1.0)

    # 客户端
    base_url = str(mc.get("base_url", "") or "")
    api_key = str(mc.get("api_key", "") or "")
    if not base_url or not api_key:
        await _safe_send(ctx, "画图的 base_url 或 API 密钥没配置")
        return False

    client = NewApiNaiClient(base_url=base_url, api_key=api_key, log_prefix=f"[{ctx.source}]")

    # 调用
    import asyncio
    result = await asyncio.to_thread(lambda: asyncio.run(client.generate(gen_ctx)) if False else None)
    # NewApiNaiClient.generate 是 async 但内部用同步 urllib
    # 用 asyncio.to_thread 包装同步部分
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
) -> bool:
    """非 newapi_nai 端点的旧路径回退。"""
    request = ImageGenerationRequest(
        prompt=prompt_result.prompt,
        selfie_mode=ctx.selfie_mode,
        llm_style=prompt_result.style,
        global_prompt=prompt_result.global_prompt,
        characters=prompt_result.characters,
        aspect=prompt_result.aspect,
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