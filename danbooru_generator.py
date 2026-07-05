# -*- coding: utf-8 -*-
"""Danbooru tag prompt generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import asyncio
import re

from src.common.logger import get_logger

from .core.rules.prompt_rules import PROMPT_GENERATOR_JSON_TEMPLATE, SFW_PROMPT_GENERATOR_JSON_TEMPLATE
from .core.services.tag_candidate_resolver import resolve_tag_candidates
from .core.utils.prompt_output_parser import (
    extract_aspect_from_structured_output,
    parse_prompt_from_structured_output,
    resolve_multi_character_payload,
)
from .core.utils.prompt_postprocessor import (
    normalize_characters_order,
    normalize_prompt_order,
    remove_self_character_from_characters,
    remove_self_character_tags,
    remove_selfie_appearance_from_characters,
    remove_selfie_appearance_tags,
    sanitize_sfw_characters,
    sanitize_sfw_prompt,
    user_requests_self_character,
    user_mentions_appearance,
)

logger = get_logger("MaiBot_LLM2pic")


@dataclass
class PromptGenerationResult:
    success: bool
    prompt: str = ""
    style: Optional[str] = None
    global_prompt: Optional[str] = None
    characters: Optional[list[dict[str, Any]]] = None
    aspect: Optional[str] = None
    error: str = ""


def _bool_config(config: dict[str, Any], path: str, default: bool) -> bool:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)




_OPER_ERROR_MARKERS = (
    "负载过高",
    "请稍后再试",
    "请求负载",
    "服务繁忙",
    "服务不可用",
    "try again later",
    "too many requests",
    "rate limit",
    "overloaded",
    "503 service",
    "502 bad gateway",
)


def _is_llm_operational_error(text: str) -> bool:
    """LLM/网关返回的说明性错误，不能当作绘图 prompt。"""
    normalized = str(text or "").strip()
    if not normalized:
        return True
    lower = normalized.lower()
    if any(marker in normalized for marker in _OPER_ERROR_MARKERS):
        return True
    if any(marker in lower for marker in ("rate limit", "overloaded", "too many requests")):
        return True
    if "模型「" in normalized and ("负载" in normalized or "稍后再试" in normalized):
        return True
    return False


def _looks_like_danbooru_prompt(text: str) -> bool:
    """粗判是否为 Danbooru tag 串，过滤中文说明/拒答。"""
    normalized = str(text or "").strip()
    if not normalized or _is_llm_operational_error(normalized):
        return False

    if parse_prompt_from_structured_output(normalized):
        return True

    lower = normalized.lower()
    tag_signals = (
        ",",
        "{{",
        "}}",
        "masterpiece",
        "best quality",
        "1girl",
        "1boy",
        "solo",
        "character:",
        "source#",
        "target#",
    )
    if any(signal in lower for signal in tag_signals):
        return True
    if re.search(r"[a-z0-9_]+\s*,\s*[a-z0-9_]+", normalized, flags=re.IGNORECASE):
        return True

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", normalized))
    if cjk_count > 0:
        return False
    return len(normalized) >= 12


def _validate_prompt_llm_response(response_text: str, cleaned_prompt: str) -> tuple[bool, str]:
    raw = str(response_text or "").strip()
    cleaned = str(cleaned_prompt or "").strip()
    if _is_llm_operational_error(raw) or _is_llm_operational_error(cleaned):
        return False, (raw or cleaned)[:160]
    if not cleaned:
        return False, "LLM返回空提示词"
    if not _looks_like_danbooru_prompt(cleaned):
        return False, (raw or cleaned)[:160]
    return True, ""


def _cleanup_llm_prompt(prompt: str) -> str:
    if not prompt:
        return ""

    parsed_prompt = parse_prompt_from_structured_output(prompt)
    if parsed_prompt:
        return parsed_prompt

    cleaned = prompt.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
        if "\n" in cleaned:
            first_line, rest = cleaned.split("\n", 1)
            if first_line.strip().isalpha() and len(first_line.strip()) < 15:
                cleaned = rest.strip()

    cleaned = re.sub(r"^\s*prompt\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("，", ", ")
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip("` \n")


def _render_generator_prompt(
    *,
    template: str,
    user_request: str,
    chat_messages: str,
    persona: str,
    selfie_mode: bool,
    custom_system_prompt: str,
    tag_candidates: str,
    reference_tags: str = "",
) -> str:
    custom_block = custom_system_prompt.strip()
    if custom_block:
        custom_block = custom_block.replace("{persona}", persona).strip() + "\n\n"
    selfie_hint = "用户明确请求自拍/当前状态，请按自拍模式生成。" if selfie_mode else ""
    request_text = f"""## 用户的绘图请求（最高优先级）
{user_request.strip() or "根据聊天内容生成一张合适的图片"}

## 最近的聊天记录（只能用于补充场景、氛围、情绪或消歧，不能替换主体）
{chat_messages.strip() or "（暂无聊天记录）"}

## 硬性优先级
- 用户绘图请求永远高于聊天上下文和人设。
- 只有用户明确要求画东雪莲、你、自拍或你的当前状态时，才允许注入东雪莲/Azuma Seren/角色专属标签。
- 如果用户请求和聊天记录冲突，以用户请求为准。
"""
    prompt = template.replace("<<CUSTOM_SYSTEM_PROMPT>>", custom_block).strip()
    prompt = prompt.replace("<<TAG_CANDIDATES>>", tag_candidates).strip()
    ref_block = reference_tags.strip() if reference_tags else ""
    prompt = prompt.replace("<<REFERENCE_TAGS>>", ref_block).strip()
    prompt = prompt.replace("<<PREVIOUS_PROMPT>>", "").strip()
    prompt = prompt.replace("<<REPLY_CONTEXT>>", "").strip()
    prompt = prompt.replace("<<REASONING_CONTEXT>>", "").strip()
    prompt = prompt.replace("<<CURRENT_TIME_CONTEXT>>", "").strip()
    prompt = prompt.replace("<<SELFIE_HINT>>", selfie_hint).strip()
    prompt = prompt.replace("<<SELFIE_SCENE_CONTEXT>>", "").strip()
    prompt = prompt.replace("<<USER_REQUEST>>", request_text)
    return prompt


def _postprocess_multi_character_payload(
    payload: Optional[dict[str, Any]],
    *,
    user_request: str,
    selfie_mode: bool,
    self_character_requested: bool,
    sfw_mode: bool,
    enforce_tag_order: bool,
    selfie_appearance_policy: str,
) -> Optional[dict[str, Any]]:
    if not payload:
        return None

    global_text = str(payload.get("global_text") or "").strip()
    raw_characters = payload.get("characters")
    if not global_text or not isinstance(raw_characters, list):
        return None

    characters = [dict(item) for item in raw_characters if isinstance(item, dict)]
    if len(characters) < 2:
        return None

    if not self_character_requested:
        global_text, characters = remove_self_character_from_characters(
            global_text,
            characters,
            remove_persona_appearance=not user_mentions_appearance(user_request),
        )
    elif "character:azuma" not in global_text.lower():
        global_text = "{{{character:AzumaSeren}}}, " + global_text
    if selfie_mode and not user_mentions_appearance(user_request) and selfie_appearance_policy in {"auto", "never"}:
        global_text, characters = remove_selfie_appearance_from_characters(global_text, characters)
    if enforce_tag_order:
        global_text, characters = normalize_characters_order(global_text, characters)
    if sfw_mode:
        global_text, characters = sanitize_sfw_characters(global_text, characters)

    if len(characters) < 2 or not global_text.strip():
        return None

    return {
        "global_text": global_text.strip(),
        "characters": characters,
        "has_coords": bool(payload.get("has_coords")) and all(str(item.get("position") or "").strip() for item in characters),
    }


def _infer_aspect_from_text(
    prompt: str,
    *,
    user_request: str,
    selfie_mode: bool,
    has_characters: bool,
) -> str:
    """LLM 未给出画幅时按题材做保守兜底。"""
    text = f"{user_request} {prompt}".lower()
    if any(token in text for token in ("close-up", "face focus", "icon", "sticker", "表情包", "头像", "近景")):
        return "square"
    if selfie_mode or any(token in text for token in ("selfie", "portrait", "半身", "全身", "立绘")):
        return "portrait"
    if has_characters or any(
        token in text
        for token in (
            "2girls",
            "3girls",
            "1boy 1girl",
            "multiple",
            "group",
            "scenery",
            "landscape",
            "wide shot",
            "panorama",
            "car",
            "vehicle",
            "building",
            "cityscape",
            "风景",
            "横图",
            "群像",
            "多人",
            "车辆",
            "建筑",
        )
    ):
        return "landscape"
    return "portrait"


async def generate_danbooru_prompt(
    *,
    config: dict[str, Any],
    llm: Any,
    model: str,
    user_request: str,
    chat_messages: str,
    persona: str,
    selfie_mode: bool,
    nsfw_allowed: bool,
    custom_system_prompt: str = "",
    reference_tags: str = "",
    reference_image_base64: str = "",
) -> PromptGenerationResult:
    """Generate Danbooru tags using the vendored nai_draw_plugin-style pipeline."""
    llm_config = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
    sfw_mode = bool(llm_config.get("danbooru_sfw_mode", True)) and not nsfw_allowed
    template = SFW_PROMPT_GENERATOR_JSON_TEMPLATE if sfw_mode else PROMPT_GENERATOR_JSON_TEMPLATE
    retriever_config = config.get("tag_retriever")
    tag_candidates = await resolve_tag_candidates(
        retriever_config if isinstance(retriever_config, dict) else {},
        user_request,
        log_prefix="[DanbooruPrompt]",
    )
    full_prompt = _render_generator_prompt(
        template=template,
        user_request=user_request,
        chat_messages=chat_messages,
        persona=persona,
        selfie_mode=selfie_mode,
        custom_system_prompt=custom_system_prompt,
        tag_candidates=tag_candidates,
        reference_tags=reference_tags,
    )

    max_attempts = max(1, min(int(llm_config.get("prompt_retry_attempts", 3) or 3), 5))
    base_delay = max(0.5, float(llm_config.get("prompt_retry_delay_seconds", 2) or 2))
    temperature = float(llm_config.get("temperature", 0.2) or 0.2)
    last_error = "LLM生成失败"
    vision_failed = False  # if vision call fails, downgrade subsequent retries to text-only

    result: dict[str, Any] | None = None
    response_text = ""
    generated_prompt = ""

    for attempt in range(1, max_attempts + 1):
        try:
            generate_kwargs: dict[str, Any] = {
                "prompt": full_prompt,
                "model": model,
                "temperature": temperature,
            }
            if reference_image_base64 and not vision_failed:
                generate_kwargs["image_base64"] = reference_image_base64
                generate_kwargs["chat_messages"] = chat_messages
            result = await llm.generate(**generate_kwargs)
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "[DanbooruPrompt] LLM 调用异常 attempt=%s/%s: %s",
                attempt,
                max_attempts,
                exc,
            )
            if reference_image_base64 and not vision_failed:
                vision_failed = True
                logger.info("[DanbooruPrompt] vision 调用失败，后续 retry 降级为纯文本（用 WD14+VLM tag）")
            if attempt < max_attempts:
                await asyncio.sleep(base_delay * attempt)
                continue
            logger.error("[DanbooruPrompt] LLM 调用失败: %s", exc, exc_info=True)
            return PromptGenerationResult(False, error=str(exc))

        if not isinstance(result, dict):
            last_error = f"LLM 返回非 dict: {type(result).__name__}"
        elif not bool(result.get("success", False)):
            last_error = str(result.get("error") or "LLM生成失败")
        else:
            response_text = str(result.get("response") or "").strip()
            generated_prompt = _cleanup_llm_prompt(response_text)
            ok, reason = _validate_prompt_llm_response(response_text, generated_prompt)
            if ok:
                break
            last_error = f"无效提示词: {reason}"
            logger.warning(
                "[DanbooruPrompt] 提示词无效 attempt=%s/%s: %s",
                attempt,
                max_attempts,
                reason[:120],
            )

        if attempt < max_attempts:
            await asyncio.sleep(base_delay * attempt)
            continue
        return PromptGenerationResult(False, error=last_error[:200])

    if not generated_prompt:
        return PromptGenerationResult(False, error=last_error[:200])

    multi_payload = resolve_multi_character_payload(response_text, generated_prompt)
    aspect = extract_aspect_from_structured_output(response_text) or _infer_aspect_from_text(
        generated_prompt,
        user_request=user_request,
        selfie_mode=selfie_mode,
        has_characters=bool(multi_payload),
    )
    selfie_appearance_policy = str(llm_config.get("selfie_appearance_policy", "auto") or "auto").strip().lower()
    enforce_tag_order = _bool_config(config, "llm.enforce_tag_order", True)
    self_character_requested = user_requests_self_character(user_request)
    if not self_character_requested:
        generated_prompt = remove_self_character_tags(
            generated_prompt,
            remove_persona_appearance=not user_mentions_appearance(user_request),
        )
    elif "character:azuma" not in generated_prompt.lower():
        # LLM 未输出角色 tag，强制注入
        generated_prompt = "{{{character:AzumaSeren}}}, " + generated_prompt
    if selfie_mode and not user_mentions_appearance(user_request):
        if selfie_appearance_policy in {"auto", "never"}:
            generated_prompt = remove_selfie_appearance_tags(generated_prompt)
    if enforce_tag_order:
        generated_prompt = normalize_prompt_order(generated_prompt)
    if sfw_mode:
        generated_prompt = sanitize_sfw_prompt(generated_prompt)
    multi_payload = _postprocess_multi_character_payload(
        multi_payload,
        user_request=user_request,
        selfie_mode=selfie_mode,
        self_character_requested=self_character_requested,
        sfw_mode=sfw_mode,
        enforce_tag_order=enforce_tag_order,
        selfie_appearance_policy=selfie_appearance_policy,
    )

    logger.info(
        "[DanbooruPrompt] generated sfw=%s nsfw_allowed=%s selfie=%s tags=%s",
        sfw_mode,
        nsfw_allowed,
        selfie_mode,
        f"aspect={aspect or '-'} {generated_prompt[:500]}",
    )

    return PromptGenerationResult(
        success=True,
        prompt=generated_prompt,
        style="anime",
        global_prompt=str(multi_payload.get("global_text") or "").strip() if multi_payload else None,
        characters=multi_payload.get("characters") if multi_payload else None,
        aspect=aspect,
    )
