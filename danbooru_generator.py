# -*- coding: utf-8 -*-
"""Danbooru tag prompt generation pipeline."""

from __future__ import annotations

from typing import Any, Optional
import re

from src.common.logger import get_logger

from .core.rules.prompt_rules import PROMPT_GENERATOR_JSON_TEMPLATE, SFW_PROMPT_GENERATOR_JSON_TEMPLATE
from .core.services.tag_candidate_resolver import resolve_tag_candidates
from .core.utils.prompt_output_parser import parse_prompt_from_structured_output, resolve_multi_character_payload
from .core.utils.prompt_postprocessor import (
    normalize_characters_order,
    normalize_prompt_order,
    remove_selfie_appearance_from_characters,
    remove_selfie_appearance_tags,
    sanitize_sfw_characters,
    sanitize_sfw_prompt,
    user_mentions_appearance,
)

logger = get_logger("MaiBot_LLM2pic")


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
) -> tuple[bool, str, Optional[str], Optional[dict[str, Any]]]:
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
    )

    try:
        result = await llm.generate(
            prompt=full_prompt,
            model=model,
            temperature=float(llm_config.get("temperature", 0.2) or 0.2),
        )
    except Exception as exc:
        logger.error("[DanbooruPrompt] LLM 调用失败: %s", exc, exc_info=True)
        return False, str(exc), None, None

    if not isinstance(result, dict):
        return False, f"LLM 返回非 dict: {type(result).__name__}", None, None
    if not bool(result.get("success", False)):
        return False, str(result.get("error") or "LLM生成失败"), None, None

    response_text = str(result.get("response") or "").strip()
    generated_prompt = _cleanup_llm_prompt(response_text)
    if not generated_prompt:
        return False, "LLM返回空提示词", None, None

    multi_payload = resolve_multi_character_payload(response_text, generated_prompt)
    selfie_appearance_policy = str(llm_config.get("selfie_appearance_policy", "auto") or "auto").strip().lower()
    enforce_tag_order = _bool_config(config, "llm.enforce_tag_order", True)
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
        sfw_mode=sfw_mode,
        enforce_tag_order=enforce_tag_order,
        selfie_appearance_policy=selfie_appearance_policy,
    )

    logger.info(
        "[DanbooruPrompt] generated sfw=%s nsfw_allowed=%s selfie=%s tags=%s",
        sfw_mode,
        nsfw_allowed,
        selfie_mode,
        generated_prompt[:500],
    )

    return True, generated_prompt, "anime", multi_payload
