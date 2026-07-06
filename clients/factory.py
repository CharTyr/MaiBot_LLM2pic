"""
图片生成客户端工厂。

根据配置选择对应的 ImageClient 实现。
"""

from __future__ import annotations

from typing import Any, Optional

from .base import ImageClient
from .newapi_nai import NewApiNaiClient


def create_client(
    style: str,
    config: dict[str, Any],
    log_prefix: str = "[NAI]",
) -> Optional[ImageClient]:
    """根据 style 和配置创建对应的图片客户端。

    Args:
        style: "anime" | "edit" | None
        config: 插件配置 dict
        log_prefix: 日志前缀

    Returns:
        ImageClient 实例，或 None（配置不可用时）
    """
    # 读取端点配置
    endpoints = config.get("endpoints", {})
    anime_config = config.get("anime", {})
    edit_config = config.get("edit", {})

    if style == "edit":
        # edit 走多模态 GPT，暂不重构
        return None  # 由 plugin.py 走旧路径

    # anime / default → NewAPI NAI
    newapi_config = config.get("newapi_nai", {})
    if not newapi_config.get("base_url") or not newapi_config.get("api_key"):
        # 退化到旧配置结构
        api_config = config.get("api", {})
        base_url = str(api_config.get("base_url", "") or "")
        api_key = str(api_config.get("api_key", "") or "")
    else:
        base_url = str(newapi_config.get("base_url", ""))
        api_key = str(newapi_config.get("api_key", ""))

    if not base_url or not api_key:
        return None

    return NewApiNaiClient(
        base_url=base_url,
        api_key=api_key,
        log_prefix=log_prefix,
    )


def get_model_config(style: str, config: dict[str, Any]) -> Optional[dict]:
    """获取指定 style 的模型配置。"""
    if style == "edit":
        edit_config = config.get("edit", {})
        return edit_config if edit_config else None

    # anime / default
    anime_config = config.get("anime", {})
    newapi_config = config.get("newapi_nai", {})

    # 合并：newapi_nai 优先，anime 兜底
    model = newapi_config.get("model") or anime_config.get("model") or "nai-diffusion-4-5-full"
    return {
        "model": model,
        "steps": int(newapi_config.get("steps", 28) or anime_config.get("steps", 28) or 28),
        "scale": float(newapi_config.get("scale", 5.0) or anime_config.get("scale", 5.0) or 5.0),
        "sampler": str(newapi_config.get("sampler", "k_euler_ancestral") or "k_euler_ancestral"),
        "negative_prompt": str(newapi_config.get("negative_prompt", "") or ""),
        "image_format": str(newapi_config.get("image_format", "png") or "png"),
        "timeout": int(newapi_config.get("timeout", 180) or 180),
        "retry_attempts": int(newapi_config.get("retry_attempts", 3) or 3),
        "proxy_mode": str(newapi_config.get("proxy_mode", "auto") or "auto"),
    }