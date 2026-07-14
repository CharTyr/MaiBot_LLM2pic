"""
图片生成客户端工厂。

根据 StyleRouter 产出的 model_config 选择 ImageClient 实现。
"""

from __future__ import annotations

from typing import Any, Optional

from .base import ImageClient
from .newapi_nai import NewApiNaiClient


def create_client_from_model_config(
    model_config: Optional[dict[str, Any]],
    *,
    log_prefix: str = "[ImageClient]",
) -> Optional[ImageClient]:
    """从 StyleRouter.route() 的 model_config 创建客户端。

    当前仅 newapi_nai 有独立适配器；其它 api_type 返回 None，由调用方走 legacy。
    """
    if not model_config:
        return None

    api_type = str(model_config.get("api_type", "") or "").lower().replace("-", "_")
    if api_type != "newapi_nai":
        return None

    base_url = str(model_config.get("base_url", "") or "")
    api_key = str(model_config.get("api_key", "") or "")
    if not base_url or not api_key:
        return None

    return NewApiNaiClient(
        base_url=base_url,
        api_key=api_key,
        log_prefix=log_prefix,
    )


def create_client(
    style: str,
    config: dict[str, Any],
    log_prefix: str = "[NAI]",
) -> Optional[ImageClient]:
    """旧入口：从整包 plugin config 猜端点。优先用 create_client_from_model_config。"""
    from ..style_router import StyleRouter

    router = StyleRouter(config)
    _selected, model_config, _reason = router.route(
        selfie_mode=False,
        manual_style=style if style in {"anime", "edit"} else None,
        llm_style=None,
    )
    return create_client_from_model_config(model_config, log_prefix=log_prefix)


def get_model_config(style: str, config: dict[str, Any]) -> Optional[dict]:
    """遗留辅助：请优先使用 StyleRouter.route()。"""
    from ..style_router import StyleRouter

    router = StyleRouter(config)
    _selected, model_config, _reason = router.route(
        selfie_mode=False,
        manual_style=style if style in {"anime", "edit"} else None,
        llm_style=None,
    )
    return model_config
