"""统一的图片生成编排服务。"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple
import asyncio

from src.common.logger import get_logger

from .style_router import StyleRouter

logger = get_logger("MaiBot_LLM2pic")


class ImageClientProtocol(Protocol):
    def get_config(self, path: str, default: Any = None) -> Any: ...

    def _build_final_prompt(self, generated_prompt: str, model_config: Optional[dict] = None) -> str: ...

    def _make_gradio_image_request(
        self,
        prompt: str,
        base_url: Optional[str] = None,
        gradio_params: Optional[dict] = None,
    ) -> Tuple[bool, str]: ...

    def _make_sd_api_request(
        self,
        prompt: str,
        base_url: str,
        api_key: str,
        sd_params: Optional[dict] = None,
    ) -> Tuple[bool, str]: ...

    def _make_regex_url_request(self, prompt: str, url_template: str) -> Tuple[bool, str]: ...

    def _make_novelai_request(
        self,
        prompt: str,
        api_key: str,
        novelai_params: Optional[dict] = None,
    ) -> Tuple[bool, str]: ...

    def _make_newapi_nai_request(
        self,
        prompt: str,
        base_url: str,
        api_key: str,
        model: str,
        params: Optional[dict] = None,
        characters: Optional[list[dict[str, Any]]] = None,
    ) -> Tuple[bool, str]: ...

    def _make_http_image_request(
        self,
        prompt: str,
        model: str,
        size: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        input_image_base64: Optional[str] = None,
    ) -> Tuple[bool, str]: ...


@dataclass
class ImageGenerationRequest:
    prompt: str
    selfie_mode: bool = False
    manual_style: Optional[str] = None
    llm_style: Optional[str] = None
    input_image_base64: Optional[str] = None
    apply_prompt_add: bool = True
    global_prompt: Optional[str] = None
    characters: Optional[list[dict[str, Any]]] = None
    aspect: Optional[str] = None


@dataclass
class ImageGenerationResult:
    success: bool
    result: str
    final_prompt: str = ""
    selected_style: str = ""
    route_reason: str = ""
    api_type: str = ""


@dataclass
class _ImageApiParams:
    api_type: str
    base_url: str
    api_key: str
    model: str
    size: str
    gradio_params: Optional[dict]
    sd_params: Optional[dict]
    novelai_params: Optional[dict]
    newapi_nai_params: Optional[dict]


def _build_api_params(client: ImageClientProtocol, model_config: Optional[dict]) -> _ImageApiParams:
    if model_config is None:
        return _ImageApiParams(
            api_type=str(client.get_config("api.api_type", "openai") or "openai"),
            base_url=str(client.get_config("api.base_url", "") or ""),
            api_key=str(client.get_config("api.api_key", "") or ""),
            model=str(client.get_config("generation.default_model", "gpt-image-1") or "gpt-image-1"),
            size=str(client.get_config("api.size", "") or client.get_config("generation.default_size", "") or ""),
            gradio_params=None,
            sd_params=None,
            novelai_params=None,
            newapi_nai_params={
                "negative_prompt": str(client.get_config("generation.newapi_nai_negative_prompt", "") or ""),
                "size": client.get_config("generation.newapi_nai_size", "portrait"),
                "steps": client.get_config("generation.newapi_nai_steps", 23),
                "scale": client.get_config("generation.newapi_nai_scale", 5),
                "sampler": client.get_config("generation.newapi_nai_sampler", "k_euler_ancestral"),
                "seed": client.get_config("generation.newapi_nai_seed", -1),
                "image_format": client.get_config("generation.newapi_nai_image_format", "png"),
                "max_tokens": client.get_config("generation.newapi_nai_max_tokens", 100000),
                "timeout": client.get_config("generation.newapi_nai_timeout", 180),
                "retry_attempts": client.get_config("generation.newapi_nai_retry_attempts", 3),
                "proxy_mode": client.get_config("generation.newapi_nai_proxy_mode", "auto"),
                "quality_toggle": client.get_config("generation.newapi_nai_quality_toggle", True),
                "auto_smea": client.get_config("generation.newapi_nai_auto_smea", False),
                "variety_boost": client.get_config("generation.newapi_nai_variety_boost", False),
                "extra_params": client.get_config("generation.newapi_nai_extra_params", {}),
            },
        )

    return _ImageApiParams(
        api_type=str(model_config.get("api_type", "openai") or "openai"),
        base_url=str(model_config.get("base_url", "") or ""),
        api_key=str(model_config.get("api_key", "") or ""),
        model=str(model_config.get("model_name", "") or ""),
        size=str(model_config.get("size", "") or client.get_config("generation.default_size", "") or ""),
        gradio_params={
            "resolution": model_config.get("gradio_resolution", "1024x1024 ( 1:1 )"),
            "steps": model_config.get("gradio_steps", 8),
            "shift": model_config.get("gradio_shift", 3),
            "timeout": model_config.get("gradio_timeout", 120),
        },
        sd_params={
            "negative_prompt": model_config.get("sd_negative_prompt", ""),
            "width": model_config.get("sd_width", 512),
            "height": model_config.get("sd_height", 512),
            "steps": model_config.get("sd_steps", 20),
            "cfg": model_config.get("sd_cfg", 7.0),
            "model_index": model_config.get("sd_model_index", 0),
            "seed": model_config.get("sd_seed", -1),
        },
        novelai_params={
            "model": model_config.get("novelai_model", "nai-diffusion-4-5-full"),
            "width": model_config.get("novelai_width", 832),
            "height": model_config.get("novelai_height", 1216),
            "steps": model_config.get("novelai_steps", 28),
            "scale": model_config.get("novelai_scale", 5.0),
            "sampler": model_config.get("novelai_sampler", "k_euler"),
            "negative_prompt": model_config.get("novelai_negative_prompt", ""),
            "seed": model_config.get("novelai_seed", -1),
            "timeout": model_config.get("novelai_timeout", 120),
        },
        newapi_nai_params={
            "negative_prompt": model_config.get("newapi_nai_negative_prompt", ""),
            "size": model_config.get("newapi_nai_size", "portrait"),
            "steps": model_config.get("newapi_nai_steps", 23),
            "scale": model_config.get("newapi_nai_scale", 5),
            "sampler": model_config.get("newapi_nai_sampler", "k_euler_ancestral"),
            "seed": model_config.get("newapi_nai_seed", -1),
            "image_format": model_config.get("newapi_nai_image_format", "png"),
            "max_tokens": model_config.get("newapi_nai_max_tokens", 100000),
            "timeout": model_config.get("newapi_nai_timeout", 180),
            "retry_attempts": model_config.get("newapi_nai_retry_attempts", 3),
            "proxy_mode": model_config.get("newapi_nai_proxy_mode", "auto"),
            "quality_toggle": model_config.get("newapi_nai_quality_toggle", True),
            "auto_smea": model_config.get("newapi_nai_auto_smea", False),
            "variety_boost": model_config.get("newapi_nai_variety_boost", False),
            "extra_params": model_config.get("newapi_nai_extra_params", {}),
        },
    )


def _normalize_aspect(aspect: Optional[str]) -> Optional[str]:
    normalized = str(aspect or "").strip().lower()
    return normalized if normalized in {"portrait", "landscape", "square"} else None


async def generate_image(
    *,
    plugin_config: dict[str, Any],
    client: ImageClientProtocol,
    request: ImageGenerationRequest,
) -> ImageGenerationResult:
    """完成风格路由、API 参数归一化和图片生成请求。"""
    style_router = StyleRouter(plugin_config)
    selected_style, model_config, route_reason = style_router.route(
        selfie_mode=request.selfie_mode,
        manual_style=request.manual_style,
        llm_style=request.llm_style,
    )
    final_prompt = (
        client._build_final_prompt(request.prompt, model_config)
        if request.apply_prompt_add
        else request.prompt
    )
    params = _build_api_params(client, model_config)
    api_type = params.api_type.lower()

    logger.info(
        "[ImageGeneration] route style=%s reason=%s api=%s prompt=%s...",
        selected_style,
        route_reason,
        api_type,
        final_prompt[:120],
    )

    if api_type != "novelai" and not params.base_url:
        return ImageGenerationResult(False, "画图的 base_url 没配置，画不了", final_prompt, selected_style, route_reason, api_type)
    if api_type not in ("gradio", "regex_url") and not params.api_key.strip():
        return ImageGenerationResult(False, "画图的 API 密钥没配，画不了", final_prompt, selected_style, route_reason, api_type)

    try:
        if api_type == "gradio":
            success, result = await asyncio.to_thread(
                client._make_gradio_image_request,
                prompt=final_prompt,
                base_url=params.base_url,
                gradio_params=params.gradio_params,
            )
        elif api_type == "sd_api":
            success, result = await asyncio.to_thread(
                client._make_sd_api_request,
                prompt=final_prompt,
                base_url=params.base_url,
                api_key=params.api_key,
                sd_params=params.sd_params if model_config else None,
            )
        elif api_type == "novelai":
            success, result = await asyncio.to_thread(
                client._make_novelai_request,
                prompt=final_prompt,
                api_key=params.api_key,
                novelai_params=params.novelai_params if model_config else None,
            )
        elif api_type in {"newapi_nai", "newapi-nai"}:
            newapi_prompt = final_prompt
            if request.characters and request.global_prompt:
                newapi_prompt = (
                    client._build_final_prompt(request.global_prompt, model_config)
                    if request.apply_prompt_add
                    else request.global_prompt
                )
            newapi_params = dict(params.newapi_nai_params or {})
            aspect = _normalize_aspect(request.aspect)
            if aspect:
                newapi_params["size"] = aspect
            success, result = await asyncio.to_thread(
                client._make_newapi_nai_request,
                prompt=newapi_prompt,
                base_url=params.base_url,
                api_key=params.api_key,
                model=params.model,
                params=newapi_params,
                characters=request.characters,
            )
        elif api_type == "regex_url":
            success, result = await asyncio.to_thread(
                client._make_regex_url_request,
                prompt=final_prompt,
                url_template=params.base_url,
            )
        else:
            success, result = await asyncio.to_thread(
                client._make_http_image_request,
                prompt=final_prompt,
                model=params.model,
                size=params.size if params.size else None,
                base_url=params.base_url,
                api_key=params.api_key,
                input_image_base64=request.input_image_base64,
            )
    except Exception as exc:
        logger.error("[ImageGeneration] 请求失败: %s", exc, exc_info=True)
        return ImageGenerationResult(False, f"图片生成请求失败: {str(exc)[:100]}", final_prompt, selected_style, route_reason, api_type)

    return ImageGenerationResult(success, result, final_prompt, selected_style, route_reason, api_type)
