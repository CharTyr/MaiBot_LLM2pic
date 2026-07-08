"""
MaiBot_LLM2pic - MaiBot图片生成插件

使用LLM根据聊天记录和人设生成符合需求的prompt，然后调用图片生成API
支持文生图和图生图功能
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal, Optional
import asyncio
import json
import re
import time

from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase, Tool
from .config import (
    PluginSectionConfig,
    GenerationConfig,
    GenerationGuardConfig,
    RegexUrlEndpointConfig,
    OpenAIEndpointConfig,
    GradioEndpointConfig,
    SdApiEndpointConfig,
    NovelAIEndpointConfig,
    NewApiNaiEndpointConfig,
    AnimeConfig,
    EditConfig,
    LlmConfig,
    TagRetrieverConfig,
    Wd14Config,
    ComponentsConfig,
    GitHubConfig,
    LLM2PicPluginConfig,
)
from maibot_sdk.types import ToolParameterInfo, ToolParamType

from src.common.logger import get_logger

from .utils import _normalize_bool, _resize_image_for_edit, _resize_image_for_wd14
from .style_router import StyleRouter
from .actions import DrawPictureToolMetadata
from .commands import DirectPicCommand
from .bridge import _RuntimeBridgeMixin, _ToolRuntimeProxy, _CommandRuntimeProxy
from .generation_service import ImageGenerationRequest, generate_image
from .pipeline import DrawPipelineContext, run_draw_pipeline
from .wd14_client import reverse_tag_image, DEFAULT_ENDPOINT as WD14_DEFAULT_ENDPOINT

logger = get_logger("MaiBot_LLM2pic")

_CONFIG_VERSION = "4.2.0"




_DRAW_PICTURE_TOOL_PARAMETERS = [
    ToolParameterInfo(
        name="description",
        param_type=ToolParamType.STRING,
        description=(
            "用户想要生成的图片描述，可以是中文或英文。必须包含用户明确指定的主体/角色名/动作/场景；"
            "禁止把普通角色请求改成东雪莲或你自己；只是 @你/提到你/问你问题时禁止调用。"
        ),
        required=True,
        default="",
    ),
    ToolParameterInfo(
        name="selfie_mode",
        param_type=ToolParamType.BOOLEAN,
        description="是否生成自拍模式图片；当用户要求自拍、想看你当前状态或环境时设为 true。",
        required=False,
        default=False,
    ),
    ToolParameterInfo(
        name="nsfw_allowed",
        param_type=ToolParamType.BOOLEAN,
        description="是否允许本次生图按 NSFW Danbooru 规则生成；仅当用户明确请求成人向/NSFW 内容时设为 true，默认 false。",
        required=False,
        default=False,
    ),
    ToolParameterInfo(
        name="use_reference_image",
        param_type=ToolParamType.BOOLEAN,
        description=(
            "是否从用户发送或引用的图片中反推 Danbooru tag 作为参考。当用户发送/引用了图片并要求"
            "基于该图片仿画风/仿角色/照着画/参考画时设为 true。插件会自动提取图片、WD14 反推 tag、"
            "注入 LLM 上下文与用户文字融合生成最终 prompt。如果用户没有发图或引用图，不要设为 true。"
        ),
        required=False,
        default=False,
    ),
    ToolParameterInfo(
        name="reference_mode",
        param_type=ToolParamType.STRING,
        description=(
            "参考图用途，仅 use_reference_image=true 时生效：i2i = 照姿势/构图/动作画（默认）；char_ref = 用这个角色/脸画；vibe = 仿画风/氛围。不确定时用 i2i。"
        ),
        required=False,
        default="i2i",
    ),
]

_EDIT_PICTURE_TOOL_DESCRIPTION = (
    "使用 GPT 图像模型生成或编辑图片。当用户发送/引用图片并明确要求修改、P 图、换背景、加/删元素、"
    "重绘或变换风格时使用；当用户明确要求写实、真实、照片级、realistic 风格时也使用。"
    "不要在用户只是分享/评价图片、没有明确编辑动词，或要求二次元/动漫文生图时使用。"
)

_EDIT_PICTURE_TOOL_PARAMETERS = [
    ToolParameterInfo(
        name="description",
        param_type=ToolParamType.STRING,
        description="用户的图片需求描述，例如“变成动漫风格”“画一只猫在月球上”“把背景换成海边”。",
        required=False,
        default="",
    )
]


class LLM2PicPlugin(MaiBotPlugin, _RuntimeBridgeMixin):
    """LLM2pic 的 rdev 原生插件入口。"""

    config_model = LLM2PicPluginConfig

    _pending_generation_streams: dict[str, float] = {}  # stream_key -> acquire timestamp
    _last_guarded_draw_at: dict[str, float] = {}
    _background_tasks: set[asyncio.Task] = set()
    _global_concurrency: int = 0  # current global concurrent image generations
    _global_concurrency_limit: int = 3  # max concurrent generations across all streams
    _NEGATIVE_DRAW_INTENT_KEYWORDS = (
        "别画",
        "不要画",
        "不用画",
        "别出图",
        "不要出图",
        "不用出图",
        "别生成图",
        "不要生成图",
        "文字回复就行",
        "不要图片",
        "别发图",
        "no image",
        "don't draw",
        "do not draw",
    )
    _EXPLICIT_DRAW_INTENT_KEYWORDS = (
        "画",
        "出图",
        "生成图",
        "来一张",
        "再来一张",
        "自拍",
        "配图",
        "draw",
        "image",
        "picture",
        "selfie",
    )
    _EXPLICIT_DRAW_INTENT_RE = re.compile(
        r"(?:^|[\s:：，,。！!？?])(?:画|绘制|生成|做|整)(?:一个|一张|个|张|点|些)?"
        r"|(?:画|出|生成|做|来|发|整|给|帮|求|想看|看看).{0,8}(?:图|图片|照片|画|自拍)"
        r"|(?:图|图片|照片|画|自拍).{0,8}(?:画|出|生成|做|来|发|整|给|帮|求|想看|看看)"
        r"|(?:draw|generate|make|send).{0,16}(?:image|picture|selfie)"
        r"|(?:image|picture|selfie).{0,16}(?:draw|generate|make|send)",
        re.IGNORECASE,
    )

    def normalize_plugin_config(self, config_data: Mapping[str, Any] | None) -> tuple[dict[str, Any], bool]:
        raw_config = deepcopy(dict(config_data)) if isinstance(config_data, Mapping) else {}
        normalized_config = self._normalize_legacy_endpoint_config(raw_config)
        if normalized_config:
            plugin_config = normalized_config.get("plugin")
            if not isinstance(plugin_config, dict):
                normalized_config["plugin"] = {"enabled": True, "config_version": _CONFIG_VERSION}
            elif not str(plugin_config.get("config_version", "") or "").strip():
                plugin_config["config_version"] = _CONFIG_VERSION
        base_normalized, changed = super().normalize_plugin_config(normalized_config)
        return base_normalized, changed or base_normalized != raw_config

    def get_webui_config_schema(
        self,
        *,
        plugin_id: str = "",
        plugin_name: str = "",
        plugin_version: str = "",
        plugin_description: str = "",
        plugin_author: str = "",
    ) -> dict[str, Any]:
        schema = super().get_webui_config_schema(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            plugin_description=plugin_description,
            plugin_author=plugin_author,
        )
        self._flatten_endpoint_sections_for_webui(schema)
        self._fix_textarea_fields_for_webui(schema)
        return schema

    @staticmethod
    def _fix_textarea_fields_for_webui(schema: dict[str, Any]) -> None:
        sections = schema.get("sections")
        if not isinstance(sections, dict):
            return
        llm_section = sections.get("llm")
        if not isinstance(llm_section, dict):
            return
        fields = llm_section.get("fields")
        if not isinstance(fields, dict):
            return
        system_prompt = fields.get("system_prompt")
        if isinstance(system_prompt, dict):
            system_prompt["ui_type"] = "textarea"
            system_prompt["rows"] = 12

    @classmethod
    def _flatten_endpoint_sections_for_webui(cls, schema: dict[str, Any]) -> None:
        sections = schema.get("sections")
        if not isinstance(sections, dict):
            return
        for style in ("anime", "edit"):
            section = sections.get(style)
            if not isinstance(section, dict):
                continue
            fields = section.get("fields")
            if not isinstance(fields, dict):
                continue
            for endpoint in ("regex_url", "newapi_nai", "openai", "gradio", "sd_api", "novelai"):
                endpoint_field = fields.pop(endpoint, None)
                if not isinstance(endpoint_field, dict):
                    continue
                endpoint_defaults = endpoint_field.get("default")
                if not isinstance(endpoint_defaults, dict):
                    continue
                endpoint_field_order = int(endpoint_field.get("order") or 0)
                endpoint_section_key = f"{style}_{endpoint}"
                sections[endpoint_section_key] = {
                    "name": f"{style}.{endpoint}",
                    "title": f"{style}.{endpoint}",
                    "description": endpoint_field.get("description") or f"{style}.{endpoint} 端点参数",
                    "icon": section.get("icon"),
                    "collapsed": endpoint != "newapi_nai",
                    "order": int(section.get("order") or 0) * 100 + endpoint_field_order + 1,
                    "fields": {},
                }
                cls._add_flat_endpoint_fields(
                    sections[endpoint_section_key]["fields"],
                    endpoint=endpoint,
                    endpoint_defaults=endpoint_defaults,
                    base_order=0,
                )

    @classmethod
    def _add_flat_endpoint_fields(
        cls,
        fields: dict[str, Any],
        *,
        endpoint: str,
        endpoint_defaults: dict[str, Any],
        base_order: int,
    ) -> None:
        for index, (field_name, default_value) in enumerate(endpoint_defaults.items()):
            fields[field_name] = cls._build_flat_endpoint_field(
                name=field_name,
                endpoint=endpoint,
                field_name=field_name,
                default_value=default_value,
                order=base_order + index,
            )

    @staticmethod
    def _build_flat_endpoint_field(
        *,
        name: str,
        endpoint: str,
        field_name: str,
        default_value: Any,
        order: int,
    ) -> dict[str, Any]:
        hidden = False
        if isinstance(default_value, bool):
            field_type = "boolean"
            ui_type = "switch"
        elif isinstance(default_value, int) and not isinstance(default_value, bool):
            field_type = "integer"
            ui_type = "number"
        elif isinstance(default_value, float):
            field_type = "number"
            ui_type = "number"
        elif isinstance(default_value, dict):
            field_type = "string"
            ui_type = "textarea"
            default_value = json.dumps(default_value, ensure_ascii=False, indent=2)
            hidden = True
        else:
            field_type = "string"
            ui_type = "text"

        choices = None
        if field_name == "api_key":
            ui_type = "password"
        elif field_name == "image_format":
            field_type = "select"
            ui_type = "select"
            choices = ["png", "webp"]
        elif field_name == "proxy_mode":
            field_type = "select"
            ui_type = "select"
            choices = ["auto", "inherit", "direct"]

        return {
            "name": name,
            "type": field_type,
            "default": default_value,
            "description": f"{endpoint}.{field_name}",
            "required": False,
            "choices": choices,
            "min": None,
            "max": None,
            "step": None,
            "pattern": None,
            "max_length": None,
            "label": name,
            "placeholder": None,
            "hint": f"写入当前端点分组的 {field_name}",
            "icon": None,
            "hidden": hidden,
            "disabled": False,
            "order": order,
            "input_type": None,
            "ui_type": ui_type,
            "rows": 4 if ui_type == "textarea" else 3,
            "group": endpoint,
            "depends_on": None,
            "depends_value": None,
            "item_type": None,
            "item_fields": None,
            "min_items": None,
            "max_items": None,
            "example": None,
        }

    @staticmethod
    def _normalize_legacy_endpoint_config(config_data: dict[str, Any]) -> dict[str, Any]:
        """兼容旧平铺配置，并把端点字段归入当前 api_type 分组。"""
        for style in ("anime", "edit"):
            style_config = config_data.get(style)
            if not isinstance(style_config, dict):
                continue
            api_type = str(style_config.get("api_type", "openai") or "openai").lower().replace("-", "_")
            endpoint_config = style_config.get(api_type)
            if not isinstance(endpoint_config, dict):
                endpoint_config = {}
                style_config[api_type] = endpoint_config

            for key in ("base_url", "api_key", "model_name", "size", "custom_prompt_add"):
                if key in style_config and key not in endpoint_config:
                    endpoint_config[key] = style_config[key]

            prefix_map = {
                "gradio": {"gradio_resolution": "resolution", "gradio_steps": "steps", "gradio_shift": "shift", "gradio_timeout": "timeout"},
                "sd_api": {
                    "sd_negative_prompt": "negative_prompt",
                    "sd_width": "width",
                    "sd_height": "height",
                    "sd_steps": "steps",
                    "sd_cfg": "cfg",
                    "sd_model_index": "model_index",
                    "sd_seed": "seed",
                },
                "novelai": {
                    "novelai_model": "model",
                    "novelai_width": "width",
                    "novelai_height": "height",
                    "novelai_steps": "steps",
                    "novelai_scale": "scale",
                    "novelai_sampler": "sampler",
                    "novelai_negative_prompt": "negative_prompt",
                    "novelai_seed": "seed",
                    "novelai_timeout": "timeout",
                },
                "newapi_nai": {
                    "newapi_nai_negative_prompt": "negative_prompt",
                    "newapi_nai_size": "size",
                    "newapi_nai_steps": "steps",
                    "newapi_nai_scale": "scale",
                    "newapi_nai_sampler": "sampler",
                    "newapi_nai_seed": "seed",
                    "newapi_nai_image_format": "image_format",
                    "newapi_nai_max_tokens": "max_tokens",
                    "newapi_nai_timeout": "timeout",
                    "newapi_nai_retry_attempts": "retry_attempts",
                    "newapi_nai_proxy_mode": "proxy_mode",
                    "newapi_nai_quality_toggle": "quality_toggle",
                    "newapi_nai_auto_smea": "auto_smea",
                    "newapi_nai_variety_boost": "variety_boost",
                    "newapi_nai_extra_params": "extra_params",
                },
            }
            for old_key, new_key in prefix_map.get(api_type, {}).items():
                if old_key in style_config and new_key not in endpoint_config:
                    value = style_config[old_key]
                    if new_key == "extra_params" and isinstance(value, str):
                        try:
                            parsed_value = json.loads(value) if value.strip() else {}
                        except json.JSONDecodeError:
                            parsed_value = {}
                        value = parsed_value if isinstance(parsed_value, dict) else {}
                    endpoint_config[new_key] = value

            for key in list(style_config.keys()):
                if key in {"enabled", "api_type", "regex_url", "newapi_nai", "openai", "gradio", "sd_api", "novelai"}:
                    continue
                style_config.pop(key, None)
        return config_data

    async def on_load(self) -> None:
        self.ctx.logger.info("MaiBot_LLM2pic 原生适配插件已加载")

    async def on_unload(self) -> None:
        try:
            from .core.services.danbooru_online_retriever import reset_online_retriever
            from .core.services.tag_retriever import reset_tag_retriever

            reset_online_retriever()
            reset_tag_retriever()
        except Exception:
            pass
        self.ctx.logger.info("MaiBot_LLM2pic 原生适配插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        del config_data
        self.ctx.logger.info("MaiBot_LLM2pic 配置更新: scope=%s version=%s", scope, version)

    @staticmethod
    def _generation_stream_key(stream_id: str) -> str:
        return stream_id or "__default__"

    def _try_acquire_generation_lock(self, plugin_config: dict[str, Any], stream_id: str) -> bool:
        guard_config = plugin_config.get("generation_guard", {})
        if not _normalize_bool(guard_config.get("pending_lock_enabled", True)):
            return True
        key = self._generation_stream_key(stream_id)
        now = time.time()
        # Expire stale locks (e.g. background task crashed without release)
        lock_timeout = float(guard_config.get("lock_timeout_seconds", 300.0) or 300.0)
        existing = self._pending_generation_streams.get(key)
        if existing is not None:
            if now - existing < lock_timeout:
                return False
            logger.warning("[LLM2pic] generation lock stale for %s (held %.0fs, timeout=%.0fs), force-releasing", key, now - existing, lock_timeout)
        # Global concurrency limit
        if len(self._pending_generation_streams) >= self._global_concurrency_limit:
            logger.info("[LLM2pic] global concurrency limit reached (%s/%s), rejecting stream %s", len(self._pending_generation_streams), self._global_concurrency_limit, key)
            return False
        self._pending_generation_streams[key] = now
        return True

    def _release_generation_lock(self, stream_id: str) -> None:
        self._pending_generation_streams.pop(self._generation_stream_key(stream_id), None)

    def _spawn_background_task(self, coro: Any) -> asyncio.Task:
        """Create a tracked asyncio task to prevent GC silent-kill."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @staticmethod
    def _last_chat_line(chat_messages: str) -> str:
        lines = [line.strip() for line in str(chat_messages or "").splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _assess_draw_guard(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        description: str,
        chat_messages: str,
        selfie_mode: bool,
    ) -> tuple[bool, str, str]:
        guard_config = plugin_config.get("generation_guard", {})
        if not _normalize_bool(guard_config.get("enabled", True)):
            return True, "guard_disabled", ""

        last_chat_line = self._last_chat_line(chat_messages)
        signal_text = f"{description}\n{last_chat_line}".lower()
        if _normalize_bool(guard_config.get("negative_intent_block_enabled", True)):
            if any(keyword in signal_text for keyword in self._NEGATIVE_DRAW_INTENT_KEYWORDS):
                return False, "blocked", "检测到用户明确表示不需要生成图片"

        explicit_request = bool(selfie_mode or self._EXPLICIT_DRAW_INTENT_RE.search(signal_text))
        if not explicit_request and any(keyword in signal_text for keyword in self._EXPLICIT_DRAW_INTENT_KEYWORDS):
            logger.info(
                "[DrawPicture] 疑似非明确出图触发，按主动出图保护处理: description=%s last_chat=%s",
                description[:80],
                last_chat_line[:80],
            )
        if not explicit_request:
            return False, "blocked", "未检测到用户明确要求生成图片"
        category = "explicit" if explicit_request else "proactive"
        interval_key = (
            "explicit_request_min_interval_seconds"
            if explicit_request
            else "proactive_min_interval_seconds"
        )
        min_interval = int(guard_config.get(interval_key, 30 if explicit_request else 240) or 0)
        if min_interval <= 0:
            return True, category, ""

        key = self._generation_stream_key(stream_id)
        last_allowed_at = self._last_guarded_draw_at.get(key, 0)
        elapsed = time.time() - last_allowed_at
        if elapsed < min_interval:
            return False, category, "同一聊天流刚刚生成过图片，已跳过本次自动出图"
        return True, category, ""

    def _mark_draw_guard_allowed(self, stream_id: str) -> None:
        self._last_guarded_draw_at[self._generation_stream_key(stream_id)] = time.time()

    @Tool(
        "draw_picture",
        description=DrawPictureToolMetadata.tool_description,
        detailed_description=DrawPictureToolMetadata.tool_detailed_description,
        parameters=_DRAW_PICTURE_TOOL_PARAMETERS,
    )
    async def handle_draw_picture(
        self,
        description: str = "",
        selfie_mode: bool = False,
        nsfw_allowed: bool = False,
        use_reference_image: bool = False,
        reference_mode: str = "i2i",
        stream_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        plugin_config = self.get_plugin_config_data()
        if not _normalize_bool(self._config_get("components.enable_image_generation", True)):
            return {"success": False, "error": "图片生成功能未启用"}

        if not self._try_acquire_generation_lock(plugin_config, stream_id):
            return {"success": False, "error": "同一聊天流已有图片生成任务正在进行"}

        task_started = False
        try:
            # 只做轻量检查（guard + 参数校验），快速返回
            proxy = _ToolRuntimeProxy(
                self,
                plugin_config=plugin_config,
                stream_id=stream_id,
                tool_args={
                    "description": description,
                    "selfie_mode": selfie_mode,
                    "nsfw_allowed": nsfw_allowed,
                    "use_reference_image": use_reference_image,
                },
                session_message=kwargs.get("message"),
            )

            original_description = str(proxy.tool_args.get("description", "") or "").strip()
            selfie_mode_bool = _normalize_bool(proxy.tool_args.get("selfie_mode", False))
            nsfw_allowed_bool = _normalize_bool(proxy.tool_args.get("nsfw_allowed", False))
            if not original_description:
                if not selfie_mode_bool:
                    return {"success": False, "error": "调用 draw_picture 时必须在 description 中写明用户要画的主体"}
                original_description = "画东雪莲/Azuma Seren 的自拍或当前状态"

            chat_messages_str = await proxy._get_recent_chat_messages()
            allowed, guard_category, guard_error = self._assess_draw_guard(
                plugin_config=plugin_config,
                stream_id=stream_id,
                description=original_description,
                chat_messages=chat_messages_str,
                selfie_mode=selfie_mode_bool,
            )
            if not allowed:
                logger.info("[DrawPicture] 出图保护拦截: category=%s error=%s", guard_category, guard_error)
                return {"success": False, "error": guard_error}

            # WD14 反推 + LLM prompt 生成 + NAI 出图全部丢后台
            self._spawn_background_task(
                self._background_draw_picture(
                    plugin_config=plugin_config,
                    stream_id=stream_id,
                    proxy=proxy,
                    original_description=original_description,
                    chat_messages_str=chat_messages_str,
                    selfie_mode_bool=selfie_mode_bool,
                    nsfw_allowed_bool=nsfw_allowed_bool,
                    use_reference_image=use_reference_image,
                    reference_mode=reference_mode,
                )
            )
            task_started = True
        except Exception as exc:
            logger.error("[DrawPicture] 启动生图任务失败: %s", exc, exc_info=True)
            return {"success": False, "error": f"启动生图任务失败: {str(exc)[:80]}"}
        finally:
            if not task_started:
                self._release_generation_lock(stream_id)
        await self._ctx_send_text("正在生成图片，请稍等...", stream_id)
        return {"success": True, "content": "已开始生成图片，完成后会直接发送。"}

    async def _background_draw_picture(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        proxy: _ToolRuntimeProxy,
        original_description: str,
        chat_messages_str: str,
        selfie_mode_bool: bool,
        nsfw_allowed_bool: bool,
        use_reference_image: bool,
        reference_mode: str = "i2i",
    ) -> None:
        """后台异步完成画图（P2 重构：走 pipeline）。"""
        try:
            ref_mode = reference_mode if _normalize_bool(use_reference_image) else ""
            ctx = DrawPipelineContext(
                source="draw_picture",
                user_request=original_description,
                chat_messages=chat_messages_str,
                persona=await proxy._get_persona(),
                selfie_mode=selfie_mode_bool,
                nsfw_allowed=nsfw_allowed_bool,
                ref_mode=ref_mode,
                custom_system_prompt=str(proxy.get_config("llm.system_prompt", "") or ""),
                config=plugin_config,
                stream_id=stream_id,
                proxy=proxy,
                plugin=self,
            )
            await run_draw_pipeline(ctx)
        except Exception as exc:
            logger.error("[LLM2PicPlugin] _background_draw_picture 异常: %s", exc, exc_info=True)
            try:
                await self._ctx_send_text(f"画图出错了: {str(exc)[:80]}", stream_id)
            except Exception:
                pass
        finally:
            self._release_generation_lock(stream_id)
    async def _background_generate_and_send(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        generated_prompt: str,
        llm_style: Optional[str],
        global_prompt: Optional[str],
        characters: Optional[list[dict[str, Any]]],
        aspect: Optional[str],
        selfie_mode_bool: bool,
        input_image_base64: Optional[str],
        proxy: _ToolRuntimeProxy,
    ) -> None:
        """后台异步完成图片生成和发送。"""
        try:
            await self._background_generate_and_send_inner(
                plugin_config=plugin_config,
                stream_id=stream_id,
                generated_prompt=generated_prompt,
                llm_style=llm_style,
                global_prompt=global_prompt,
                characters=characters,
                aspect=aspect,
                selfie_mode_bool=selfie_mode_bool,
                input_image_base64=input_image_base64,
                proxy=proxy,
            )
        except Exception as exc:
            logger.error("[LLM2PicPlugin] 后台任务未捕获异常: %s", exc, exc_info=True)
            try:
                await self._ctx_send_text(f"画图出错了: {str(exc)[:80]}", stream_id)
            except Exception:
                pass
        finally:
            self._release_generation_lock(stream_id)

    async def _background_generate_and_send_inner(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        generated_prompt: str,
        llm_style: Optional[str],
        global_prompt: Optional[str],
        characters: Optional[list[dict[str, Any]]],
        aspect: Optional[str],
        selfie_mode_bool: bool,
        input_image_base64: Optional[str],
        proxy: _ToolRuntimeProxy,
    ) -> None:
        """后台异步完成图片生成和发送（内部实现）。"""
        await self._run_generation_and_send(
            plugin_config=plugin_config,
            stream_id=stream_id,
            proxy=proxy,
            request=ImageGenerationRequest(
                prompt=generated_prompt,
                selfie_mode=selfie_mode_bool,
                llm_style=llm_style,
                global_prompt=global_prompt,
                characters=characters,
                aspect=aspect,
                input_image_base64=input_image_base64,
            ),
            failure_prefix="画图失败了",
            update_draw_guard=True,
        )

    async def _run_generation_and_send(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        proxy: _ToolRuntimeProxy | _CommandRuntimeProxy,
        request: ImageGenerationRequest,
        failure_prefix: str,
        update_draw_guard: bool = False,
    ) -> None:
        """统一执行生图请求并发送结果。"""
        generation_result = await generate_image(
            plugin_config=plugin_config,
            client=proxy,
            request=request,
        )
        if generation_result.success:
            success, message = await proxy._handle_image_result(generation_result.result, prompt=request.prompt)
            if not success:
                await self._ctx_send_text(message, stream_id)
            elif update_draw_guard:
                self._mark_draw_guard_allowed(stream_id)
        else:
            await self._ctx_send_text(f"{failure_prefix}: {str(generation_result.result)[:80]}", stream_id)

    # ===== edit_picture Tool =====

    @Tool(
        "edit_picture",
        description=_EDIT_PICTURE_TOOL_DESCRIPTION,
        detailed_description="""此工具让你能够编辑图片或生成写实风格图片。仅在用户有非常明确的编辑/修改意图时才触发。

【核心判断原则】
用户必须同时满足以下条件之一才能触发：
A) 图片编辑：用户发送/引用了图片 + 使用了明确的编辑动词（改、P、换、加、删、去掉、重绘、编辑、修改、变成、转换）
B) 写实文生图：用户明确说出"写实"/"真实"/"照片级"/"realistic" 等关键词

【触发条件 - 图片编辑（必须有明确编辑动词）】
1. 用户发送图片 + 明确说"帮我改/P一下/换个背景/加个XX/去掉XX/重绘"
2. 用户引用图片 + 明确说"把这张图改成.../编辑一下/P帅一点"
3. 关键：仅仅发送图片+随便说话 ≠ 编辑需求！必须有编辑动词！

【触发条件 - 写实文生图（必须有关键词）】
1. 用户明确说"写实/真实/照片级/realistic"风格
2. 用户的描述明确要求非动漫的真实感图片

【典型触发语句】
- [图片] "帮我P帅一点" ✓
- [图片] "把背景换成星空" ✓
- [引用图片] "给这张加个圣诞帽" ✓
- "画一张写实风格的猫咪" ✓

【绝对禁止触发的情况】
- 用户只是发了图片，没有要求修改（如分享照片、表情包）
- 用户发图片+说了无关的话（如"哈哈哈"、"你看这个"、"好看吗"）
- 用户在讨论/评价图片但没有要求改动
- 用户要求画二次元/动漫风格的图（应使用 draw_picture）
- 没有出现任何编辑动词，也没有提到写实/真实
- 你自己觉得"可能需要编辑"但用户没明确说 → 不触发
- 前面聊天记录中你已经发过图片时，禁止再次触发
""",
        parameters=_EDIT_PICTURE_TOOL_PARAMETERS,
    )
    async def handle_edit_picture(
        self,
        description: str = "",
        stream_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        plugin_config = self.get_plugin_config_data()

        # 检查 edit 模型是否配置
        style_router = StyleRouter(plugin_config)
        if not style_router.is_style_available("edit"):
            return {"success": False, "error": "图片编辑功能未配置 edit 模型"}

        if not self._try_acquire_generation_lock(plugin_config, stream_id):
            return {"success": False, "error": "同一聊天流已有图片生成任务正在进行"}

        try:
            # 后台异步执行，快速返回避免 RPC 超时
            self._spawn_background_task(
                self._background_edit_picture(
                    plugin_config=plugin_config,
                    stream_id=stream_id,
                    description=description,
                )
            )
        except Exception as exc:
            self._release_generation_lock(stream_id)
            logger.error("[EditPicture] 启动编辑任务失败: %s", exc, exc_info=True)
            return {"success": False, "error": f"启动编辑任务失败: {str(exc)[:80]}"}
        await self._ctx_send_text("正在编辑图片，请稍等...", stream_id)
        return {"success": True, "content": "已开始编辑图片，完成后会直接发送。"}

    async def _background_edit_picture(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        description: str,
    ) -> None:
        """后台异步完成图片编辑和发送。"""
        try:
            await self._background_edit_picture_inner(
                plugin_config=plugin_config,
                stream_id=stream_id,
                description=description,
            )
        except Exception as exc:
            logger.error("[LLM2PicPlugin] edit_picture 后台任务异常: %s", exc, exc_info=True)
            try:
                await self._ctx_send_text(f"图片编辑出错了: {str(exc)[:80]}", stream_id)
            except Exception:
                pass
        finally:
            self._release_generation_lock(stream_id)

    async def _background_edit_picture_inner(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        description: str,
    ) -> None:
        """后台图片编辑的内部实现。"""
        # 1. 从最近消息中提取图片（可能为 None，此时退化为纯文生图）
        input_image_base64 = await self._ctx_extract_image_from_recent(stream_id)

        # 1.5 预处理图片：缩放到合理大小，避免 API 拒绝
        if input_image_base64:
            input_image_base64 = _resize_image_for_edit(input_image_base64)

        # 2. 获取 edit 模型配置
        style_router = StyleRouter(plugin_config)
        _, model_config, _ = style_router.route(
            selfie_mode=False,
            manual_style="edit",
            llm_style=None,
        )
        if model_config is None:
            await self._ctx_send_text("edit 模型未配置", stream_id)
            return

        # 3. 构建 prompt
        final_prompt = description.strip() or ("edit this image" if input_image_base64 else "generate an image")
        mode_label = "图生图" if input_image_base64 else "文生图"
        logger.info(f"[EditPicture] {mode_label}: prompt={final_prompt[:100]}...")

        proxy = _CommandRuntimeProxy(
            self,
            plugin_config=plugin_config,
            stream_id=stream_id,
            session_message=None,
        )
        await self._run_generation_and_send(
            plugin_config=plugin_config,
            stream_id=stream_id,
            proxy=proxy,
            request=ImageGenerationRequest(
                prompt=final_prompt,
                manual_style="edit",
                input_image_base64=input_image_base64,
            ),
            failure_prefix="图片生成失败",
        )

    @Command(
        "direct_pic",
        description=DirectPicCommand.command_description,
        pattern=DirectPicCommand.command_pattern,
    )
    async def handle_direct_pic(
        self,
        stream_id: str = "",
        matched_groups: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, Optional[str], bool]:
        plugin_config = self.get_plugin_config_data()
        if not _normalize_bool(self._config_get("components.enable_direct_pic_command", True)):
            return True, "图片命令未启用", True

        raw_prompt = (matched_groups or {}).get("prompt", "").strip()
        manual_style = (matched_groups or {}).get("style")
        nsfw_allowed = str((matched_groups or {}).get("nsfw", "") or "").strip().lower() == "nsfw"
        ref_mode = str((matched_groups or {}).get("ref", "") or "").strip().lower().replace("-", "_")

        if not raw_prompt:
            return True, "用法: /pic <prompt> | /pic i2i <prompt> | /pic char-ref <prompt> | /pic vibe <prompt> | /pic nsfw <prompt>", True

        if not self._try_acquire_generation_lock(plugin_config, stream_id):
            return True, "同一聊天流已有图片生成任务正在进行", True

        try:
            # 非空 prompt：后台异步生成
            self._spawn_background_task(
                self._background_direct_pic(
                    plugin_config=plugin_config,
                    stream_id=stream_id,
                    raw_prompt=raw_prompt,
                    manual_style=manual_style,
                    nsfw_allowed=nsfw_allowed,
                    ref_mode=ref_mode,
                    session_message=kwargs.get("message"),
                )
            )
        except Exception as exc:
            self._release_generation_lock(stream_id)
            logger.error("[DirectPic] 启动 /pic 任务失败: %s", exc, exc_info=True)
            return True, f"/pic 启动失败: {str(exc)[:80]}", True
        return True, None, True

    async def _background_direct_pic(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        raw_prompt: str,
        manual_style: Optional[str],
        nsfw_allowed: bool = False,
        ref_mode: str = "",
        session_message: Any = None,
    ) -> None:
        """后台异步处理 /pic 命令（P2 重构：走 pipeline）。"""
        try:
            proxy = _CommandRuntimeProxy(
                self,
                plugin_config=plugin_config,
                stream_id=stream_id,
                session_message=session_message,
            )
            ctx = DrawPipelineContext(
                source="direct_pic",
                user_request=raw_prompt,
                selfie_mode=False,
                nsfw_allowed=nsfw_allowed,
                manual_style=manual_style,
                ref_mode=ref_mode,
                custom_system_prompt=str(proxy.get_config("llm.system_prompt", "") or ""),
                config=plugin_config,
                stream_id=stream_id,
                proxy=proxy,
                plugin=self,
            )
            await run_draw_pipeline(ctx)
        except Exception as exc:
            logger.error("[LLM2PicPlugin] /pic 后台任务异常: %s", exc, exc_info=True)
            try:
                await self._ctx_send_text(f"/pic 出错了: {str(exc)[:80]}", stream_id)
            except Exception:
                pass
        finally:
            self._release_generation_lock(stream_id)
def create_plugin() -> LLM2PicPlugin:
    """rdev Runner 原生插件工厂。"""
    return LLM2PicPlugin()
