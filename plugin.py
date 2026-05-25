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
from maibot_sdk.types import ToolParameterInfo, ToolParamType

from src.common.logger import get_logger

from .utils import _normalize_bool, _resize_image_for_edit
from .style_router import StyleRouter
from .actions import DrawPictureToolMetadata
from .commands import DirectPicCommand
from .bridge import _RuntimeBridgeMixin, _ToolRuntimeProxy, _CommandRuntimeProxy
from .generation_service import ImageGenerationRequest, generate_image

logger = get_logger("MaiBot_LLM2pic")

_CONFIG_VERSION = "4.2.0"


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(default=_CONFIG_VERSION, description="配置版本")


class GenerationConfig(PluginConfigBase):
    """通用生成参数。"""

    __ui_label__ = "生成"
    __ui_icon__ = "wand-sparkles"
    __ui_order__ = 1

    default_style: Literal["anime", "edit"] = Field(default="anime", description="LLM 无法判断风格时使用的默认风格")
    custom_prompt_add: str = Field(default="", description="全局附加提示词；端点分组中的 custom_prompt_add 优先级更高")
    crop_enabled: bool = Field(default=False, description="是否裁切生成图")
    crop_position: Literal["top", "bottom", "left", "right"] = Field(default="bottom", description="裁切位置")
    crop_pixels: int = Field(default=40, ge=0, description="裁切像素数")


class GenerationGuardConfig(PluginConfigBase):
    """自动出图保护。"""

    __ui_label__ = "出图保护"
    __ui_icon__ = "shield"
    __ui_order__ = 3

    enabled: bool = Field(default=True, description="是否启用 Tool 自动出图保护")
    pending_lock_enabled: bool = Field(default=True, description="同一聊天流已有生图任务时拒绝新任务")
    negative_intent_block_enabled: bool = Field(default=True, description="检测到用户明确不要画图时阻止自动出图")
    explicit_request_min_interval_seconds: int = Field(default=30, ge=0, description="明确请求出图的最小间隔")
    proactive_min_interval_seconds: int = Field(default=240, ge=0, description="LLM 主动出图的最小间隔")


class RegexUrlEndpointConfig(PluginConfigBase):
    """regex_url 端点参数。"""

    base_url: str = Field(default="", description="URL 模板，$1 会替换为 URL encode 后的 prompt")
    custom_prompt_add: str = Field(default="{{{masterpiece,best quality}}},", description="该端点专用正向提示词前缀")


class OpenAIEndpointConfig(PluginConfigBase):
    """OpenAI 兼容端点参数。"""

    base_url: str = Field(default="", description="OpenAI 兼容 base_url，例如 https://api.openai.com/v1")
    api_key: str = Field(default="", description="API 密钥", json_schema_extra={"input_type": "password"})
    model_name: str = Field(default="", description="模型名称")
    size: str = Field(default="", description="图片尺寸，可留空")
    custom_prompt_add: str = Field(default="", description="该端点专用正向提示词前缀")


class GradioEndpointConfig(PluginConfigBase):
    """Gradio 端点参数。"""

    base_url: str = Field(default="https://tongyi-mai-z-image-turbo.hf.space", description="Gradio Space 地址")
    custom_prompt_add: str = Field(default="", description="该端点专用正向提示词前缀")
    resolution: str = Field(default="1024x1024 ( 1:1 )", description="图片分辨率")
    steps: int = Field(default=8, ge=1, le=50, description="推理步数")
    shift: int = Field(default=3, description="时间偏移参数")
    timeout: int = Field(default=120, ge=1, description="轮询超时时间（秒）")


class SdApiEndpointConfig(PluginConfigBase):
    """Stable Diffusion API 端点参数。"""

    base_url: str = Field(default="", description="SD API 地址")
    api_key: str = Field(default="", description="API 密钥", json_schema_extra={"input_type": "password"})
    custom_prompt_add: str = Field(default="", description="该端点专用正向提示词前缀")
    negative_prompt: str = Field(default="", description="负向提示词")
    width: int = Field(default=832, ge=1, description="图片宽度")
    height: int = Field(default=1216, ge=1, description="图片高度")
    steps: int = Field(default=28, ge=1, description="推理步数")
    cfg: float = Field(default=7, description="CFG scale")
    model_index: int = Field(default=9, ge=0, description="模型索引")
    seed: int = Field(default=-1, description="随机种子，-1 表示随机")


class NovelAIEndpointConfig(PluginConfigBase):
    """NovelAI 官方 API 端点参数。"""

    api_key: str = Field(default="", description="NovelAI Bearer Token", json_schema_extra={"input_type": "password"})
    custom_prompt_add: str = Field(default="", description="该端点专用正向提示词前缀")
    model: str = Field(default="nai-diffusion-4-5-full", description="NovelAI 模型")
    width: int = Field(default=832, ge=1, description="图片宽度")
    height: int = Field(default=1216, ge=1, description="图片高度")
    steps: int = Field(default=28, ge=1, le=50, description="推理步数")
    scale: float = Field(default=5.0, description="提示词引导强度")
    sampler: str = Field(default="k_euler", description="采样器")
    negative_prompt: str = Field(default="", description="负向提示词")
    seed: int = Field(default=-1, description="随机种子，-1 表示随机")
    timeout: int = Field(default=120, ge=1, description="请求超时时间（秒）")


class NewApiNaiEndpointConfig(PluginConfigBase):
    """NewAPI NAI 端点参数。"""

    base_url: str = Field(default="", description="NewAPI 地址，例如 https://api.tuercha.com/v1")
    api_key: str = Field(default="", description="NewAPI 密钥", json_schema_extra={"input_type": "password"})
    model_name: str = Field(default="nai-diffusion-4-5-full", description="模型 ID")
    custom_prompt_add: str = Field(default="{{{masterpiece,best quality}}},", description="画师串、质量词等正向提示词前缀")
    negative_prompt: str = Field(default="lowres, bad anatomy, bad hands, text, watermark", description="负向提示词")
    size: str = Field(default="portrait", description="portrait、landscape、square、832x1216 或数组尺寸")
    steps: int = Field(default=23, ge=1, le=28, description="推理步数，NewAPI NAI 最大 28")
    scale: float = Field(default=5, description="提示词引导强度")
    sampler: str = Field(default="k_euler_ancestral", description="采样器")
    seed: int = Field(default=-1, description="随机种子，-1 表示随机")
    image_format: Literal["png", "webp"] = Field(default="png", description="返回图片格式")
    max_tokens: int = Field(default=100000, ge=1, description="最大预算 tokens")
    timeout: int = Field(default=180, ge=1, description="请求超时时间（秒）")
    retry_attempts: int = Field(default=3, ge=1, le=5, description="429/5xx 或临时网络错误的重试次数")
    proxy_mode: Literal["auto", "inherit", "direct"] = Field(default="auto", description="代理模式：auto、inherit 或 direct")
    quality_toggle: bool = Field(default=True, description="是否透传 NovelAI 质量增强参数 qualityToggle")
    auto_smea: bool = Field(default=False, description="是否透传 NovelAI autoSmea 参数")
    variety_boost: bool = Field(default=False, description="是否透传 NovelAI variety_boost 参数")
    extra_params: dict[str, Any] = Field(default_factory=dict, description="额外透传给 NewAPI NAI 内层 JSON 的参数")


class AnimeConfig(PluginConfigBase):
    """Anime 文生图配置。"""

    __ui_label__ = "Anime"
    __ui_icon__ = "image"
    __ui_order__ = 2

    enabled: bool = Field(default=True, description="是否启用 anime 风格")
    api_type: Literal["regex_url", "newapi_nai", "openai", "gradio", "sd_api", "novelai"] = Field(
        default="gradio",
        description="当前启用端点类型",
    )
    regex_url: RegexUrlEndpointConfig = Field(default_factory=RegexUrlEndpointConfig, description="regex_url 端点参数")
    newapi_nai: NewApiNaiEndpointConfig = Field(default_factory=NewApiNaiEndpointConfig, description="NewAPI NAI 端点参数")
    openai: OpenAIEndpointConfig = Field(default_factory=OpenAIEndpointConfig, description="OpenAI 兼容端点参数")
    gradio: GradioEndpointConfig = Field(default_factory=GradioEndpointConfig, description="Gradio 端点参数")
    sd_api: SdApiEndpointConfig = Field(default_factory=SdApiEndpointConfig, description="SD API 端点参数")
    novelai: NovelAIEndpointConfig = Field(default_factory=NovelAIEndpointConfig, description="NovelAI 端点参数")


class EditConfig(PluginConfigBase):
    """图片编辑配置。"""

    __ui_label__ = "Edit"
    __ui_icon__ = "image-plus"
    __ui_order__ = 4

    enabled: bool = Field(default=False, description="是否启用 edit 风格")
    api_type: Literal["openai"] = Field(default="openai", description="图片编辑目前使用 OpenAI 兼容端点")
    openai: OpenAIEndpointConfig = Field(default_factory=OpenAIEndpointConfig, description="OpenAI 兼容端点参数")


class LlmConfig(PluginConfigBase):
    """提示词 LLM 配置。"""

    __ui_label__ = "LLM"
    __ui_icon__ = "brain"
    __ui_order__ = 5

    model_name: str = Field(default="", description="用于生成提示词的 LLM 模型名，留空使用系统默认")
    context_message_limit: int = Field(default=20, ge=1, le=100, description="聊天记录条数上限")
    context_time_minutes: int = Field(default=30, ge=1, le=1440, description="聊天记录时间范围（分钟）")
    prompt_mode: Literal["legacy", "danbooru"] = Field(default="danbooru", description="提示词生成模式")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Danbooru 提示词生成温度")
    danbooru_sfw_mode: bool = Field(default=True, description="Danbooru 模式默认是否启用 SFW 安全模板")
    enforce_tag_order: bool = Field(default=True, description="Danbooru 模式是否启用轻量 tag 排序")
    selfie_appearance_policy: Literal["auto", "never", "keep"] = Field(default="auto", description="自拍外貌 tag 过滤策略")
    system_prompt: str = Field(default="", description="自定义系统提示词", json_schema_extra={"ui_type": "textarea", "rows": 8})


class TagRetrieverConfig(PluginConfigBase):
    """Danbooru tag 检索增强配置。"""

    __ui_label__ = "Tag 检索"
    __ui_icon__ = "tags"
    __ui_order__ = 6

    enabled: bool = Field(default=True, description="是否启用 Danbooru 候选 tag 检索增强")
    mode: Literal["online", "local"] = Field(default="online", description="检索模式")
    api_url: str = Field(default="https://sakizuki-danboorusearch.hf.space/api", description="在线检索 API 地址")
    timeout: float = Field(default=90.0, ge=1.0, le=300.0, description="在线检索超时时间")
    search_limit: int = Field(default=30, ge=1, le=500, description="在线语义检索返回上限")
    search_top_k: int = Field(default=5, ge=1, le=50, description="在线每个分词段召回数")
    related_limit: int = Field(default=20, ge=0, le=200, description="在线共现推荐上限")
    related_seed_count: int = Field(default=8, ge=1, le=50, description="在线共现推荐种子 tag 数")
    show_nsfw: bool = Field(default=False, description="检索结果是否包含 NSFW tag")
    popularity_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="在线检索热度排序权重")
    fallback_local: bool = Field(default=True, description="在线检索无结果或失败时是否回退本地检索")
    top_k: int = Field(default=20, ge=1, le=200, description="本地检索返回上限")
    min_score: float = Field(default=0.3, ge=0.0, le=1.0, description="本地检索最低相似度")


class ComponentsConfig(PluginConfigBase):
    """组件启用配置。"""

    __ui_label__ = "组件"
    __ui_icon__ = "toggle-right"
    __ui_order__ = 6

    enable_image_generation: bool = Field(default=True, description="是否启用图片生成 Tool")
    enable_direct_pic_command: bool = Field(default=True, description="是否启用 /pic 指令")


class LLM2PicPluginConfig(PluginConfigBase):
    """LLM2PIC 插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    generation_guard: GenerationGuardConfig = Field(default_factory=GenerationGuardConfig)
    anime: AnimeConfig = Field(default_factory=AnimeConfig)
    edit: EditConfig = Field(default_factory=EditConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    tag_retriever: TagRetrieverConfig = Field(default_factory=TagRetrieverConfig)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)


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

    _pending_generation_streams: set[str] = set()
    _last_guarded_draw_at: dict[str, float] = {}
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
        if key in self._pending_generation_streams:
            return False
        self._pending_generation_streams.add(key)
        return True

    def _release_generation_lock(self, stream_id: str) -> None:
        self._pending_generation_streams.discard(self._generation_stream_key(stream_id))

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
            # 先生成 prompt（快速，在 RPC 超时内完成）
            proxy = _ToolRuntimeProxy(
                self,
                plugin_config=plugin_config,
                stream_id=stream_id,
                tool_args={
                    "description": description,
                    "selfie_mode": selfie_mode,
                    "nsfw_allowed": nsfw_allowed,
                },
                session_message=kwargs.get("message"),
            )

            # 获取聊天记录、生成提示词（都在30秒内）
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

            persona = await proxy._get_persona()
            custom_system_prompt = str(proxy.get_config("llm.system_prompt", "") or "")

            prompt_result = await proxy._generate_prompt_with_style(
                user_request=original_description or "根据聊天内容生成一张合适的图片",
                chat_messages=chat_messages_str,
                persona=persona,
                selfie_mode=selfie_mode_bool,
                nsfw_allowed=nsfw_allowed_bool,
                custom_system_prompt=custom_system_prompt,
            )
            if not prompt_result.success:
                return {"success": False, "error": f"提示词生成失败: {prompt_result.error}"}

            self._mark_draw_guard_allowed(stream_id)

            # 快速返回，后台异步请求生图 API
            asyncio.create_task(
                self._background_generate_and_send(
                    plugin_config=plugin_config,
                    stream_id=stream_id,
                    generated_prompt=prompt_result.prompt,
                    llm_style=prompt_result.style,
                    global_prompt=prompt_result.global_prompt,
                    characters=prompt_result.characters,
                    aspect=prompt_result.aspect,
                    selfie_mode_bool=selfie_mode_bool,
                    input_image_base64=None,
                    proxy=proxy,
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
        )

    async def _run_generation_and_send(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        proxy: _ToolRuntimeProxy | _CommandRuntimeProxy,
        request: ImageGenerationRequest,
        failure_prefix: str,
    ) -> None:
        """统一执行生图请求并发送结果。"""
        generation_result = await generate_image(
            plugin_config=plugin_config,
            client=proxy,
            request=request,
        )
        if generation_result.success:
            success, message = await proxy._handle_image_result(generation_result.result)
            if not success:
                await self._ctx_send_text(message, stream_id)
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
            asyncio.create_task(
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

        if not raw_prompt:
            return True, "用法: /pic <prompt> | /pic anime <prompt> | /pic edit <prompt>", True

        if not self._try_acquire_generation_lock(plugin_config, stream_id):
            return True, "同一聊天流已有图片生成任务正在进行", True

        try:
            # 非空 prompt：后台异步生成
            asyncio.create_task(
                self._background_direct_pic(
                    plugin_config=plugin_config,
                    stream_id=stream_id,
                    raw_prompt=raw_prompt,
                    manual_style=manual_style,
                    nsfw_allowed=nsfw_allowed,
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
        session_message: Any = None,
    ) -> None:
        """后台异步处理 /pic 命令。"""
        try:
            proxy = _CommandRuntimeProxy(
                self,
                plugin_config=plugin_config,
                stream_id=stream_id,
                session_message=session_message,
            )

            # 尝试提取输入图片
            input_image_base64 = await proxy._extract_input_image()
            characters = None
            global_prompt = None
            aspect = None
            if input_image_base64:
                generated_prompt = raw_prompt
            else:
                prompt_result = await proxy._generate_prompt_with_style(
                    user_request=raw_prompt,
                    chat_messages="",
                    persona=await proxy._get_persona(),
                    selfie_mode=False,
                    nsfw_allowed=nsfw_allowed,
                    custom_system_prompt=str(proxy.get_config("llm.system_prompt", "") or ""),
                )
                if not prompt_result.success:
                    await self._ctx_send_text(f"/pic 提示词生成失败: {prompt_result.error[:80]}", stream_id)
                    return
                generated_prompt = prompt_result.prompt
                global_prompt = prompt_result.global_prompt
                characters = prompt_result.characters
                aspect = prompt_result.aspect
            await self._run_generation_and_send(
                plugin_config=plugin_config,
                stream_id=stream_id,
                proxy=proxy,
                request=ImageGenerationRequest(
                    prompt=generated_prompt,
                    manual_style=manual_style,
                    input_image_base64=input_image_base64,
                    global_prompt=global_prompt,
                    characters=characters,
                    aspect=aspect,
                ),
                failure_prefix="/pic 失败了",
            )
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
