"""
MaiBot_LLM2pic - MaiBot图片生成插件

使用LLM根据聊天记录和人设生成符合需求的prompt，然后调用图片生成API
支持文生图和图生图功能
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal, Optional
import asyncio
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

_CONFIG_VERSION = "4.1.0"


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
    __ui_order__ = 3

    enabled: bool = Field(default=False, description="是否启用 edit 风格")
    api_type: Literal["openai"] = Field(default="openai", description="图片编辑目前使用 OpenAI 兼容端点")
    openai: OpenAIEndpointConfig = Field(default_factory=OpenAIEndpointConfig, description="OpenAI 兼容端点参数")


class LlmConfig(PluginConfigBase):
    """提示词 LLM 配置。"""

    __ui_label__ = "LLM"
    __ui_icon__ = "brain"
    __ui_order__ = 4

    model_name: str = Field(default="", description="用于生成提示词的 LLM 模型名，留空使用系统默认")
    context_message_limit: int = Field(default=20, ge=1, le=100, description="聊天记录条数上限")
    context_time_minutes: int = Field(default=30, ge=1, le=1440, description="聊天记录时间范围（分钟）")
    system_prompt: str = Field(default="", description="自定义系统提示词", json_schema_extra={"ui_type": "textarea", "rows": 8})


class ComponentsConfig(PluginConfigBase):
    """组件启用配置。"""

    __ui_label__ = "组件"
    __ui_icon__ = "toggle-right"
    __ui_order__ = 5

    enable_image_generation: bool = Field(default=True, description="是否启用图片生成 Tool")
    enable_direct_pic_command: bool = Field(default=True, description="是否启用 /pic 指令")


class LLM2PicPluginConfig(PluginConfigBase):
    """LLM2PIC 插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    anime: AnimeConfig = Field(default_factory=AnimeConfig)
    edit: EditConfig = Field(default_factory=EditConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)


_DRAW_PICTURE_TOOL_PARAMETERS = [
    ToolParameterInfo(
        name="description",
        param_type=ToolParamType.STRING,
        description="用户想要生成的图片描述，可以是中文或英文；留空时会根据聊天上下文生成合适图片。",
        required=False,
        default="",
    ),
    ToolParameterInfo(
        name="selfie_mode",
        param_type=ToolParamType.BOOLEAN,
        description="是否生成自拍模式图片；当用户要求自拍、想看你当前状态或环境时设为 true。",
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

    # 防重复调用冷却记录: {stream_id: last_trigger_timestamp}
    _draw_cooldowns: dict = {}
    _DRAW_COOLDOWN_SECONDS = 30  # draw_picture 同一聊天流冷却30秒

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
                },
            }
            for old_key, new_key in prefix_map.get(api_type, {}).items():
                if old_key in style_config and new_key not in endpoint_config:
                    endpoint_config[new_key] = style_config[old_key]

            for key in list(style_config.keys()):
                if key in {"enabled", "api_type", "regex_url", "newapi_nai", "openai", "gradio", "sd_api", "novelai"}:
                    continue
                style_config.pop(key, None)
        return config_data

    async def on_load(self) -> None:
        self.ctx.logger.info("MaiBot_LLM2pic 原生适配插件已加载")

    async def on_unload(self) -> None:
        self.ctx.logger.info("MaiBot_LLM2pic 原生适配插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        del config_data
        self.ctx.logger.info("MaiBot_LLM2pic 配置更新: scope=%s version=%s", scope, version)

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
        stream_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        plugin_config = self.get_plugin_config_data()
        if not _normalize_bool(self._config_get("components.enable_image_generation", True)):
            return {"success": False, "error": "图片生成功能未启用"}

        # 防重复调用：同一聊天流冷却期内拒绝
        now = time.time()
        if stream_id:
            last_trigger = self._draw_cooldowns.get(stream_id, 0)
            if now - last_trigger < self._DRAW_COOLDOWN_SECONDS:
                logger.debug(f"[DrawPicture] 冷却中，跳过重复调用: stream={stream_id}")
                return {"success": False, "error": "同一聊天流的图片生成正在冷却中"}
            self._draw_cooldowns[stream_id] = now

        # 先生成 prompt（快速，在 RPC 超时内完成）
        proxy = _ToolRuntimeProxy(
            self,
            plugin_config=plugin_config,
            stream_id=stream_id,
            tool_args={
                "description": description,
                "selfie_mode": selfie_mode,
            },
            session_message=kwargs.get("message"),
        )

        # 获取聊天记录、生成提示词（都在30秒内）
        original_description = str(proxy.tool_args.get("description", "") or "").strip()
        selfie_mode_bool = _normalize_bool(proxy.tool_args.get("selfie_mode", False))

        chat_messages_str = await proxy._get_recent_chat_messages()
        persona = await proxy._get_persona()
        custom_system_prompt = str(proxy.get_config("llm.system_prompt", "") or "")

        success, generated_prompt, llm_style = await proxy._generate_prompt_with_style(
            user_request=original_description or "根据聊天内容生成一张合适的图片",
            chat_messages=chat_messages_str,
            persona=persona,
            selfie_mode=selfie_mode_bool,
            custom_system_prompt=custom_system_prompt,
        )
        if not success:
            return {"success": False, "error": f"提示词生成失败: {generated_prompt}"}

        # 快速返回，后台异步请求生图 API
        asyncio.create_task(
            self._background_generate_and_send(
                plugin_config=plugin_config,
                stream_id=stream_id,
                generated_prompt=generated_prompt,
                llm_style=llm_style,
                selfie_mode_bool=selfie_mode_bool,
                input_image_base64=None,
                proxy=proxy,
            )
        )
        await self._ctx_send_text("正在生成图片，请稍等...", stream_id)
        return {"success": True, "content": "已开始生成图片，完成后会直接发送。"}

    async def _background_generate_and_send(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        generated_prompt: str,
        llm_style: Optional[str],
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

    async def _background_generate_and_send_inner(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        generated_prompt: str,
        llm_style: Optional[str],
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

        # 后台异步执行，快速返回避免 RPC 超时
        asyncio.create_task(
            self._background_edit_picture(
                plugin_config=plugin_config,
                stream_id=stream_id,
                description=description,
            )
        )
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

        if not raw_prompt:
            return True, "用法: /pic <prompt> | /pic anime <prompt> | /pic edit <prompt>", True

        # 非空 prompt：后台异步生成
        asyncio.create_task(
            self._background_direct_pic(
                plugin_config=plugin_config,
                stream_id=stream_id,
                raw_prompt=raw_prompt,
                manual_style=manual_style,
                session_message=kwargs.get("message"),
            )
        )
        return True, None, True

    async def _background_direct_pic(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        raw_prompt: str,
        manual_style: Optional[str],
        session_message: Any = None,
    ) -> None:
        """后台异步处理 /pic 命令。"""
        proxy = _CommandRuntimeProxy(
            self,
            plugin_config=plugin_config,
            stream_id=stream_id,
            session_message=session_message,
        )

        # 尝试提取输入图片
        input_image_base64 = await proxy._extract_input_image()
        await self._run_generation_and_send(
            plugin_config=plugin_config,
            stream_id=stream_id,
            proxy=proxy,
            request=ImageGenerationRequest(
                prompt=raw_prompt,
                manual_style=manual_style,
                input_image_base64=input_image_base64,
            ),
            failure_prefix="/pic 失败了",
        )


def create_plugin() -> LLM2PicPlugin:
    """rdev Runner 原生插件工厂。"""
    return LLM2PicPlugin()
