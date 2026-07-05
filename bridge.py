"""LLM2pic 运行时桥接层。

提供 _RuntimeBridgeMixin、_ToolRuntimeProxy 和 _CommandRuntimeProxy，
将生图逻辑桥接到 rdev 原生运行时上下文。
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional
import re
import time

from .danbooru_generator import (
    PromptGenerationResult,
    generate_danbooru_prompt,
    _validate_prompt_llm_response,
    _cleanup_llm_prompt,
)
from .utils import download_image_to_base64, _peel_envelope
from .style_router import LLMOutputParser, DEFAULT_SYSTEM_PROMPT
from .actions import DrawPictureToolMetadata
from .commands import DirectPicCommand

from src.common.logger import get_logger

logger = get_logger("MaiBot_LLM2pic")


@dataclass(frozen=True)
class _LLMTarget:
    task_name: str = "planner"
    model_name: Optional[str] = None




def _llm_response_usable_for_prompt(result: Any) -> bool:
    if not isinstance(result, dict) or not bool(result.get("success", False)):
        return False
    response_text = str(result.get("response") or "").strip()
    if not response_text:
        return False
    cleaned = _cleanup_llm_prompt(response_text)
    ok, _ = _validate_prompt_llm_response(response_text, cleaned)
    return ok

class _FallbackLLMProxy:
    """优先直连指定具体模型，失败后回退任务组。"""

    def __init__(self, runtime: "_RuntimeBridgeMixin", target: _LLMTarget) -> None:
        self._runtime = runtime
        self._target = target

    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        prompt = kwargs.get("prompt")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        image_base64 = kwargs.get("image_base64") or ""

        if self._target.model_name:
            try:
                result = await self._runtime._ctx_generate_with_direct_model(
                    prompt=prompt,
                    task_name=self._target.task_name,
                    model_name=self._target.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    image_base64=image_base64 or None,
                )
                if _llm_response_usable_for_prompt(result):
                    return result
                logger.warning(
                    "[LLM2picBridge] 指定模型 %s 生成失败，将回退任务 %s: %s",
                    self._target.model_name,
                    self._target.task_name,
                    str((result or {}).get("error") if isinstance(result, dict) else result)[:120],
                )
            except Exception as exc:
                logger.warning(
                    "[LLM2picBridge] 指定模型 %s 调用异常，将回退任务 %s: %s",
                    self._target.model_name,
                    self._target.task_name,
                    exc,
                    exc_info=True,
                )

        fallback_kwargs = dict(kwargs)
        fallback_kwargs["model"] = self._target.task_name
        had_image = bool(fallback_kwargs.pop("image_base64", None) or image_base64)
        chat_messages = fallback_kwargs.pop("chat_messages", None)
        prompt = fallback_kwargs.get("prompt")
        if (
            had_image
            and isinstance(prompt, str)
            and "## VLM 识图结果" not in prompt
            and chat_messages
        ):
            vlm_desc = _RuntimeBridgeMixin._extract_vlm_image_description(str(chat_messages))
            if vlm_desc:
                vlm_block = (
                    "\n\n## VLM 识图结果（指定模型回退，无法直接看图，由 VLM 补充）\n"
                    f"{vlm_desc}"
                )
                fallback_kwargs["prompt"] = f"{prompt}{vlm_block}"
        return await self._runtime.ctx.llm.generate(**fallback_kwargs)


class _RuntimeBridgeMixin:
    """为生图逻辑补齐 rdev 原生运行时上下文。"""

    ctx: Any

    def _config_get(self, path: str, default: Any = None) -> Any:
        current: Any = self.get_plugin_config_data()
        for part in str(path or "").split("."):
            if not isinstance(current, Mapping):
                return default
            if part not in current:
                return default
            current = current[part]
        return current

    async def _ctx_send_text(self, text: str, stream_id: str = "") -> bool:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return False
        try:
            await self.ctx.send.text(normalized_text, stream_id)
            return True
        except Exception as exc:
            logger.error("[LLM2picBridge] 发送文本失败: %s", exc, exc_info=True)
            return False

    async def _ctx_send_image(self, image_base64: str, stream_id: str = "") -> bool:
        try:
            return bool(await self.ctx.send.image(image_base64, stream_id))
        except Exception as exc:
            logger.error("[LLM2picBridge] 发送图片失败: %s", exc, exc_info=True)
            return False

    async def _ctx_get_recent_chat_messages(
        self,
        stream_id: str,
        *,
        message_limit: int,
        time_minutes: int,
    ) -> str:
        if not stream_id:
            return "（暂无聊天记录）"

        end_time = time.time()
        start_time = end_time - max(1, min(1440, time_minutes)) * 60
        try:
            messages = await self.ctx.message.get_by_time_in_chat(
                chat_id=stream_id,
                start_time=start_time,
                end_time=end_time,
                limit=max(1, min(100, message_limit)),
            )
        except Exception as exc:
            logger.error("[LLM2picBridge] 获取聊天记录失败: %s", exc, exc_info=True)
            return "（获取聊天记录失败）"

        messages = _peel_envelope(messages)
        if isinstance(messages, dict):
            messages = messages.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return "（暂无聊天记录）"

        readable_lines: list[str] = []
        for item in messages[-message_limit:]:
            if not isinstance(item, dict):
                continue
            sender = (
                item.get("display_name")
                or item.get("user_nickname")
                or item.get("nickname")
                or item.get("user_id")
                or item.get("sender_name")
                or "未知用户"
            )
            text = (
                item.get("processed_plain_text")
                or item.get("plain_text")
                or item.get("text")
                or item.get("raw_message")
                or ""
            )
            normalized_text = str(text).strip()
            if not normalized_text:
                continue
            readable_lines.append(f"{sender}: {normalized_text}")
        return "\n".join(readable_lines) if readable_lines else "（暂无聊天记录）"

    async def _ctx_get_persona(self) -> str:
        nickname = ""
        personality = ""
        visual_style = ""
        try:
            nickname = str(await self.ctx.config.get("bot.nickname", "") or "").strip()
        except Exception:
            nickname = ""
        try:
            personality = str(await self.ctx.config.get("personality.personality", "") or "").strip()
        except Exception:
            personality = ""
        try:
            visual_style = str(await self.ctx.config.get("personality.visual_style", "") or "").strip()
        except Exception:
            visual_style = ""

        persona_parts = [f"你的名字是{nickname or 'MaiBot'}。"]
        if personality:
            persona_parts.append(f"你的性格特点：{personality}")
        if visual_style:
            persona_parts.append(f"你的外貌特征：{visual_style}")
        return "\n".join(persona_parts)

    async def _ctx_get_llm_model_name(self) -> str:
        return (await self._ctx_resolve_llm_target()).task_name

    async def _ctx_resolve_llm_target(self) -> _LLMTarget:
        custom_model_name = str(self._config_get("llm.model_name", "") or "").strip()
        if not custom_model_name:
            return _LLMTarget()

        try:
            from src.llm_models.utils_model import TempMethodsLLMUtils
            from src.services import llm_service

            available_models = llm_service.get_available_models()
            if custom_model_name in available_models:
                return _LLMTarget(task_name=custom_model_name)

            TempMethodsLLMUtils.get_model_info_by_name(custom_model_name)
            logger.info("[LLM2picBridge] 将直连指定 LLM 模型 %s，失败时回退 planner", custom_model_name)
            return _LLMTarget(task_name="planner", model_name=custom_model_name)
        except Exception as exc:
            logger.warning(
                "[LLM2picBridge] 配置的 LLM 模型/任务 %s 在当前运行时不可用，将回退 planner: %s",
                custom_model_name,
                exc,
            )
            return _LLMTarget()

    async def _ctx_generate_with_direct_model(
        self,
        *,
        prompt: Any,
        task_name: str,
        model_name: str,
        temperature: Any = None,
        max_tokens: Any = None,
        image_base64: Optional[str] = None,
    ) -> dict[str, Any]:
        from src.services import llm_service

        # Check if the model supports vision
        use_image = False
        if image_base64:
            try:
                from src.llm_models.utils_model import TempMethodsLLMUtils
                model_info = TempMethodsLLMUtils.get_model_info_by_name(model_name)
                use_image = bool(getattr(model_info, "visual", False))
                if use_image:
                    logger.info("[LLM2picBridge] 模型 %s 支持视觉，直接传图给 LLM", model_name)
                else:
                    logger.info("[LLM2picBridge] 模型 %s 不支持视觉，不传图", model_name)
            except Exception as exc:
                logger.warning("[LLM2picBridge] 检查模型 %s 视觉能力失败: %s", model_name, exc)

        if use_image and image_base64:
            # Use message_factory to include image in the request
            from src.llm_models.utils_model import TempMethodsLLMUtils
            from src.llm_models.payload_content.message import MessageBuilder

            def message_factory(client) -> list:
                builder = MessageBuilder()
                builder.add_text_content(str(prompt or ""))
                builder.add_image_content(
                    image_base64=image_base64,
                    image_format="png",
                    support_formats=client.get_support_image_formats(),
                )
                return [builder.build()]

            llm_client = llm_service.LLMServiceClient(
                task_name=task_name,
                request_type=f"plugin.{getattr(self, 'plugin_id', 'chartyr.maibot-llm2pic')}",
            )
            try:
                generation_result = await llm_client.generate_response_with_message_async(
                    message_factory=message_factory,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model_name=model_name,
                )
                return generation_result.to_capability_payload() if hasattr(generation_result, "to_capability_payload") else {
                    "success": bool(generation_result.response),
                    "response": generation_result.response,
                    "reasoning": generation_result.reasoning,
                }
            except Exception as exc:
                logger.warning("[LLM2picBridge] 视觉模型调用失败，降级为纯文本: %s", exc)
                # Fallback to text-only

        result = await llm_service.generate(
            llm_service.LLMServiceRequest(
                task_name=task_name,
                request_type=f"plugin.{getattr(self, 'plugin_id', 'chartyr.maibot-llm2pic')}",
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        return result.to_capability_payload()

    async def _ctx_extract_image_from_recent(self, stream_id: str) -> Optional[str]:
        """从最近的聊天消息中提取图片（用于 edit 模式）。

        优先查找最近一条包含图片的消息，支持：
        1. 消息本身包含图片
        2. 消息引用了一条包含图片的消息（通过 message.get_by_id 获取）

        Returns:
            Optional[str]: 图片的 base64 数据，如果没有找到则返回 None
        """
        if not stream_id:
            return None

        end_time = time.time()
        start_time = end_time - 300  # 最近5分钟
        try:
            result = await self.ctx.call_capability(
                "message.get_by_time_in_chat",
                chat_id=stream_id,
                start_time=start_time,
                end_time=end_time,
                limit=10,
                include_binary_data=True,
            )
        except Exception as exc:
            logger.error("[LLM2picBridge] 获取最近消息失败: %s", exc, exc_info=True)
            return None

        messages = result if isinstance(result, list) else []
        if isinstance(result, dict):
            messages = result.get("messages", [])
        if not messages:
            return None

        # 从最新消息开始向前查找
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue

            raw_message = msg.get("raw_message", [])
            if not isinstance(raw_message, list):
                continue

            # 1. 检查消息本身是否包含图片
            image_base64 = self._extract_image_from_segments(raw_message)
            if image_base64:
                return image_base64

            # 2. 检查消息是否引用了另一条消息，尝试获取引用消息中的图片
            reply_to = msg.get("reply_to")
            if reply_to:
                reply_image = await self._ctx_get_image_by_message_id(reply_to, stream_id)
                if reply_image:
                    return reply_image

            # 也检查 raw_message 中的 reply 段
            for seg in raw_message:
                if isinstance(seg, dict) and seg.get("type") == "reply":
                    seg_data = seg.get("data", {})
                    target_msg_id = ""
                    if isinstance(seg_data, dict):
                        target_msg_id = str(seg_data.get("target_message_id", "") or "")
                    elif isinstance(seg_data, str):
                        target_msg_id = seg_data
                    if target_msg_id:
                        reply_image = await self._ctx_get_image_by_message_id(target_msg_id, stream_id)
                        if reply_image:
                            return reply_image

        return None

    async def _ctx_get_image_by_message_id(self, message_id: str, stream_id: str) -> Optional[str]:
        """通过消息 ID 获取该消息中的图片 base64 数据。"""
        if not message_id:
            return None
        try:
            result = await self.ctx.call_capability(
                "message.get_by_id",
                message_id=message_id,
                chat_id=stream_id,
                include_binary_data=True,
            )
        except Exception as exc:
            logger.debug("[LLM2picBridge] 获取引用消息失败: %s", exc)
            return None

        if not isinstance(result, dict):
            return None
        msg = result.get("message") if "message" in result else result
        if not isinstance(msg, dict):
            return None

        raw_message = msg.get("raw_message", [])
        if not isinstance(raw_message, list):
            return None

        return self._extract_image_from_segments(raw_message)

    @staticmethod
    def _extract_image_from_segments(segments: list) -> Optional[str]:
        """从消息段列表中提取第一张图片的 base64 数据。"""
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            if seg.get("type") != "image":
                continue
            # 优先使用 binary_data_base64
            b64 = seg.get("binary_data_base64", "")
            if b64:
                return b64
            # 其次尝试 data 字段（可能是 base64）
            data = seg.get("data", "")
            if isinstance(data, str) and data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                return data
            # 如果 data 是 URL，同步下载（兜底）
            if isinstance(data, str) and data.startswith("http"):
                success, result = download_image_to_base64(data)
                if success:
                    return result
        return None

    @staticmethod
    def _extract_vlm_image_description(chat_messages: str) -> str:
        """从聊天记录文本中提取 VLM 识图结果（格式 `[图片：描述内容]`）。

        MaiBot 的消息处理会把图片识别结果写入 processed_plain_text，格式为
        `[图片：描述内容]`。当写 tag 的 LLM 不支持视觉时，用这些描述作为补充。
        """
        text = str(chat_messages or "")
        if not text:
            return ""
        # 匹配 [图片：...] / [图片:...] 片段
        matches = re.findall(r"\[图片[:：]\s*([^\]]+)\]", text)
        if not matches:
            return ""
        # 合并多张图的描述，去重
        seen: set[str] = set()
        descriptions: list[str] = []
        for desc in matches:
            normalized = desc.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                descriptions.append(normalized)
        return "\n".join(descriptions) if descriptions else ""

    async def _ctx_generate_prompt_with_style(
        self,
        *,
        user_request: str,
        chat_messages: str,
        persona: str,
        selfie_mode: bool,
        nsfw_allowed: bool = False,
        custom_system_prompt: str = "",
        reference_tags: str = "",
        reference_image_base64: str = "",
        vlm_description: str = "",
    ) -> PromptGenerationResult:
        # 当有参考图但写 tag 的模型不支持视觉时，把 VLM 识图结果拼到 reference_tags
        effective_reference_tags = reference_tags
        if reference_image_base64 and not vlm_description:
            vlm_desc = self._extract_vlm_image_description(chat_messages)
            if vlm_desc:
                vlm_description = vlm_desc
        if vlm_description:
            vlm_block = f"\n\n## VLM 识图结果（写 tag 的模型不支持视觉，由 VLM 识别补充）\n{vlm_description}"
            effective_reference_tags = f"{effective_reference_tags}{vlm_block}".strip()

        if str(self._config_get("llm.prompt_mode", "danbooru") or "danbooru").strip().lower() == "danbooru":
            llm_target = await self._ctx_resolve_llm_target()
            return await generate_danbooru_prompt(
                config=self.get_plugin_config_data(),
                llm=_FallbackLLMProxy(self, llm_target),
                model=llm_target.task_name,
                user_request=user_request,
                chat_messages=chat_messages,
                persona=persona,
                selfie_mode=selfie_mode,
                nsfw_allowed=nsfw_allowed,
                custom_system_prompt=custom_system_prompt,
                reference_tags=effective_reference_tags,
                reference_image_base64=reference_image_base64,
            )

        base_prompt = custom_system_prompt.strip() if custom_system_prompt else DEFAULT_SYSTEM_PROMPT
        system_prompt = base_prompt.replace("{persona}", persona)

        mode_hint = ""
        if selfie_mode:
            mode_hint = "\n【自拍模式】请以你的角色身份生成一张自拍照的提示词，风格应为 anime。"

        user_prompt = f"""## 最近的聊天记录
{chat_messages}

## 用户的绘图请求
{user_request}
{mode_hint}

## 优先级与上下文使用规则
1. 用户的绘图请求是最高优先级，提示词主体必须直接来自“用户的绘图请求”。
2. 最近聊天记录只能用于补充自然场景、氛围、情绪或消歧，不能改变、替换或扩写成另一个主体。
3. 只有当用户明确要求画东雪莲、你、自拍或你的当前状态时，才允许注入东雪莲/Azuma Seren/角色专属标签；否则禁止加入东雪莲相关角色标签或外貌。
4. 如果用户请求和聊天记录冲突，以用户请求为准。

如果用户没有明确指定场景，请主动补充一个具体、自然、符合聊天语境的背景/地点/时间/光线；如果用户已经指定场景，请不要替换。
请根据以上信息，生成适合的图片提示词和风格判断。必须以 JSON 格式输出。"""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        llm_target = await self._ctx_resolve_llm_target()
        try:
            result = await _FallbackLLMProxy(self, llm_target).generate(
                prompt=full_prompt,
                model=llm_target.task_name,
                temperature=0.7,
            )
        except Exception as exc:
            logger.error("[LLM2picBridge] ctx.llm.generate 失败: %s", exc, exc_info=True)
            return PromptGenerationResult(False, error=str(exc))

        result = _peel_envelope(result)
        if not isinstance(result, dict):
            return PromptGenerationResult(False, error=f"LLM 返回非 dict: {type(result).__name__}")

        success = bool(result.get("success", False))
        response_text = str(result.get("response") or "").strip()
        if not success:
            return PromptGenerationResult(False, error=str(result.get("error") or "LLM生成失败"))
        if not response_text:
            return PromptGenerationResult(False, error="LLM返回空响应")

        prompt, style = LLMOutputParser.parse(response_text)
        if prompt:
            return PromptGenerationResult(True, prompt=prompt, style=style)
        return PromptGenerationResult(True, prompt=response_text, style=style)


class _ToolRuntimeProxy(DrawPictureToolMetadata):
    """承接 Tool 元数据与图片客户端能力，并映射到 rdev ctx。"""

    def __init__(
        self,
        runtime: _RuntimeBridgeMixin,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        tool_args: dict[str, Any],
        session_message: Any = None,
    ) -> None:
        self._runtime = runtime
        self.plugin_config = plugin_config
        self._stream_id = stream_id
        self.tool_args = tool_args
        self.log_prefix = "[DrawPictureTool]"
        self.message = session_message
        self.chat_id = stream_id

    def get_config(self, path: str, default: Any = None) -> Any:
        return self._runtime._config_get(path, default)

    async def send_text(self, text: str) -> bool:
        return await self._runtime._ctx_send_text(text, self._stream_id)

    async def send_image(self, image_base64: str) -> bool:
        return await self._runtime._ctx_send_image(image_base64, self._stream_id)

    async def _get_persona(self) -> str:
        return await self._runtime._ctx_get_persona()

    async def _get_recent_chat_messages(self) -> str:
        message_limit = int(self.get_config("llm.context_message_limit", 20) or 20)
        time_minutes = int(self.get_config("llm.context_time_minutes", 30) or 30)
        return await self._runtime._ctx_get_recent_chat_messages(
            self._stream_id,
            message_limit=message_limit,
            time_minutes=time_minutes,
        )

    async def _generate_prompt_with_style(
        self,
        *,
        user_request: str,
        chat_messages: str,
        persona: str,
        selfie_mode: bool,
        nsfw_allowed: bool,
        custom_system_prompt: str,
        reference_tags: str = "",
        reference_image_base64: str = "",
        vlm_description: str = "",
    ) -> PromptGenerationResult:
        return await self._runtime._ctx_generate_prompt_with_style(
            user_request=user_request,
            chat_messages=chat_messages,
            persona=persona,
            selfie_mode=selfie_mode,
            nsfw_allowed=nsfw_allowed,
            custom_system_prompt=custom_system_prompt,
            reference_tags=reference_tags,
            reference_image_base64=reference_image_base64,
            vlm_description=vlm_description,
        )


class _CommandRuntimeProxy(DirectPicCommand):
    """承接命令元数据与图片客户端能力，并映射到 rdev ctx。"""

    def __init__(
        self,
        runtime: _RuntimeBridgeMixin,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        session_message: Any = None,
    ) -> None:
        self._runtime = runtime
        self.plugin_config = plugin_config
        self._stream_id = stream_id
        self.log_prefix = "[DirectPic]"
        self.message = session_message
        self.chat_id = stream_id

    def get_config(self, path: str, default: Any = None) -> Any:
        return self._runtime._config_get(path, default)

    async def send_text(self, text: str) -> bool:
        return await self._runtime._ctx_send_text(text, self._stream_id)

    async def send_image(self, image_base64: str) -> bool:
        return await self._runtime._ctx_send_image(image_base64, self._stream_id)

    async def _get_persona(self) -> str:
        return await self._runtime._ctx_get_persona()

    async def _generate_prompt_with_style(
        self,
        *,
        user_request: str,
        chat_messages: str,
        persona: str,
        selfie_mode: bool,
        nsfw_allowed: bool,
        custom_system_prompt: str,
        reference_tags: str = "",
        reference_image_base64: str = "",
        vlm_description: str = "",
    ) -> PromptGenerationResult:
        return await self._runtime._ctx_generate_prompt_with_style(
            user_request=user_request,
            chat_messages=chat_messages,
            persona=persona,
            selfie_mode=selfie_mode,
            nsfw_allowed=nsfw_allowed,
            custom_system_prompt=custom_system_prompt,
            reference_tags=reference_tags,
            reference_image_base64=reference_image_base64,
            vlm_description=vlm_description,
        )
