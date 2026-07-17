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
from .image_clients import ImageClientMixin

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
                    chat_messages=kwargs.get("chat_messages"),
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
        chat_messages: Optional[str] = None,
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
            # Host LLMServiceClient 正确视觉入口：
            # - generate_response_with_messages(message_factory, options=LLMGenerationOptions)
            # - 或 generate_response_for_image(prompt, image_base64, image_format, options)
            # 旧代码误调 generate_response_with_message_async（不存在）→ 必炸降级纯文本
            from src.common.data_models.llm_service_data_models import LLMGenerationOptions
            from src.llm_models.payload_content.message import MessageBuilder

            raw_b64 = str(image_base64 or "")
            image_format = "jpeg"
            if raw_b64.startswith("data:"):
                # data:image/png;base64,xxxx
                header, _, payload = raw_b64.partition(",")
                raw_b64 = payload or raw_b64
                if "image/png" in header:
                    image_format = "png"
                elif "image/webp" in header:
                    image_format = "webp"
                elif "image/gif" in header:
                    image_format = "gif"
                else:
                    image_format = "jpeg"
            elif raw_b64.startswith("iVBORw"):
                image_format = "png"

            def message_factory(client) -> list:
                builder = MessageBuilder()
                builder.add_text_content(str(prompt or ""))
                builder.add_image_content(
                    image_base64=raw_b64,
                    image_format=image_format,
                    support_formats=client.get_support_image_formats(),
                )
                return [builder.build()]

            llm_client = llm_service.LLMServiceClient(
                task_name=task_name,
                request_type=f"plugin.{getattr(self, 'plugin_id', 'chartyr.maibot-llm2pic')}",
            )
            try:
                options = LLMGenerationOptions(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model_name=model_name,
                )
                generation_result = await llm_client.generate_response_with_messages(
                    message_factory=message_factory,
                    options=options,
                )
                logger.info(
                    "[LLM2picBridge] 视觉直连成功: model=%s resp_len=%s",
                    getattr(generation_result, "model_name", model_name),
                    len(str(getattr(generation_result, "response", "") or "")),
                )
                return generation_result.to_capability_payload() if hasattr(generation_result, "to_capability_payload") else {
                    "success": bool(generation_result.response),
                    "response": generation_result.response,
                    "reasoning": generation_result.reasoning,
                }
            except Exception as exc:
                logger.warning("[LLM2picBridge] 视觉模型调用失败，降级为纯文本: %s", exc)
                # Fallback to text-only — inject VLM description if available
                if chat_messages and isinstance(prompt, str):
                    vlm_desc = self._extract_vlm_image_description(str(chat_messages))
                    if vlm_desc and "## VLM 识图结果" not in str(prompt):
                        prompt = f"{prompt}\n\n## VLM 识图结果（视觉调用失败，由 VLM 识别补充）\n{vlm_desc}"

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

    def _ctx_collect_reply_targets(self, session_message: Any) -> list[str]:
        """收集当前命令消息上的显式引用 id（reply_to 字段 + reply 段）。"""
        targets: list[str] = []

        def _add(value: Any) -> None:
            text = str(value or "").strip()
            if text and text not in targets:
                targets.append(text)

        if not session_message:
            return targets

        if isinstance(session_message, dict):
            _add(session_message.get("reply_to"))
            raw = session_message.get("raw_message", [])
            if isinstance(raw, list):
                for seg in raw:
                    if not isinstance(seg, dict) or seg.get("type") != "reply":
                        continue
                    seg_data = seg.get("data", {})
                    if isinstance(seg_data, dict):
                        _add(
                            seg_data.get("target_message_id")
                            or seg_data.get("id")
                            or seg_data.get("message_id")
                        )
                    elif isinstance(seg_data, str):
                        _add(seg_data)
            return targets

        _add(getattr(session_message, "reply_to", None))
        raw_obj = getattr(session_message, "raw_message", None)
        if raw_obj is None:
            return targets
        try:
            if isinstance(raw_obj, list):
                segs = raw_obj
            else:
                from src.plugin_runtime.host.message_utils import PluginMessageUtils
                segs = PluginMessageUtils._message_sequence_to_dict(raw_obj, include_binary_data=False)
            if isinstance(segs, list):
                for seg in segs:
                    if not isinstance(seg, dict) or seg.get("type") != "reply":
                        continue
                    seg_data = seg.get("data", {})
                    if isinstance(seg_data, dict):
                        _add(seg_data.get("target_message_id") or seg_data.get("id") or seg_data.get("message_id"))
                    elif isinstance(seg_data, str):
                        _add(seg_data)
        except Exception as exc:
            logger.debug("[LLM2picBridge] 收集 reply 目标失败: %s", exc)
        return targets

    async def _ctx_resolve_session_reference(
        self, session_message: Any, stream_id: str = ""
    ) -> dict[str, Any]:
        """解析当前命令消息上的参考图，并报告显式引用是否失效。

        返回:
          image: base64 或 None
          source: current_message_image / reply_to / reply_seg / None
          reply_to: 显式引用 id（若有）
          error: 取图失败原因
        """
        info: dict[str, Any] = {
            "image": None,
            "source": None,
            "reply_to": "",
            "error": "",
        }
        if not session_message:
            return info

        # 1) 当前消息自身图片
        if isinstance(session_message, dict):
            raw = session_message.get("raw_message", [])
            if isinstance(raw, list):
                img = self._extract_image_from_segments(raw)
                if img:
                    info["image"] = img
                    info["source"] = "current_message_image"
                    logger.info("[LLM2picBridge] 当前消息自身含图 (dict), b64_len=%s", len(img))
                    return info
        else:
            raw_obj = getattr(session_message, "raw_message", None)
            if raw_obj is not None:
                try:
                    if isinstance(raw_obj, list):
                        segs = raw_obj
                    else:
                        from src.plugin_runtime.host.message_utils import PluginMessageUtils
                        segs = PluginMessageUtils._message_sequence_to_dict(
                            raw_obj, include_binary_data=True
                        )
                    if isinstance(segs, list):
                        img = self._extract_image_from_segments(segs)
                        if img:
                            info["image"] = img
                            info["source"] = "current_message_image"
                            logger.info("[LLM2picBridge] 当前消息自身含图 (obj), b64_len=%s", len(img))
                            return info
                except Exception as exc:
                    logger.debug("[LLM2picBridge] 解析 session_message.raw_message 失败: %s", exc)

        # 2) 显式引用
        targets = self._ctx_collect_reply_targets(session_message)
        if targets:
            info["reply_to"] = targets[0]
        for target in targets:
            img = await self._ctx_get_image_by_message_id(str(target), stream_id)
            if img:
                info["image"] = img
                info["source"] = "reply_to"
                info["reply_to"] = str(target)
                logger.info(
                    "[LLM2picBridge] 从当前消息 reply_to 取到参考图: reply_to=%s b64_len=%s",
                    target,
                    len(img),
                )
                return info
            logger.warning(
                "[LLM2picBridge] 显式引用取图失败: reply_to=%s stream_id=%s",
                target,
                stream_id,
            )

        if targets:
            info["error"] = (
                f"引用消息无法访问或不含图片（reply_to={targets[0]}）。"
                "原消息可能已被撤回/未入库/协议层失效"
            )
        return info

    async def _ctx_extract_image_from_session_message(self, session_message: Any, stream_id: str = "") -> Optional[str]:
        """兼容旧调用：只返回图片 base64。"""
        info = await self._ctx_resolve_session_reference(session_message, stream_id)
        return info.get("image")


    async def _ctx_extract_image_from_recent(self, stream_id: str) -> Optional[str]:
        """从最近的聊天消息中提取图片（兜底，用于 edit / 无显式引用时）。

        注意：/pic i2i 引用图应优先走 _ctx_extract_image_from_session_message，
        本函数会从最新消息向前扫，容易拿到 bot 刚出的图而不是用户引用的图。

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
                mid = msg.get("message_id") or msg.get("id") or "?"
                logger.info(
                    "[LLM2picBridge] recent 命中消息自身图片: message_id=%s b64_len=%s",
                    mid,
                    len(image_base64),
                )
                return image_base64

            # 2. 检查消息是否引用了另一条消息，尝试获取引用消息中的图片
            reply_to = msg.get("reply_to")
            if reply_to:
                reply_image = await self._ctx_get_image_by_message_id(reply_to, stream_id)
                if reply_image:
                    logger.info(
                        "[LLM2picBridge] recent 命中消息 reply_to 图片: reply_to=%s b64_len=%s",
                        reply_to,
                        len(reply_image),
                    )
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
            logger.warning("[LLM2picBridge] 获取引用消息异常: message_id=%s err=%s", message_id, exc)
            return None

        if not isinstance(result, dict):
            logger.warning("[LLM2picBridge] 获取引用消息返回非 dict: message_id=%s type=%s", message_id, type(result))
            return None
        if result.get("success") is False:
            logger.warning(
                "[LLM2picBridge] 获取引用消息失败: message_id=%s error=%s",
                message_id,
                result.get("error"),
            )
            return None
        msg = result.get("message") if "message" in result else result
        if not isinstance(msg, dict) or not msg:
            logger.warning("[LLM2picBridge] 引用消息不存在/空: message_id=%s stream_id=%s", message_id, stream_id)
            return None

        raw_message = msg.get("raw_message", [])
        if not isinstance(raw_message, list):
            logger.warning("[LLM2picBridge] 引用消息 raw_message 非列表: message_id=%s", message_id)
            return None

        img = self._extract_image_from_segments(raw_message)
        if not img:
            # 也尝试 hash 路径：仅有 hash 时从本地 images 读
            for seg in raw_message:
                if not isinstance(seg, dict) or seg.get("type") != "image":
                    continue
                data = seg.get("data", {})
                image_hash = ""
                if isinstance(data, dict):
                    image_hash = str(data.get("hash") or data.get("file") or "").strip()
                if image_hash:
                    local = self._load_image_b64_by_hash(image_hash)
                    if local:
                        logger.info(
                            "[LLM2picBridge] 引用消息经 hash 取图: message_id=%s hash=%s b64_len=%s",
                            message_id,
                            image_hash[:16],
                            len(local),
                        )
                        return local
            logger.warning(
                "[LLM2picBridge] 引用消息存在但不含可解码图片: message_id=%s segs=%s",
                message_id,
                [s.get("type") if isinstance(s, dict) else type(s).__name__ for s in raw_message[:8]],
            )
        return img


    def _load_image_b64_by_hash(self, image_hash: str) -> Optional[str]:
        """从 data/images / images 表按 hash 读图。"""
        import base64
        from pathlib import Path as _Path

        h = str(image_hash or "").strip()
        if not h:
            return None
        stem = h.split("/")[-1].split(".")[0]
        if not stem:
            return None

        candidates: list[_Path] = []
        for root in (_Path("data/images"), _Path("/root/seren/rdev-Maibot/data/images")):
            for ext in (".png", ".jpg", ".jpeg", ".webp", ""):
                candidates.append(root / f"{stem}{ext}")
            if root.exists():
                candidates.extend(root.glob(stem + ".*"))
        try:
            import sqlite3

            for db in (_Path("data/MaiBot.db"), _Path("/root/seren/rdev-Maibot/data/MaiBot.db")):
                if not db.exists():
                    continue
                con = sqlite3.connect(str(db))
                try:
                    row = con.execute(
                        "SELECT full_path FROM images WHERE image_hash=? LIMIT 1",
                        (stem,),
                    ).fetchone()
                    if row and row[0]:
                        candidates.insert(0, _Path(str(row[0])))
                finally:
                    con.close()
        except Exception:
            pass

        seen: set[str] = set()
        for p in candidates:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            try:
                if p.is_file():
                    raw = p.read_bytes()
                    if raw:
                        return base64.b64encode(raw).decode("ascii")
            except Exception:
                continue
        return None

    def _extract_image_from_segments(self, segments: list) -> Optional[str]:
        """从消息段列表中提取第一张图片的 base64 数据。支持 hash-only 段。"""
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            if seg.get("type") != "image":
                continue
            b64 = seg.get("binary_data_base64", "")
            if b64:
                return b64

            image_hash = str(seg.get("hash") or "").strip()
            data = seg.get("data", "")
            if not image_hash and isinstance(data, dict):
                image_hash = str(data.get("hash") or data.get("file") or "").strip()
            if (
                image_hash
                and not image_hash.startswith("http")
                and not image_hash.startswith("base64://")
            ):
                local = self._load_image_b64_by_hash(image_hash)
                if local:
                    return local

            if isinstance(data, str):
                if data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                    return data
                if data.startswith("base64://"):
                    return data[9:]
                if data.startswith("http"):
                    success, result = download_image_to_base64(data)
                    if success:
                        return result
                if re.fullmatch(r"[0-9a-fA-F]{32,128}", data.strip() or ""):
                    local = self._load_image_b64_by_hash(data.strip())
                    if local:
                        return local
            elif isinstance(data, dict):
                img_url = data.get("url") or data.get("file")
                if isinstance(img_url, str) and img_url.startswith("http"):
                    success, result = download_image_to_base64(img_url)
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


class _ToolRuntimeProxy(DrawPictureToolMetadata, ImageClientMixin):
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


class _CommandRuntimeProxy(DirectPicCommand, ImageClientMixin):
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
