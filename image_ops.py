"""图片发送/裁剪/提示词拼接等与 API 无关的操作。

从 ImageClientMixin 抽出，避免 Tool/Command 元数据继承整坨出图客户端。
"""

from __future__ import annotations

from http.client import IncompleteRead
from io import BytesIO
from typing import Any, Optional, Tuple
import asyncio
import base64
import re
import urllib.error
import urllib.request

from src.common.logger import get_logger

from .github_uploader import upload_image_to_github
from .utils import (
    _compress_image_if_needed,
    download_image_to_base64,
    get_image_mime_type,
)

logger = get_logger("MaiBot_LLM2pic")

# CJK 清洗（与 clients/newapi_nai.sanitize_prompt 行为对齐；legacy 路径仍用此实现）
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef\u2e80-\u2eff\uac00-\ud7af\u3040-\u30ff\uf900-\ufaff]+"
)


def strip_cjk_from_prompt_segment(segment: str) -> str:
    if not segment or not segment.strip():
        return segment
    stripped = _CJK_RE.sub("", segment)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip(" ,;")
    return stripped


def sanitize_prompt_for_newapi(prompt: str) -> str:
    if not prompt or not _CJK_RE.search(prompt):
        return prompt
    parts = [p.strip() for p in prompt.split(",")]
    cleaned = [strip_cjk_from_prompt_segment(p) for p in parts]
    cleaned = [p for p in cleaned if p]
    result = ", ".join(cleaned)
    result = re.sub(r",\s*,+", ", ", result).strip(" ,")
    return result


class ImageOps:
    """出图周边操作：拼 prompt、发图、裁剪、从消息抽图。

    不是 ImageClient。不含 _make_* API 请求。
    """

    log_prefix = "[ImageOps]"

    def get_config(self, path: str, default: object = None) -> object:
        return default

    async def send_text(self, text: str) -> bool:
        del text
        return False

    async def send_image(self, image_base64: str) -> bool:
        del image_base64
        return False

    def _schedule_github_upload(self, image_base64: str, *, prompt: str = "") -> None:
        """图片发送成功后，后台异步上传到 GitHub 仓库（不阻塞主流程）。

        失败仅记录日志。通过 ``asyncio.create_task`` 调度，避免影响返回值。
        ``prompt`` 会作为 commit message 保存，前端可据此展示该图的 tag。
        """
        try:
            asyncio.create_task(
                upload_image_to_github(image_base64, get_config=self.get_config, prompt=prompt)
            )
        except Exception as exc:
            logger.warning(f"{self.log_prefix} 调度 GitHub 上传失败: {exc!r}")


    async def _extract_input_image(self) -> Optional[str]:
        """从当前消息中提取图片，用于图生图。"""
        try:
            message = getattr(self, "message", None)
            if not message:
                logger.debug(f"{self.log_prefix} 消息中未找到图片")
                return None

            if hasattr(message, "message_segment"):
                for seg in message.message_segment:
                    if not hasattr(seg, "type") or seg.type != "image" or not hasattr(seg, "data"):
                        continue
                    img_data = seg.data
                    if isinstance(img_data, dict):
                        img_url = img_data.get("url") or img_data.get("file")
                        if isinstance(img_url, str) and img_url.startswith("http"):
                            success, result = await asyncio.to_thread(download_image_to_base64, img_url)
                            if success:
                                return result
                        if isinstance(img_url, str) and img_url.startswith("base64://"):
                            return img_url[9:]
                    elif isinstance(img_data, str):
                        if img_data.startswith("base64://"):
                            return img_data[9:]
                        if img_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                            return img_data

            if hasattr(message, "raw_message"):
                raw_message = str(message.raw_message)
                cq_matches = re.findall(r"\[CQ:image[^\]]*url=([^\],]+)", raw_message)
                if cq_matches:
                    success, result = await asyncio.to_thread(download_image_to_base64, cq_matches[0])
                    if success:
                        return result

                url_matches = re.findall(r"https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp)", raw_message, re.IGNORECASE)
                if url_matches:
                    success, result = await asyncio.to_thread(download_image_to_base64, url_matches[0])
                    if success:
                        return result

            logger.debug(f"{self.log_prefix} 消息中未找到图片")
            return None
        except Exception as exc:
            logger.error(f"{self.log_prefix} 提取输入图片失败: {exc}", exc_info=True)
            return None


    def _build_final_prompt(self, generated_prompt: str, model_config: Optional[dict] = None) -> str:
        """构建最终图片提示词。"""
        custom_prompt_add = ""
        if model_config and model_config.get("custom_prompt_add"):
            custom_prompt_add = str(model_config.get("custom_prompt_add") or "")
        else:
            custom_prompt_add = str(self.get_config("generation.custom_prompt_add", "") or "")

        parts = [part.strip().strip(",") for part in (custom_prompt_add, generated_prompt) if part and part.strip()]
        merged = self._remove_duplicate_keywords(", ".join(parts))
        sanitized = sanitize_prompt_for_newapi(merged)
        if sanitized != merged:
            logger.warning(
                f"{self.log_prefix} 已从最终 prompt 移除 CJK 片段（NewAPI 禁止中文）: before_len={len(merged)} after_len={len(sanitized)}"
            )
        return sanitized

    @staticmethod

    def _remove_duplicate_keywords(prompt: str) -> str:
        if not prompt or not prompt.strip():
            return prompt

        seen: set[str] = set()
        unique_keywords: list[str] = []
        for keyword in (kw.strip() for kw in prompt.split(",") if kw.strip()):
            keyword_lower = keyword.lower()
            if keyword_lower in seen:
                continue
            seen.add(keyword_lower)
            unique_keywords.append(keyword)
        return ", ".join(unique_keywords)


    async def _handle_image_result(self, result: str, *, prompt: str = "") -> Tuple[bool, str]:
        """发送 base64 图片或下载 URL 后发送，并上传原始 PNG 到 GitHub（保留 tag 元数据）。"""
        if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
            crop_enabled = bool(self.get_config("generation.crop_enabled", False))
            if crop_enabled:
                try:
                    image_bytes = base64.b64decode(result)
                    image_bytes = self._crop_image(image_bytes)
                    result = base64.b64encode(image_bytes).decode("utf-8")
                except Exception as exc:
                    logger.error(f"{self.log_prefix} Base64 图片裁切失败: {exc}")

            # 上传原始图片到 GitHub（保留 PNG tag 元数据），不压缩直接发送
            self._schedule_github_upload(result, prompt=prompt)
            if await self.send_image(result):
                logger.info(f"{self.log_prefix} 图片已发送")
                return True, "图片已发送"
            logger.error(f"{self.log_prefix} 图片生成成功但发送失败")
            return False, "图片发送失败"

        image_url = result
        logger.info(f"{self.log_prefix} 下载图片: {image_url[:70]}...")
        try:
            encode_success, encode_result = await asyncio.to_thread(self._download_and_encode_base64, image_url)
        except Exception as exc:
            logger.error(f"{self.log_prefix} 下载图片失败: {exc!r}", exc_info=True)
            encode_success = False
            encode_result = str(exc)

        if not encode_success:
            logger.error(f"{self.log_prefix} 下载图片失败: {encode_result}")
            return False, f"图片下载失败: {encode_result}"
        # 上传原始图片到 GitHub，不压缩直接发送
        self._schedule_github_upload(encode_result, prompt=prompt)
        if await self.send_image(encode_result):
            logger.info(f"{self.log_prefix} 图片已发送")
            return True, "图片已发送"
        logger.error(f"{self.log_prefix} 图片下载成功但发送失败")
        return False, "图片发送失败"


    def _download_and_encode_base64(self, image_url: str) -> Tuple[bool, str]:
        try:
            req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status != 200:
                    return False, f"下载失败 (状态: {response.status})"
                try:
                    image_bytes = response.read()
                except IncompleteRead as exc:
                    logger.warning(f"{self.log_prefix} 下载图片 IncompleteRead，使用已读取的 {len(exc.partial)} bytes")
                    image_bytes = exc.partial

            if not image_bytes:
                return False, "下载的图片数据为空"
            if bool(self.get_config("generation.crop_enabled", False)):
                image_bytes = self._crop_image(image_bytes)
            return True, base64.b64encode(image_bytes).decode("utf-8")
        except Exception as exc:
            logger.error(f"{self.log_prefix} 下载图片错误: {exc!r}", exc_info=True)
            return False, str(exc)


    def _crop_image(self, image_bytes: bytes) -> bytes:
        try:
            from PIL import Image

            crop_position = str(self.get_config("generation.crop_position", "bottom") or "bottom")
            crop_pixels = int(self.get_config("generation.crop_pixels", 40) or 40)

            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            if crop_position == "bottom":
                if crop_pixels >= height:
                    return image_bytes
                crop_box = (0, 0, width, height - crop_pixels)
            elif crop_position == "top":
                if crop_pixels >= height:
                    return image_bytes
                crop_box = (0, crop_pixels, width, height)
            elif crop_position == "left":
                if crop_pixels >= width:
                    return image_bytes
                crop_box = (crop_pixels, 0, width, height)
            elif crop_position == "right":
                if crop_pixels >= width:
                    return image_bytes
                crop_box = (0, 0, width - crop_pixels, height)
            else:
                return image_bytes

            cropped_img = img.crop(crop_box)
            output = BytesIO()
            cropped_img.save(output, format=img.format or "PNG")
            logger.info(f"{self.log_prefix} 已裁切图片{crop_position} {crop_pixels} 像素")
            return output.getvalue()
        except ImportError:
            return image_bytes
        except Exception as exc:
            logger.error(f"{self.log_prefix} 图片裁切失败: {exc}", exc_info=True)
            return image_bytes


