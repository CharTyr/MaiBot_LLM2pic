"""图片生成 API 客户端与通用图片处理逻辑。"""

from http.client import IncompleteRead
from io import BytesIO
from typing import Any, Optional, Tuple
import asyncio
import base64
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from src.common.logger import get_logger

from .utils import (
    _compress_image_if_needed,
    _looks_like_image_bytes,
    _looks_like_image_url,
    _normalize_url_for_request,
    _probe_url_is_image,
    download_image_to_base64,
    get_image_mime_type,
)

logger = get_logger("MaiBot_LLM2pic")

_NEWAPI_NAI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_NEWAPI_NAI_SEEDS_PATTERN = re.compile(r"<!--\s*seeds:\s*(\[[^\]]*])\s*-->")
_NEWAPI_NAI_POSITION_GRID_RE = re.compile(r"^[A-E][1-5]$")
_NEWAPI_NAI_MULTI_CHARACTER_MODEL_KEYWORDS = ("nai-diffusion-4",)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")


def _strip_cjk_from_prompt_segment(segment: str) -> str:
    if not segment or not segment.strip():
        return segment
    stripped = _CJK_RE.sub("", segment)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip(" ,;")
    return stripped


def _sanitize_prompt_for_newapi(prompt: str) -> str:
    if not prompt or not _CJK_RE.search(prompt):
        return prompt
    parts = [p.strip() for p in prompt.split(",")]
    cleaned = [_strip_cjk_from_prompt_segment(p) for p in parts]
    cleaned = [p for p in cleaned if p]
    result = ", ".join(cleaned)
    result = re.sub(r",\s*,+", ", ", result).strip(" ,")
    return result


class ImageClientMixin:
    """复用图片输入提取、结果处理和各类生图 API 请求。"""

    log_prefix = "[ImageClient]"

    def get_config(self, path: str, default: object = None) -> object:
        return default

    async def send_text(self, text: str) -> bool:
        del text
        return False

    async def send_image(self, image_base64: str) -> bool:
        del image_base64
        return False

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
        sanitized = _sanitize_prompt_for_newapi(merged)
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

    async def _handle_image_result(self, result: str) -> Tuple[bool, str]:
        """发送 base64 图片或下载 URL 后发送。"""
        if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
            crop_enabled = bool(self.get_config("generation.crop_enabled", False))
            if crop_enabled:
                try:
                    image_bytes = base64.b64decode(result)
                    image_bytes = self._crop_image(image_bytes)
                    result = base64.b64encode(image_bytes).decode("utf-8")
                except Exception as exc:
                    logger.error(f"{self.log_prefix} Base64 图片裁切失败: {exc}")

            result = _compress_image_if_needed(result)
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
        encode_result = _compress_image_if_needed(encode_result)
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

    def _make_gradio_image_request(
        self,
        prompt: str,
        base_url: Optional[str] = None,
        gradio_params: Optional[dict] = None,
    ) -> Tuple[bool, str]:
        if base_url is None:
            base_url = str(self.get_config("api.base_url", "") or "")
        if gradio_params:
            resolution = gradio_params.get("resolution", "1024x1024 ( 1:1 )")
            steps = gradio_params.get("steps", 8)
            shift = gradio_params.get("shift", 3)
            timeout = gradio_params.get("timeout", 120)
        else:
            resolution = self.get_config("generation.gradio_resolution", "1024x1024 ( 1:1 )")
            steps = self.get_config("generation.gradio_steps", 8)
            shift = self.get_config("generation.gradio_shift", 3)
            timeout = self.get_config("generation.gradio_timeout", 120)

        endpoint = f"{base_url.rstrip('/')}/gradio_api/call/generate"
        payload = {"data": [prompt, resolution, 42, steps, shift, True, []]}
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        logger.info(f"{self.log_prefix} 发起 Gradio 图片请求, Prompt: {prompt[:100]}...")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                response_body = response.read().decode("utf-8")
                if not 200 <= response.status < 300:
                    return False, f"POST 请求失败 (状态码 {response.status})"
                response_data = json.loads(response_body)
                event_id = response_data.get("event_id")
                if not event_id:
                    return False, "未获取到 event_id"

            result_endpoint = f"{base_url.rstrip('/')}/gradio_api/call/generate/{event_id}"
            start_time = time.time()
            while time.time() - start_time < int(timeout):
                try:
                    result_req = urllib.request.Request(result_endpoint, method="GET")
                    with urllib.request.urlopen(result_req, timeout=30) as result_response:
                        result_body = result_response.read().decode("utf-8")
                    for line in result_body.split("\n"):
                        if not line.startswith("data: "):
                            continue
                        try:
                            result_data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        if isinstance(result_data, list) and result_data:
                            gallery = result_data[0]
                            if isinstance(gallery, list) and gallery:
                                first_image = gallery[0]
                                if isinstance(first_image, dict):
                                    image_url = (first_image.get("image") or {}).get("url")
                                    if image_url:
                                        logger.info(f"{self.log_prefix} 获取到 Gradio 图片 URL")
                                        return True, image_url
                    time.sleep(2)
                except Exception as exc:
                    logger.debug(f"{self.log_prefix} Gradio 轮询中: {exc}")
                    time.sleep(2)
            return False, f"轮询超时（{timeout}秒）"
        except Exception as exc:
            logger.error(f"{self.log_prefix} Gradio API 请求错误: {exc!r}", exc_info=True)
            return False, str(exc)

    def _make_sd_api_request(
        self,
        prompt: str,
        base_url: str,
        api_key: str,
        sd_params: Optional[dict] = None,
    ) -> Tuple[bool, str]:
        endpoint = f"{base_url.rstrip('/')}/api/v1/generate_image"
        payload: dict[str, object] = {"prompt": prompt}
        if sd_params:
            if sd_params.get("negative_prompt"):
                payload["negative_prompt"] = sd_params["negative_prompt"]
            payload["width"] = sd_params.get("width", 512)
            payload["height"] = sd_params.get("height", 512)
            payload["steps"] = sd_params.get("steps", 20)
            payload["cfg"] = sd_params.get("cfg", 7.0)
            payload["model_index"] = sd_params.get("model_index", 0)
            payload["seed"] = sd_params.get("seed", -1)

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0",
        }
        logger.info(f"{self.log_prefix} 发起 SD API 图片请求, Prompt: {prompt[:100]}...")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                response_body = response.read().decode("utf-8")
                if not 200 <= response.status < 300:
                    return False, f"SD API 请求失败 (状态码 {response.status})"
                response_data = json.loads(response_body)

            image_data = self._extract_image_data(response_data)
            if image_data:
                logger.info(f"{self.log_prefix} SD API 返回图片成功")
                return True, image_data
            if isinstance(response_data, str) and response_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                return True, response_data
            return False, f"SD API 响应中未找到图片数据: {str(response_data)[:200]}"
        except Exception as exc:
            logger.error(f"{self.log_prefix} SD API 请求错误: {exc!r}", exc_info=True)
            return False, str(exc)

    @staticmethod
    def _extract_image_data(response_data: object) -> Optional[str]:
        if not isinstance(response_data, dict):
            return None
        direct = response_data.get("image") or response_data.get("url")
        if isinstance(direct, str):
            return direct
        images = response_data.get("images")
        if isinstance(images, list) and images:
            first_img = images[0]
            if isinstance(first_img, str):
                return first_img
            if isinstance(first_img, dict):
                candidate = first_img.get("url") or first_img.get("image") or first_img.get("base64")
                if isinstance(candidate, str):
                    return candidate
        data_obj = response_data.get("data")
        if isinstance(data_obj, dict):
            candidate = data_obj.get("image") or data_obj.get("url") or data_obj.get("image_url")
            return candidate if isinstance(candidate, str) else None
        return data_obj if isinstance(data_obj, str) else None

    def _make_regex_url_request(self, prompt: str, url_template: str) -> Tuple[bool, str]:
        if not url_template or not url_template.strip():
            return False, "regex_url 未配置 URL 模板"

        encoded_prompt = urllib.parse.quote(prompt, safe="")
        if "$1" in url_template:
            endpoint = url_template.replace("$1", encoded_prompt)
        else:
            connector = "&" if "?" in url_template else "?"
            endpoint = f"{url_template}{connector}tag={encoded_prompt}"
        endpoint = _normalize_url_for_request(endpoint)

        headers = {"Accept": "*/*", "User-Agent": "Mozilla/5.0"}
        parsed = urllib.parse.urlsplit(endpoint)
        logger.info(f"{self.log_prefix} 发起 regex_url 请求: {parsed.scheme}://{parsed.netloc}{parsed.path}")
        req = urllib.request.Request(endpoint, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()
                try:
                    response_body = response.read()
                except IncompleteRead as exc:
                    logger.warning(f"{self.log_prefix} IncompleteRead: 已读 {len(exc.partial)} bytes")
                    response_body = exc.partial

            if not response_body:
                return False, "regex_url 响应为空"
            if content_type.startswith("image/") or _looks_like_image_bytes(response_body[:32]):
                return True, base64.b64encode(response_body).decode("utf-8")

            text = response_body.decode("utf-8", errors="ignore")
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    candidates: list[str] = []
                    for value in (data.get("url"), data.get("image_url")):
                        if isinstance(value, str):
                            candidates.append(value)
                    nested = data.get("data")
                    if isinstance(nested, dict):
                        for value in (nested.get("url"), nested.get("image_url")):
                            if isinstance(value, str):
                                candidates.append(value)
                    elif isinstance(nested, str):
                        candidates.append(nested)
                    for image_url in candidates:
                        if _looks_like_image_url(image_url) or _probe_url_is_image(image_url):
                            return True, image_url
            except Exception:
                pass

            for url in re.findall(r"https?://[^\s\)\]\"'<>]+", text):
                if _looks_like_image_url(url) or _probe_url_is_image(url):
                    return True, url
            return False, f"regex_url 响应中未找到图片数据: {text[:200]}"
        except Exception as exc:
            logger.error(f"{self.log_prefix} regex_url 请求错误: {exc!r}", exc_info=True)
            return False, str(exc)

    def _make_novelai_request(
        self,
        prompt: str,
        api_key: str,
        novelai_params: Optional[dict] = None,
    ) -> Tuple[bool, str]:
        params = novelai_params or {}
        endpoint = "https://image.novelai.net/ai/generate-image"
        model = params.get("model", "nai-diffusion-4-5-full")
        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        payload = {
            "input": prompt,
            "model": model,
            "action": "generate",
            "parameters": {
                "width": params.get("width", 832),
                "height": params.get("height", 1216),
                "scale": params.get("scale", 5.0),
                "sampler": params.get("sampler", "k_euler"),
                "steps": params.get("steps", 28),
                "seed": seed,
                "n_samples": 1,
                "negative_prompt": params.get("negative_prompt", ""),
                "noise_schedule": "karras",
                "qualityToggle": True,
                "ucPreset": 0,
            },
        }
        if "nai-diffusion-4" in str(model):
            payload["parameters"]["cfg_rescale"] = 0

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/zip, image/*",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0",
        }
        logger.info(f"{self.log_prefix} 发起 NovelAI 图片请求, model={model}, prompt={prompt[:80]}...")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=int(params.get("timeout", 120))) as response:
                response_data = response.read()
                content_type = response.headers.get("Content-Type", "")
                if not 200 <= response.status < 300:
                    return False, f"NovelAI 请求失败 (状态码 {response.status})"

            if "zip" in content_type or response_data[:4] == b"PK\x03\x04":
                try:
                    with zipfile.ZipFile(BytesIO(response_data)) as zf:
                        file_list = zf.namelist()
                        if not file_list:
                            return False, "NovelAI 返回的 zip 文件为空"
                        image_bytes = zf.read(file_list[0])
                    logger.info(f"{self.log_prefix} NovelAI 请求成功，图片大小: {len(image_bytes)} bytes")
                    return True, base64.b64encode(image_bytes).decode("utf-8")
                except zipfile.BadZipFile:
                    return False, "NovelAI 返回的不是有效的 zip 文件"
            if content_type.startswith("image/") or response_data.startswith((b"\x89PNG", b"\xff\xd8")):
                logger.info(f"{self.log_prefix} NovelAI 请求成功，图片大小: {len(response_data)} bytes")
                return True, base64.b64encode(response_data).decode("utf-8")
            try:
                return False, f"NovelAI 返回未知格式: {response_data.decode('utf-8')[:500]}"
            except UnicodeDecodeError:
                return False, f"NovelAI 返回未知格式 (Content-Type: {content_type})"
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8")[:300]
            except Exception:
                pass
            logger.error(f"{self.log_prefix} NovelAI HTTP 错误: {exc.code} - {error_body}")
            if exc.code == 401:
                return False, "NovelAI 认证失败，请检查 API token"
            if exc.code == 402:
                return False, "NovelAI 配额不足，请充值 Anlas"
            if exc.code == 429:
                return False, "NovelAI 请求过于频繁，请稍后重试"
            return False, f"NovelAI HTTP 错误 {exc.code}: {error_body}"
        except urllib.error.URLError as exc:
            logger.error(f"{self.log_prefix} NovelAI 连接错误: {exc.reason}")
            return False, f"连接错误: {exc.reason}"
        except Exception as exc:
            logger.error(f"{self.log_prefix} NovelAI 请求错误: {exc!r}", exc_info=True)
            return False, str(exc)

    @staticmethod
    def _parse_size_to_newapi_nai_size(size: object) -> object:
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return [int(size[0]), int(size[1])]

        normalized_size = str(size or "").strip().lower()
        if not normalized_size:
            return [832, 1216]
        # 语义画幅映射到 NAI 分辨率
        size_map = {
            "portrait": [832, 1216],
            "landscape": [1216, 832],
            "square": [1024, 1024],
        }
        if normalized_size in size_map:
            return size_map[normalized_size]
        if "x" in normalized_size:
            parts = normalized_size.split("x", 1)
            try:
                return [int(parts[0]), int(parts[1])]
            except (ValueError, IndexError):
                pass
        return [832, 1216]

    @staticmethod
    def _extract_data_uri_image(content: str) -> Optional[str]:
        data_uri_match = re.search(r"data:image/[^;)]+;base64,([A-Za-z0-9+/=\s]+)", content)
        if data_uri_match:
            return "".join(data_uri_match.group(1).split())
        return None

    @staticmethod
    def _extract_newapi_nai_message_content(choice: object) -> str:
        if not isinstance(choice, dict):
            return ""
        message = choice.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return ""

    @staticmethod
    def _extract_newapi_nai_seeds(content: str) -> list[object]:
        match = _NEWAPI_NAI_SEEDS_PATTERN.search(content or "")
        if not match:
            return []
        try:
            seeds = json.loads(match.group(1))
        except (TypeError, ValueError):
            return []
        return seeds if isinstance(seeds, list) else []

    @staticmethod
    def _extract_newapi_nai_error(response_data: object) -> str:
        if not isinstance(response_data, dict):
            return ""
        error = response_data.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            code = str(error.get("code") or "").strip()
            return f"{message} (code={code})" if message and code else message or code
        if isinstance(error, str) and error.strip():
            return error.strip()
        for key in ("message", "detail"):
            value = response_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _is_newapi_nai_retryable_error(exc: urllib.error.URLError) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code in _NEWAPI_NAI_RETRYABLE_STATUS_CODES
        reason = str(getattr(exc, "reason", exc)).lower()
        return any(token in reason for token in ("timed out", "timeout", "connection reset", "temporarily unavailable"))

    @staticmethod
    def _normalize_newapi_nai_token(api_key: str) -> str:
        token = str(api_key or "").strip()
        if token.lower().startswith("bearer "):
            return token.split(" ", 1)[1].strip()
        return token

    @staticmethod
    def _normalize_newapi_nai_proxy_mode(proxy_mode: object) -> str:
        mode = str(proxy_mode or "auto").strip().lower()
        return mode if mode in {"auto", "inherit", "direct"} else "auto"

    @staticmethod
    def _normalize_newapi_nai_characters(characters: Optional[list[dict[str, Any]]]) -> list[dict[str, str]]:
        if not characters:
            return []

        cleaned: list[dict[str, str]] = []
        for item in characters:
            if not isinstance(item, dict):
                continue
            char_prompt = str(item.get("prompt") or "").strip()
            if not char_prompt:
                continue
            raw_position = str(item.get("position") or "").strip().upper()
            entry = {
                "prompt": _sanitize_prompt_for_newapi(char_prompt),
                "negative_prompt": _sanitize_prompt_for_newapi(str(item.get("negative_prompt") or "").strip()),
                "position": raw_position if _NEWAPI_NAI_POSITION_GRID_RE.match(raw_position) else "",
            }
            cleaned.append(entry)

        if len(cleaned) < 2:
            return []
        if any(not item["position"] for item in cleaned):
            for item in cleaned:
                item["position"] = ""

        result: list[dict[str, str]] = []
        for item in cleaned:
            entry = {"prompt": item["prompt"]}
            if item["negative_prompt"]:
                entry["negative_prompt"] = item["negative_prompt"]
            if item["position"]:
                entry["position"] = item["position"]
            result.append(entry)
        return result

    @staticmethod
    def _newapi_nai_supports_characters(model: str) -> bool:
        lowered = str(model or "").lower()
        return any(keyword in lowered for keyword in _NEWAPI_NAI_MULTI_CHARACTER_MODEL_KEYWORDS)

    def _newapi_nai_urlopen(self, req: urllib.request.Request, *, timeout: int, proxy_mode: str) -> object:
        if proxy_mode == "inherit":
            return urllib.request.urlopen(req, timeout=timeout)
        if proxy_mode == "direct":
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return opener.open(req, timeout=timeout)

        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.URLError as exc:
            if isinstance(exc, urllib.error.HTTPError):
                raise
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            logger.warning(f"{self.log_prefix} NewAPI NAI 继承代理失败，尝试直连: {exc}")
            return opener.open(req, timeout=timeout)

    @classmethod
    def _parse_newapi_nai_response(cls, response_data: object) -> Tuple[bool, str]:
        if not isinstance(response_data, dict):
            return False, "NewAPI 响应数据格式错误"
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            return False, cls._extract_newapi_nai_error(response_data) or "NewAPI 未返回 choices"

        content = cls._extract_newapi_nai_message_content(choices[0])
        if not content:
            return False, cls._extract_newapi_nai_error(response_data) or "NewAPI 未返回 message.content"

        seeds = cls._extract_newapi_nai_seeds(content)
        if seeds:
            logger.info(f"NewAPI NAI 返回 seeds={seeds}, usage={response_data.get('usage')}")
        else:
            logger.info(f"NewAPI NAI 返回 usage={response_data.get('usage')}")

        image_base64 = cls._extract_data_uri_image(content)
        if image_base64:
            try:
                base64.b64decode(image_base64, validate=False)
            except Exception:
                return False, "NewAPI 返回的 base64 数据无法解码"
            return True, image_base64

        text_error = ""
        try:
            text_error = cls._extract_newapi_nai_error(json.loads(content.strip()))
        except (TypeError, ValueError):
            text_error = ""
        return False, text_error or f"NewAPI 响应中未找到图片数据: {content[:200]}"

    def _make_newapi_nai_request(
        self,
        prompt: str,
        base_url: str,
        api_key: str,
        model: str,
        params: Optional[dict] = None,
        characters: Optional[list[dict[str, Any]]] = None,
    ) -> Tuple[bool, str]:
        """调用 NewAPI chat/completions 形式的 NAI 绘图渠道。"""
        options = params or {}
        draw_payload: dict[str, object] = {
            "model": model,
            "prompt": _sanitize_prompt_for_newapi(prompt),
            "size": self._parse_size_to_newapi_nai_size(options.get("size", "portrait")),
            "steps": min(max(int(options.get("steps", 23) or 23), 1), 28),
            "scale": options.get("scale", 5),
            "sampler": str(options.get("sampler", "k_euler_ancestral") or "k_euler_ancestral"),
            "n_samples": 1,
            "image_format": str(options.get("image_format", "png") or "png"),
        }
        negative_prompt = str(options.get("negative_prompt", "") or "").strip()
        if negative_prompt:
            sanitized_neg = _sanitize_prompt_for_newapi(negative_prompt)
            if sanitized_neg:
                draw_payload["negative_prompt"] = sanitized_neg
            elif negative_prompt != sanitized_neg:
                logger.warning(
                    f"{self.log_prefix} negative_prompt 含 CJK，sanitize 后为空，已丢弃"
                )
        seed = options.get("seed", None)
        if seed not in (None, "", -1):
            draw_payload["seed"] = int(seed)
        if bool(options.get("quality_toggle", False)):
            draw_payload["qualityToggle"] = True
        if bool(options.get("auto_smea", False)):
            draw_payload["autoSmea"] = True
        if bool(options.get("variety_boost", False)):
            draw_payload["variety_boost"] = True
        normalized_characters = self._normalize_newapi_nai_characters(characters)
        if normalized_characters and self._newapi_nai_supports_characters(model):
            draw_payload["characters"] = normalized_characters
            draw_payload["use_coords"] = all("position" in item for item in normalized_characters)
            draw_payload["use_order"] = True
        elif normalized_characters:
            logger.warning(
                f"{self.log_prefix} NewAPI NAI 模型 {model!r} 不支持 characters[]，已降级为单 prompt"
            )
        extra_params = options.get("extra_params") or {}
        if isinstance(extra_params, dict):
            for key, value in extra_params.items():
                if value not in (None, ""):
                    draw_payload[str(key)] = value

        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": json.dumps(draw_payload, ensure_ascii=False)}],
            "stream": False,
            "max_tokens": int(options.get("max_tokens", 100000) or 100000),
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._normalize_newapi_nai_token(api_key)}",
            "User-Agent": "Mozilla/5.0",
        }
        logger.info(f"{self.log_prefix} 发起 NewAPI NAI 绘图请求: model={model}, prompt={prompt[:80]}...")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        timeout = int(options.get("timeout", 180) or 180)
        retry_attempts = max(1, min(int(options.get("retry_attempts", 3) or 3), 5))
        proxy_mode = self._normalize_newapi_nai_proxy_mode(options.get("proxy_mode", "auto"))

        for attempt in range(1, retry_attempts + 1):
            try:
                with self._newapi_nai_urlopen(req, timeout=timeout, proxy_mode=proxy_mode) as response:
                    response_body = response.read().decode("utf-8")
                    if not 200 <= response.status < 300:
                        return False, f"NewAPI 请求失败 (状态码 {response.status})"
                    response_data = json.loads(response_body)

                return self._parse_newapi_nai_response(response_data)
            except urllib.error.HTTPError as exc:
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8")[:300]
                except Exception:
                    pass
                if exc.code in _NEWAPI_NAI_RETRYABLE_STATUS_CODES and attempt < retry_attempts:
                    sleep_seconds = 6.0 if exc.code == 429 else 1.5 * attempt
                    logger.warning(f"{self.log_prefix} NewAPI HTTP {exc.code}，{sleep_seconds:.1f}s 后重试")
                    time.sleep(sleep_seconds)
                    continue
                logger.error(f"{self.log_prefix} NewAPI HTTP 错误: {exc.code} - {error_body}")
                return False, f"NewAPI HTTP 错误 {exc.code}: {error_body}"
            except urllib.error.URLError as exc:
                if self._is_newapi_nai_retryable_error(exc) and attempt < retry_attempts:
                    sleep_seconds = 1.5 * attempt
                    logger.warning(f"{self.log_prefix} NewAPI 网络错误，{sleep_seconds:.1f}s 后重试: {exc}")
                    time.sleep(sleep_seconds)
                    continue
                logger.error(f"{self.log_prefix} NewAPI 连接错误: {exc}")
                return False, f"NewAPI 连接错误: {getattr(exc, 'reason', exc)}"
            except json.JSONDecodeError as exc:
                logger.error(f"{self.log_prefix} NewAPI JSON 解析失败: {exc}")
                return False, "NewAPI 返回了非 JSON 响应"
            except Exception as exc:
                logger.error(f"{self.log_prefix} NewAPI 请求错误: {exc!r}", exc_info=True)
                return False, str(exc)
        return False, "NewAPI 请求失败"

    def _make_http_image_request(
        self,
        prompt: str,
        model: str,
        size: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        input_image_base64: Optional[str] = None,
    ) -> Tuple[bool, str]:
        del size
        if base_url is None:
            base_url = str(self.get_config("api.base_url", "") or "")
        if api_key is None:
            api_key = str(self.get_config("api.api_key", "") or "")

        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        if input_image_base64:
            mime_type = get_image_mime_type(input_image_base64)
            message_content: object = [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{input_image_base64}"}},
                {"type": "text", "text": prompt},
            ]
            logger.info(f"{self.log_prefix} 发起图生图请求: {model}, Prompt: {prompt[:100]}...")
        else:
            message_content = prompt
            logger.info(f"{self.log_prefix} 发起文生图请求: {model}, Prompt: {prompt[:100]}...")

        payload = {"model": model, "messages": [{"role": "user", "content": message_content}]}
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0",
        }
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                response_body = response.read().decode("utf-8")
                if not 200 <= response.status < 300:
                    return False, f"API 请求失败 (状态码 {response.status})"
                response_data = json.loads(response_body)

            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            data_uri_match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content)
            if data_uri_match:
                return True, data_uri_match.group(1)

            urls = re.findall(r"https?://[^\s\)\]\"'<>]+", content)
            for url in urls:
                if any(ext in url.lower() for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]) or "image" in url.lower():
                    return True, url
            if urls:
                return True, urls[0]
            if content.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                return True, content
            return False, f"API 响应中未找到图片数据: {content[:200]}"
        except Exception as exc:
            logger.error(f"{self.log_prefix} HTTP 请求错误: {exc!r}", exc_info=True)
            return False, str(exc)
