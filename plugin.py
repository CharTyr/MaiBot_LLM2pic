"""
MaiBot_LLM2pic - MaiBot图片生成插件

使用LLM根据聊天记录和人设生成符合需求的prompt，然后调用图片生成API
支持文生图和图生图功能
"""

import asyncio
import json
import urllib.request
import urllib.parse
import base64
import traceback
import time
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

from maibot_sdk import Action, Command, MaiBotPlugin
from maibot_sdk.types import ActivationType

from src.common.logger import get_logger


# ===== 图片工具函数 =====

def download_image_to_base64(url: str, timeout: int = 60) -> Tuple[bool, str]:
    """
    下载图片并转换为 base64，容忍 IncompleteRead
    
    Args:
        url: 图片 URL
        timeout: 超时时间（秒）
        
    Returns:
        Tuple[bool, str]: (是否成功, base64数据或错误信息)
    """
    from http.client import IncompleteRead

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                try:
                    image_bytes = response.read()
                except IncompleteRead as e:
                    image_bytes = e.partial

                if not image_bytes:
                    return False, "下载的图片数据为空"
                base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
                return True, base64_encoded
            else:
                return False, f"下载失败 (状态: {response.status})"
    except Exception as e:
        return False, str(e)


def get_image_mime_type(base64_data: str) -> str:
    """
    根据 base64 数据判断图片 MIME 类型
    
    Args:
        base64_data: base64 编码的图片数据
        
    Returns:
        str: MIME 类型
    """
    if base64_data.startswith("iVBORw"):
        return "image/png"
    elif base64_data.startswith("/9j/"):
        return "image/jpeg"
    elif base64_data.startswith("UklGR"):
        return "image/webp"
    elif base64_data.startswith("R0lGOD"):
        return "image/gif"
    else:
        return "image/png"  # 默认


def _looks_like_image_bytes(data: bytes) -> bool:
    if not data:
        return False
    return (
        data.startswith(b"\x89PNG\r\n\x1a\n")
        or data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"RIFF") and b"WEBP" in data[:16]
        or data.startswith(b"GIF87a")
        or data.startswith(b"GIF89a")
    )


def _looks_like_image_url(url: str) -> bool:
    if not url:
        return False
    lower = url.lower()
    if any(lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]):
        return True
    if "gradio_api/file=" in lower or "/image/" in lower or "image_url" in lower:
        return True
    if lower.endswith(".css") or lower.endswith(".js") or "bootstrap" in lower:
        return False
    return False


def _probe_url_is_image(url: str, timeout: int = 20) -> bool:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "image/*,*/*;q=0.8"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if content_type.startswith("image/"):
                return True
            head = resp.read(32)
            return _looks_like_image_bytes(head)
    except Exception:
        return False


def _normalize_url_for_request(url: str) -> str:
    """
    将可能包含空格/中文/未转义字符的 URL 规范化为可请求格式。
    重点处理 query 参数，避免 InvalidURL(control characters)。
    """
    parts = urllib.parse.urlsplit(url)
    if not parts.query:
        return url

    normalized_items = []
    for item in parts.query.split("&"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            key, value = item, ""
        # 先反解再重编码，避免重复编码和非法字符
        key_decoded = urllib.parse.unquote(key)
        value_decoded = urllib.parse.unquote(value)
        key_encoded = urllib.parse.quote(key_decoded, safe="._-")
        value_encoded = urllib.parse.quote(value_decoded, safe="._-")
        normalized_items.append(f"{key_encoded}={value_encoded}")

    normalized_query = "&".join(normalized_items)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, normalized_query, parts.fragment)
    )

logger = get_logger("MaiBot_LLM2pic")



def _compress_image_if_needed(image_base64: str) -> str:
    """将 PNG 转为 JPEG 以减小体积，避免 QQ 发送大图失败。"""
    # 只处理 PNG（iVBORw 开头），JPEG 已经够小
    if not image_base64.startswith("iVBORw"):
        return image_base64
    try:
        from io import BytesIO
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        image_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_bytes))
        img.load()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)
        compressed = base64.b64encode(buf.getvalue()).decode("utf-8")
        logger.info(f"[LLM2pic] PNG→JPEG: {len(image_base64)//1024}KB -> {len(compressed)//1024}KB")
        return compressed
    except ImportError:
        logger.warning("[LLM2pic] Pillow 未安装，无法转换图片格式")
        return image_base64
    except Exception as exc:
        logger.warning(f"[LLM2pic] PNG→JPEG 转换失败: {exc}")
        return image_base64


def _resize_image_for_edit(image_base64: str, max_pixels: int = 4_000_000) -> str:
    """缩放图片使总像素不超过限制，避免 API 拒绝过大图片。

    Args:
        image_base64: 原始 base64 图片数据
        max_pixels: 最大像素数（默认 4MP，适合大多数图片编辑 API）

    Returns:
        str: 处理后的 base64 图片数据（JPEG 格式）
    """
    try:
        from io import BytesIO
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        image_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_bytes))
        img.load()

        w, h = img.size
        total_pixels = w * h

        if total_pixels <= max_pixels:
            # 不需要缩放，但统一转为 JPEG 减小体积
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        # 等比缩放
        scale = (max_pixels / total_pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        result = base64.b64encode(buf.getvalue()).decode("utf-8")
        logger.info(f"[LLM2pic] 图片缩放: {w}x{h} -> {new_w}x{new_h} ({len(image_base64)//1024}KB -> {len(result)//1024}KB)")
        return result
    except ImportError:
        logger.warning("[LLM2pic] Pillow 未安装，无法缩放图片")
        return image_base64
    except Exception as exc:
        logger.warning(f"[LLM2pic] 图片缩放失败: {exc}")
        return image_base64


def _peel_envelope(payload: Any) -> Any:
    """剥离 SDK/Runner 常见的 result/data 包装层。"""
    current = payload
    visited: set[int] = set()
    while isinstance(current, dict):
        current_id = id(current)
        if current_id in visited:
            break
        visited.add(current_id)

        for wrapper_key in ("result", "data"):
            nested = current.get(wrapper_key)
            if isinstance(nested, dict):
                current = nested
                break
        else:
            return current
    return current


def _normalize_bool(value: Any) -> bool:
    """兼容表单/动作参数中字符串布尔值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


# ===== 默认提示词模板 =====

DEFAULT_SYSTEM_PROMPT = """你是一位专业的AI绘画提示词生成专家。你的任务是根据用户的请求和聊天上下文，生成高质量的英文图片生成提示词，并判断适合的绘画风格。

## 你的角色设定
{persona}

## 输出格式
你必须以 JSON 格式输出，包含两个字段：
- prompt: 英文图片生成提示词
- style: 绘画风格，只能是 "anime"（二次元/动漫风格）或 "edit"（图片编辑/改图风格）

## 风格判断规则
- anime（二次元）：动漫角色、游戏角色、虚拟人物、可爱风格、日系风格、卡通风格、写实场景、自然风景、真实物品等所有纯文生图需求
- edit（图片编辑）：用户发送了图片并要求对图片进行修改、编辑、重绘、风格转换等操作

## 提示词规则
1. 使用逗号分隔的英文关键词格式
2. 关键词顺序：人物/主体 -> 外貌特征 -> 服装 -> 动作/姿势 -> 表情 -> 背景/场景
3. 对于角色请求，使用角色的罗马音名称并补充作品名称，如 rem (re zero)
4. 单人构图时添加 solo 标签
5. 不要添加质量词如 masterpiece, best quality 等（系统会自动添加）
6. 不要添加任何NSFW内容
7. 对于 edit 风格，prompt 应描述用户希望图片变成什么样子

## 自拍模式
当用户要求自拍时，你需要以你的角色身份生成自拍照的提示词，包含：
- 你的外貌特征（根据角色设定）
- selfie, front-facing camera, close-up shot
- 自然的表情和姿势
- 适合自拍的背景
- 自拍模式下风格应为 anime

## 示例
用户请求: "画一个女孩在雨中"
输出: {{"prompt": "1girl, solo, standing in rain, wet hair, wet clothes, sad expression, rainy day, city street background", "style": "anime"}}

用户请求: [发送了一张图片] "把这张图变成动漫风格"
输出: {{"prompt": "convert to anime style, vibrant colors, cel shading, anime aesthetic", "style": "edit"}}

用户请求: "自拍"
输出: {{"prompt": "1girl, solo, selfie, front-facing camera, close-up shot, smile, casual clothes, indoor background", "style": "anime"}}"""


# ===== 风格路由器 =====

class StyleRouter:
    """风格路由器，根据各种条件决定使用哪个模型"""
    
    VALID_STYLES = ["anime", "edit"]
    
    def __init__(self, config: dict):
        """
        初始化风格路由器
        
        Args:
            config: 插件配置字典
        """
        self.config = config
        self.default_style = config.get("generation", {}).get("default_style", "anime")
        self.anime_config = self._extract_model_config("anime")
        self.edit_config = self._extract_model_config("edit")
        
        logger.debug(f"[StyleRouter] 初始化完成: default_style={self.default_style}, "
                    f"anime_enabled={self.anime_config is not None}, "
                    f"edit_enabled={self.edit_config is not None}")
    
    def _extract_model_config(self, style: str) -> Optional[dict]:
        """
        提取指定风格的模型配置
        
        Args:
            style: 风格名称 ("anime" 或 "edit")
            
        Returns:
            Optional[dict]: 模型配置字典，如果未启用则返回 None
        """
        style_config = self.config.get(style, {})
        
        # 检查是否启用
        if not style_config.get("enabled", False):
            return None
        
        return {
            "api_type": style_config.get("api_type", "openai"),
            "base_url": style_config.get("base_url", ""),
            "api_key": style_config.get("api_key", ""),
            "model_name": style_config.get("model_name", ""),
            "custom_prompt_add": style_config.get("custom_prompt_add", ""),
            "gradio_resolution": style_config.get("gradio_resolution", "1024x1024 ( 1:1 )"),
            "gradio_steps": style_config.get("gradio_steps", 8),
            "gradio_shift": style_config.get("gradio_shift", 3),
            "gradio_timeout": style_config.get("gradio_timeout", 120),
            # SD API 参数
            "sd_negative_prompt": style_config.get("sd_negative_prompt", ""),
            "sd_width": style_config.get("sd_width", 512),
            "sd_height": style_config.get("sd_height", 512),
            "sd_steps": style_config.get("sd_steps", 20),
            "sd_cfg": style_config.get("sd_cfg", 7.0),
            "sd_model_index": style_config.get("sd_model_index", 0),
            "sd_seed": style_config.get("sd_seed", -1),
            # NovelAI API 参数
            "novelai_model": style_config.get("novelai_model", "nai-diffusion-4-5-full"),
            "novelai_width": style_config.get("novelai_width", 832),
            "novelai_height": style_config.get("novelai_height", 1216),
            "novelai_steps": style_config.get("novelai_steps", 28),
            "novelai_scale": style_config.get("novelai_scale", 5.0),
            "novelai_sampler": style_config.get("novelai_sampler", "k_euler"),
            "novelai_negative_prompt": style_config.get("novelai_negative_prompt", ""),
            "novelai_seed": style_config.get("novelai_seed", -1),
            "novelai_timeout": style_config.get("novelai_timeout", 120),
        }
    
    def route(
        self,
        selfie_mode: bool = False,
        manual_style: Optional[str] = None,
        llm_style: Optional[str] = None
    ) -> Tuple[str, Optional[dict], str]:
        """
        决定使用哪个模型
        
        优先级：selfie_mode > manual_style > llm_style > default_style
        
        Args:
            selfie_mode: 是否为自拍模式
            manual_style: 手动指定的风格
            llm_style: LLM 判断的风格
            
        Returns:
            Tuple[str, Optional[dict], str]: (style, model_config, reason)
            - style: 选择的风格
            - model_config: 对应的模型配置，如果未配置则为 None
            - reason: 路由原因，用于日志
        """
        # 向后兼容：如果只配置了一个模型，所有请求都使用该模型
        available_configs = []
        if self.anime_config:
            available_configs.append(("anime", self.anime_config))
        if self.edit_config:
            available_configs.append(("edit", self.edit_config))
        
        if len(available_configs) == 0:
            logger.warning("[StyleRouter] 没有配置任何模型")
            return self.default_style, None, "no_model_configured"
        
        if len(available_configs) == 1:
            only_style, only_config = available_configs[0]
            logger.debug(f"[StyleRouter] 只配置了 {only_style} 模型，使用该模型")
            return only_style, only_config, f"only_{only_style}_configured"
        
        # 1. selfie_mode 强制使用 anime
        if selfie_mode:
            if self.anime_config:
                logger.debug("[StyleRouter] selfie_mode=True，使用 anime 模型")
                return "anime", self.anime_config, "selfie_mode"
            else:
                # anime 未配置，回退到 edit
                logger.warning("[StyleRouter] selfie_mode=True 但 anime 未配置，回退到 edit")
                return "edit", self.edit_config, "selfie_mode_fallback_to_edit"
        
        # 2. 手动指定风格
        if manual_style:
            manual_style_lower = manual_style.lower().strip()
            if manual_style_lower in self.VALID_STYLES:
                config = self.anime_config if manual_style_lower == "anime" else self.edit_config
                if config:
                    logger.debug(f"[StyleRouter] 手动指定风格: {manual_style_lower}")
                    return manual_style_lower, config, "manual_style"
                else:
                    logger.warning(f"[StyleRouter] 手动指定的风格 {manual_style_lower} 未配置")
                    # 回退到另一个可用的模型
                    fallback_style, fallback_config = available_configs[0]
                    return fallback_style, fallback_config, f"manual_style_{manual_style_lower}_not_configured"
        
        # 3. LLM 判断的风格
        if llm_style:
            llm_style_lower = llm_style.lower().strip()
            if llm_style_lower in self.VALID_STYLES:
                config = self.anime_config if llm_style_lower == "anime" else self.edit_config
                if config:
                    logger.debug(f"[StyleRouter] LLM 判断风格: {llm_style_lower}")
                    return llm_style_lower, config, "llm_style"
                else:
                    logger.warning(f"[StyleRouter] LLM 判断的风格 {llm_style_lower} 未配置，使用默认风格")
        
        # 4. 使用默认风格
        default_config = self.anime_config if self.default_style == "anime" else self.edit_config
        if default_config:
            logger.debug(f"[StyleRouter] 使用默认风格: {self.default_style}")
            return self.default_style, default_config, "default_style"
        
        # 默认风格未配置，使用另一个可用的
        fallback_style, fallback_config = available_configs[0]
        logger.warning(f"[StyleRouter] 默认风格 {self.default_style} 未配置，回退到 {fallback_style}")
        return fallback_style, fallback_config, "default_style_fallback"
    
    def is_style_available(self, style: str) -> bool:
        """检查指定风格是否可用"""
        if style == "anime":
            return self.anime_config is not None
        elif style == "edit":
            return self.edit_config is not None
        return False
    
    def get_available_styles(self) -> List[str]:
        """获取所有可用的风格列表"""
        styles = []
        if self.anime_config:
            styles.append("anime")
        if self.edit_config:
            styles.append("edit")
        return styles


# ===== LLM 输出解析器 =====

class LLMOutputParser:
    """LLM 输出解析器，用于解析 JSON 格式的 LLM 输出"""
    
    VALID_STYLES = ["anime", "edit"]
    
    @staticmethod
    def parse(llm_output: str, default_style: str = "anime") -> Tuple[str, Optional[str]]:
        """
        解析 LLM 输出
        
        Args:
            llm_output: LLM 的原始输出
            default_style: 默认风格（当无法解析时使用）
            
        Returns:
            Tuple[str, Optional[str]]: (prompt, style)
            - 如果解析成功，返回提取的 prompt 和 style
            - 如果解析失败，返回原始输出作为 prompt，style 为 None
        """
        if not llm_output:
            return "", None
        
        llm_output = llm_output.strip()
        
        # 尝试解析 JSON
        try:
            # 尝试直接解析
            data = json.loads(llm_output)
            
            # 确保解析结果是字典
            if not isinstance(data, dict):
                logger.warning(f"[LLMOutputParser] JSON 解析结果不是字典，使用原始输出")
                return llm_output, None
            
            prompt = data.get("prompt", "")
            style = data.get("style", None)
            
            # 验证 style
            validated_style = LLMOutputParser.validate_style(style)
            
            logger.debug(f"[LLMOutputParser] JSON 解析成功: prompt={prompt[:50]}..., style={validated_style}")
            return prompt, validated_style
            
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON
            json_match = LLMOutputParser._extract_json_from_text(llm_output)
            if json_match:
                try:
                    data = json.loads(json_match)
                    
                    # 确保解析结果是字典
                    if not isinstance(data, dict):
                        logger.warning(f"[LLMOutputParser] 提取的 JSON 不是字典，使用原始输出")
                        return llm_output, None
                    
                    prompt = data.get("prompt", "")
                    style = data.get("style", None)
                    validated_style = LLMOutputParser.validate_style(style)
                    logger.debug(f"[LLMOutputParser] 从文本中提取 JSON 成功: prompt={prompt[:50]}..., style={validated_style}")
                    return prompt, validated_style
                except json.JSONDecodeError:
                    pass
            
            # 回退：使用整个输出作为 prompt
            logger.warning(f"[LLMOutputParser] LLM 输出不是有效 JSON，使用原始输出作为 prompt")
            return llm_output, None
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[str]:
        """从文本中提取 JSON 字符串"""
        # 查找 { 和 } 的位置
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return None
    
    @staticmethod
    def validate_style(style: Optional[str]) -> Optional[str]:
        """
        验证风格是否有效
        
        Args:
            style: 待验证的风格
            
        Returns:
            Optional[str]: 有效的风格或 None
        """
        if style is None:
            return None
        
        style_lower = style.lower().strip()
        if style_lower in LLMOutputParser.VALID_STYLES:
            return style_lower
        
        logger.warning(f"[LLMOutputParser] 无效的风格 '{style}'，返回 None")
        return None


# ===== Action组件 =====

class CustomPicAction:
    """生成一张图片并发送"""

    # 激活设置
    parallel_action = False

    # 动作基本信息
    action_name = "draw_picture"
    action_description = "生成并发送图片（纯文生图）。触发场景：用户让你画图/发图/自拍、要求看你长什么样/在干嘛、说\"画一张...\"\"给我看看...\"。当你判断需要用图而非纯文字回应时，应立即调用此工具。注意：如果用户发送了图片并要求修改/编辑，请使用 edit_picture 工具。"

    # LLM判定提示词
    llm_judge_prompt = """
此动作让你能够从零生成并发送图片，用于回应群友想要"看到"某些视觉内容的请求。仅用于文生图。

【触发条件】当群友想要看到以下内容时使用：
1. 你当前的状态/环境/正在做的事（自拍、你在哪、你在干嘛）
2. 你拍的照片/摄影作品（发张你拍的照片、看看你的摄影）
3. 你正在吃/喝/用的东西（你在吃什么、给我看看）
4. 你画的画/创作的图（画一张、帮我画个）
5. 某个具体场景/角色/事物的图片（我想看看...的样子）

【典型触发语句示例】
- "自拍/来张自拍/发张照片看看"
- "你现在在哪/在干嘛，发张图看看"
- "画一张.../帮我画个..."
- "我想看看...长什么样"

【禁止触发】
- 纯文字聊天、问答、讨论（不涉及"看图"需求）
- 只是提到图片相关词汇但不是要求生成
- 讨论或评价已经存在的图片（不要求修改）
- 用户明确表示不需要图片
- 并不是对你提出的看图需求
- 前面聊天记录中你已经发过图片时，禁止再次生成并发送图片
- 用户发送了图片并要求修改/编辑/重绘/换风格（应使用 edit_picture）
"""

    # 动作参数定义
    action_parameters = {
        "description": "用户想要生成的图片描述，可以是中文或英文，系统会自动处理",
        "selfie_mode": "是否生成自拍模式的图片，设置为true时会以角色身份生成自拍，默认为false",
    }

    # 动作使用场景
    action_require = [
        "当有人让你画一张图时使用",
        "当有人要求生成自拍照片时使用，设置selfie_mode为true",
        "如果最近的消息内你发过图片请不要选择此动作",
        "如果用户发送了图片并要求修改，请使用 edit_picture 而非此工具",
    ]
    associated_types = ["text"]

    async def _extract_input_image(self) -> Optional[str]:
        """
        从当前消息中提取图片，用于图生图
        
        Returns:
            Optional[str]: 图片的 base64 数据，如果没有图片则返回 None
        """
        try:
            # 尝试从消息中获取图片
            if hasattr(self, 'message') and self.message:
                # 检查消息段中是否有图片
                if hasattr(self.message, 'message_segment'):
                    for seg in self.message.message_segment:
                        if hasattr(seg, 'type') and seg.type == 'image':
                            # 获取图片 URL 或 base64
                            if hasattr(seg, 'data'):
                                img_data = seg.data
                                # 如果是 URL，下载并转换
                                if isinstance(img_data, dict):
                                    img_url = img_data.get('url') or img_data.get('file')
                                    if img_url:
                                        if img_url.startswith('http'):
                                            success, result = await asyncio.to_thread(
                                                download_image_to_base64, img_url
                                            )
                                            if success:
                                                return result
                                        elif img_url.startswith('base64://'):
                                            return img_url[9:]  # 去掉 base64:// 前缀
                                # 如果直接是 base64
                                elif isinstance(img_data, str):
                                    if img_data.startswith('base64://'):
                                        return img_data[9:]
                                    elif img_data.startswith(('iVBORw', '/9j/', 'UklGR', 'R0lGOD')):
                                        return img_data
                
                # 尝试从 raw_message 中提取图片 URL
                if hasattr(self.message, 'raw_message'):
                    import re
                    # 匹配 CQ 码中的图片
                    cq_pattern = r'\[CQ:image[^\]]*url=([^\],]+)'
                    matches = re.findall(cq_pattern, str(self.message.raw_message))
                    if matches:
                        img_url = matches[0]
                        success, result = await asyncio.to_thread(
                            download_image_to_base64, img_url
                        )
                        if success:
                            return result
                    
                    # 匹配普通图片 URL
                    url_pattern = r'https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp)'
                    url_matches = re.findall(url_pattern, str(self.message.raw_message), re.IGNORECASE)
                    if url_matches:
                        success, result = await asyncio.to_thread(
                            download_image_to_base64, url_matches[0]
                        )
                        if success:
                            return result
            
            logger.debug(f"{self.log_prefix} 消息中未找到图片")
            return None
            
        except Exception as e:
            logger.error(f"{self.log_prefix} 提取输入图片失败: {e}", exc_info=True)
            return None

    async def _get_recent_chat_messages(self) -> str:
        """获取最近的聊天记录"""
        try:
            # 从配置获取参数
            message_limit = self.get_config("llm.context_message_limit", 20)
            time_minutes = self.get_config("llm.context_time_minutes", 30)
            
            # 限制范围
            message_limit = max(1, min(100, message_limit))
            time_minutes = max(1, min(1440, time_minutes))  # 最多24小时
            
            end_time = time.time()
            start_time = end_time - time_minutes * 60
            
            messages = message_api.get_messages_by_time_in_chat(
                chat_id=self.chat_id,
                start_time=start_time,
                end_time=end_time,
                limit=message_limit,
                limit_mode="latest",
                filter_mai=False,
                filter_command=True,
            )
            
            if not messages:
                return "（暂无聊天记录）"
            
            # 构建可读的聊天记录
            readable = message_api.build_readable_messages_to_str(
                messages=messages,
                replace_bot_name=True,
                timestamp_mode="relative",
                truncate=True,
            )
            
            return readable if readable else "（暂无聊天记录）"
            
        except Exception as e:
            logger.error(f"{self.log_prefix} 获取聊天记录失败: {e}", exc_info=True)
            return "（获取聊天记录失败）"

    def _get_persona(self) -> str:
        """获取人设信息"""
        try:
            bot_nickname = global_config.bot.nickname
            personality = global_config.personality.personality
            visual_style = global_config.personality.visual_style
            
            persona_parts = [f"你的名字是{bot_nickname}。"]
            
            if personality:
                persona_parts.append(f"你的性格特点：{personality}")
            
            if visual_style:
                persona_parts.append(f"你的外貌特征：{visual_style}")
            
            return "\n".join(persona_parts)
            
        except Exception as e:
            logger.error(f"{self.log_prefix} 获取人设信息失败: {e}", exc_info=True)
            return "你是一个友好的AI助手。"

    def _get_llm_model_config(self):
        """获取LLM模型配置"""
        # 优先使用插件配置的模型
        custom_model_name = self.get_config("llm.model_name", "")
        
        if custom_model_name:
            # 尝试从可用模型中获取
            available_models = llm_api.get_available_models()
            if custom_model_name in available_models:
                logger.info(f"{self.log_prefix} 使用插件配置的模型: {custom_model_name}")
                return available_models[custom_model_name]
        
        # 使用默认的planner模型
        try:
            return model_config.model_task_config.planner
        except Exception:
            # 回退到replyer模型
            return model_config.model_task_config.replyer

    def _build_final_prompt(self, generated_prompt: str, model_config: Optional[dict] = None) -> str:
        """构建最终的图片生成提示词"""
        parts = []
        
        # 优先使用风格特定的附加提示词，否则使用全局附加提示词
        custom_prompt_add = ""
        if model_config and model_config.get("custom_prompt_add"):
            custom_prompt_add = model_config.get("custom_prompt_add", "")
        else:
            custom_prompt_add = self.get_config("generation.custom_prompt_add", "")
        
        if custom_prompt_add:
            parts.append(custom_prompt_add)
        
        # 添加LLM生成的提示词
        if generated_prompt:
            parts.append(generated_prompt)
        
        # 合并并去重
        final_prompt = ", ".join(part.strip().strip(",") for part in parts if part and part.strip())
        return self._remove_duplicate_keywords(final_prompt)

    def _remove_duplicate_keywords(self, prompt: str) -> str:
        """删除提示词中的重复关键词"""
        if not prompt or not prompt.strip():
            return prompt
        
        keywords = [kw.strip() for kw in prompt.split(',') if kw.strip()]
        seen = set()
        unique_keywords = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in seen:
                seen.add(keyword_lower)
                unique_keywords.append(keyword)
        
        return ', '.join(unique_keywords)

    async def _handle_image_result(self, result: str) -> Tuple[bool, str]:
        """处理图片生成结果"""
        # 检查是否是Base64数据
        if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
            # 检查是否需要裁切
            crop_enabled = self.get_config("generation.crop_enabled", False)
            if crop_enabled:
                try:
                    image_bytes = base64.b64decode(result)
                    image_bytes = self._crop_image(image_bytes)
                    result = base64.b64encode(image_bytes).decode("utf-8")
                except Exception as e:
                    logger.error(f"{self.log_prefix} Base64图片裁切失败: {e}")
            
            send_success = await self.send_image(result)
            if send_success:
                logger.info(f"{self.log_prefix} 图片已发送")
                return True, "图片已发送"
            else:
                logger.error(f"{self.log_prefix} 图片生成成功但发送失败")
                return False, "图片发送失败"
        else:
            # 是URL，需要下载
            image_url = result
            logger.info(f"{self.log_prefix} 下载图片: {image_url[:70]}...")
            
            try:
                encode_success, encode_result = await asyncio.to_thread(
                    self._download_and_encode_base64, image_url
                )
            except Exception as e:
                logger.error(f"{self.log_prefix} 下载图片失败: {e!r}", exc_info=True)
                encode_success = False
                encode_result = str(e)

            if encode_success:
                send_success = await self.send_image(encode_result)
                if send_success:
                    logger.info(f"{self.log_prefix} 图片已发送")
                    return True, "图片已发送"
                else:
                    logger.error(f"{self.log_prefix} 图片下载成功但发送失败")
                    return False, "图片发送失败"
            else:
                logger.error(f"{self.log_prefix} 下载图片失败: {encode_result}")
                return False, f"图片下载失败: {encode_result}"

    def _download_and_encode_base64(self, image_url: str) -> Tuple[bool, str]:
        """下载图片并编码为Base64，容忍 IncompleteRead"""
        from http.client import IncompleteRead

        try:
            req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status == 200:
                    try:
                        image_bytes = response.read()
                    except IncompleteRead as e:
                        logger.warning(f"{self.log_prefix} 下载图片 IncompleteRead，使用已读取的 {len(e.partial)} bytes")
                        image_bytes = e.partial

                    if not image_bytes:
                        return False, "下载的图片数据为空"

                    # 检查是否需要裁切
                    crop_enabled = self.get_config("generation.crop_enabled", False)
                    if crop_enabled:
                        image_bytes = self._crop_image(image_bytes)

                    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
                    return True, base64_encoded
                else:
                    return False, f"下载失败 (状态: {response.status})"
        except Exception as e:
            logger.error(f"{self.log_prefix} 下载图片错误: {e!r}", exc_info=True)
            return False, str(e)

    def _crop_image(self, image_bytes: bytes) -> bytes:
        """根据配置裁切图片"""
        try:
            from io import BytesIO
            from PIL import Image
            
            crop_position = self.get_config("generation.crop_position", "bottom")
            crop_pixels = self.get_config("generation.crop_pixels", 40)
            
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            
            # 根据位置计算裁切区域
            if crop_position == "bottom":
                if crop_pixels >= height:
                    logger.warning(f"{self.log_prefix} 裁切像素({crop_pixels})大于等于图片高度({height})，跳过裁切")
                    return image_bytes
                crop_box = (0, 0, width, height - crop_pixels)
            elif crop_position == "top":
                if crop_pixels >= height:
                    logger.warning(f"{self.log_prefix} 裁切像素({crop_pixels})大于等于图片高度({height})，跳过裁切")
                    return image_bytes
                crop_box = (0, crop_pixels, width, height)
            elif crop_position == "left":
                if crop_pixels >= width:
                    logger.warning(f"{self.log_prefix} 裁切像素({crop_pixels})大于等于图片宽度({width})，跳过裁切")
                    return image_bytes
                crop_box = (crop_pixels, 0, width, height)
            elif crop_position == "right":
                if crop_pixels >= width:
                    logger.warning(f"{self.log_prefix} 裁切像素({crop_pixels})大于等于图片宽度({width})，跳过裁切")
                    return image_bytes
                crop_box = (0, 0, width - crop_pixels, height)
            else:
                logger.warning(f"{self.log_prefix} 未知的裁切位置: {crop_position}，跳过裁切")
                return image_bytes
            
            cropped_img = img.crop(crop_box)
            
            # 保存为bytes
            output = BytesIO()
            img_format = img.format or 'PNG'
            cropped_img.save(output, format=img_format)
            
            logger.info(f"{self.log_prefix} 已裁切图片{crop_position} {crop_pixels} 像素")
            return output.getvalue()
            
        except ImportError:
            logger.warning(f"{self.log_prefix} PIL未安装，跳过图片裁切")
            return image_bytes
        except Exception as e:
            logger.error(f"{self.log_prefix} 图片裁切失败: {e}", exc_info=True)
            return image_bytes

    def _make_gradio_image_request(
        self, 
        prompt: str, 
        base_url: Optional[str] = None,
        gradio_params: Optional[dict] = None
    ) -> Tuple[bool, str]:
        """发送Gradio API请求生成图片（如HuggingFace Space）"""
        # 使用传入的参数或从配置获取
        if base_url is None:
            base_url = self.get_config("api.base_url", "")
        
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
        
        # 第一步：POST 请求获取 event_id
        endpoint = f"{base_url.rstrip('/')}/gradio_api/call/generate"
        
        payload = {
            "data": [
                prompt,           # [0] prompt
                resolution,       # [1] resolution
                42,              # [2] seed (固定值，因为会使用random_seed=true)
                steps,           # [3] steps
                shift,           # [4] shift
                True,            # [5] random_seed
                []               # [6] gallery_images
            ]
        }
        
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        
        logger.info(f"{self.log_prefix} 发起Gradio图片请求, Prompt: {prompt[:100]}...")
        
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        
        try:
            # 获取 event_id
            with urllib.request.urlopen(req, timeout=30) as response:
                response_body = response.read().decode("utf-8")
                
                if 200 <= response.status < 300:
                    response_data = json.loads(response_body)
                    event_id = response_data.get("event_id")
                    
                    if not event_id:
                        return False, "未获取到event_id"
                    
                    logger.info(f"{self.log_prefix} 获取到event_id: {event_id}")
                else:
                    return False, f"POST请求失败 (状态码 {response.status})"
            
            # 第二步：GET 请求轮询结果
            result_endpoint = f"{base_url.rstrip('/')}/gradio_api/call/generate/{event_id}"
            
            import time
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    result_req = urllib.request.Request(result_endpoint, method="GET")
                    with urllib.request.urlopen(result_req, timeout=30) as result_response:
                        result_body = result_response.read().decode("utf-8")
                        
                        # 解析 SSE 格式的响应
                        for line in result_body.split('\n'):
                            if line.startswith('event: complete'):
                                # 下一行是数据
                                continue
                            elif line.startswith('data: '):
                                data_str = line[6:]  # 去掉 "data: " 前缀
                                try:
                                    result_data = json.loads(data_str)
                                    
                                    # 提取图片URL
                                    # 格式: [[{"image": {"url": "..."}, ...}], seed_str, seed_int]
                                    if isinstance(result_data, list) and len(result_data) > 0:
                                        gallery = result_data[0]
                                        if isinstance(gallery, list) and len(gallery) > 0:
                                            first_image = gallery[0]
                                            if isinstance(first_image, dict):
                                                image_data = first_image.get("image", {})
                                                image_url = image_data.get("url")
                                                
                                                if image_url:
                                                    logger.info(f"{self.log_prefix} 获取到图片URL")
                                                    return True, image_url
                                
                                except json.JSONDecodeError:
                                    continue
                        
                        # 如果没有找到complete事件，等待后重试
                        time.sleep(2)
                        
                except Exception as e:
                    logger.debug(f"{self.log_prefix} 轮询中: {e}")
                    time.sleep(2)
            
            return False, f"轮询超时（{timeout}秒）"
            
        except Exception as e:
            logger.error(f"{self.log_prefix} Gradio API请求错误: {e!r}", exc_info=True)
            return False, str(e)

    def _make_sd_api_request(
        self,
        prompt: str,
        base_url: str,
        api_key: str,
        sd_params: Optional[dict] = None,
    ) -> Tuple[bool, str]:
        """发送 SD API 请求生成图片"""
        endpoint = f"{base_url.rstrip('/')}/api/v1/generate_image"
        
        # 构建请求参数
        payload = {
            "prompt": prompt,
        }
        
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        
        logger.info(f"{self.log_prefix} 发起SD API图片请求, Prompt: {prompt[:100]}...")
        
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                response_body = response.read().decode("utf-8")
                
                if 200 <= response.status < 300:
                    response_data = json.loads(response_body)
                    
                    # 尝试从响应中提取图片
                    # 常见的响应格式：{"image": "base64..."} 或 {"url": "..."} 或 {"images": [...]}
                    image_data = None
                    
                    if "image" in response_data:
                        image_data = response_data["image"]
                    elif "url" in response_data:
                        image_data = response_data["url"]
                    elif "images" in response_data and response_data["images"]:
                        first_img = response_data["images"][0]
                        if isinstance(first_img, str):
                            image_data = first_img
                        elif isinstance(first_img, dict):
                            image_data = first_img.get("url") or first_img.get("image") or first_img.get("base64")
                    elif "data" in response_data:
                        # 有些 API 返回 {"data": {"image": "..."}} 或 {"data": {"image_url": "..."}}
                        data_obj = response_data["data"]
                        if isinstance(data_obj, dict):
                            image_data = data_obj.get("image") or data_obj.get("url") or data_obj.get("image_url")
                        elif isinstance(data_obj, str):
                            image_data = data_obj
                    
                    if image_data:
                        logger.info(f"{self.log_prefix} SD API 返回图片成功")
                        return True, image_data
                    
                    # 如果响应本身是 base64
                    if isinstance(response_data, str) and response_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                        return True, response_data
                    
                    return False, f"SD API 响应中未找到图片数据: {str(response_data)[:200]}"
                else:
                    return False, f"SD API 请求失败 (状态码 {response.status})"
                    
        except Exception as e:
            logger.error(f"{self.log_prefix} SD API 请求错误: {e!r}", exc_info=True)
            return False, str(e)

    def _make_regex_url_request(self, prompt: str, url_template: str) -> Tuple[bool, str]:
        """通过 URL 模板请求生图接口，使用 $1 占位符填充 prompt"""
        from http.client import IncompleteRead

        if not url_template or not url_template.strip():
            return False, "regex_url 未配置 URL 模板"

        encoded_prompt = urllib.parse.quote(prompt, safe="")
        if "$1" in url_template:
            endpoint = url_template.replace("$1", encoded_prompt)
        else:
            connector = "&" if "?" in url_template else "?"
            endpoint = f"{url_template}{connector}tag={encoded_prompt}"
        endpoint = _normalize_url_for_request(endpoint)

        headers = {
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        }
        logger.info(f"{self.log_prefix} 发起regex_url请求: {endpoint[:300]}")
        req = urllib.request.Request(endpoint, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()
                try:
                    response_body = response.read()
                except IncompleteRead as e:
                    logger.warning(f"{self.log_prefix} IncompleteRead: 已读 {len(e.partial)} bytes")
                    response_body = e.partial

                if not response_body:
                    return False, "regex_url 响应为空"

                # 直接返回图片二进制为base64
                if content_type.startswith("image/") or _looks_like_image_bytes(response_body[:32]):
                    base64_data = base64.b64encode(response_body).decode("utf-8")
                    return True, base64_data

                text = response_body.decode("utf-8", errors="ignore")
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        candidates = []
                        direct_url = data.get("url") or data.get("image_url")
                        if isinstance(direct_url, str):
                            candidates.append(direct_url)
                        if isinstance(data.get("data"), dict):
                            nested_url = data["data"].get("url") or data["data"].get("image_url")
                            if isinstance(nested_url, str):
                                candidates.append(nested_url)
                        if isinstance(data.get("data"), str):
                            candidates.append(data.get("data"))
                        for image_url in candidates:
                            if _looks_like_image_url(image_url) or _probe_url_is_image(image_url):
                                return True, image_url
                except Exception:
                    pass

                urls = re.findall(r'https?://[^\s\)\]\"\'<>]+', text)
                for url in urls:
                    if _looks_like_image_url(url) or _probe_url_is_image(url):
                        return True, url
                return False, f"regex_url 响应中未找到图片数据: {text[:200]}"
        except Exception as e:
            logger.error(f"{self.log_prefix} regex_url 请求错误: {e!r}", exc_info=True)
            return False, str(e)

    def _make_novelai_request(
        self,
        prompt: str,
        api_key: str,
        novelai_params: Optional[dict] = None,
    ) -> Tuple[bool, str]:
        """
        发送 NovelAI 官方 API 请求生成图片
        
        Args:
            prompt: 图片生成提示词
            api_key: NovelAI API token (Bearer token)
            novelai_params: NovelAI 参数配置
                - model: 模型名称 (默认 "nai-diffusion-4-5-full")
                - width: 图片宽度 (默认 832)
                - height: 图片高度 (默认 1216)
                - steps: 推理步数 (默认 28)
                - scale: CFG scale (默认 5.0)
                - sampler: 采样器 (默认 "k_euler")
                - negative_prompt: 负面提示词
                - seed: 随机种子 (-1 为随机)
                - timeout: 请求超时时间（秒）
        
        Returns:
            Tuple[bool, str]: (是否成功, base64编码的图片或错误信息)
        """
        import zipfile
        from io import BytesIO
        
        if novelai_params is None:
            novelai_params = {}
        
        # NovelAI API 端点
        endpoint = "https://image.novelai.net/ai/generate-image"
        
        # 解析参数
        model = novelai_params.get("model", "nai-diffusion-4-5-full")
        width = novelai_params.get("width", 832)
        height = novelai_params.get("height", 1216)
        steps = novelai_params.get("steps", 28)
        scale = novelai_params.get("scale", 5.0)
        sampler = novelai_params.get("sampler", "k_euler")
        negative_prompt = novelai_params.get("negative_prompt", "")
        seed = novelai_params.get("seed", -1)
        timeout = novelai_params.get("timeout", 120)
        
        # 如果 seed 为 -1，生成随机种子
        import random
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # 构建请求体
        payload = {
            "input": prompt,
            "model": model,
            "action": "generate",
            "parameters": {
                "width": width,
                "height": height,
                "scale": scale,
                "sampler": sampler,
                "steps": steps,
                "seed": seed,
                "n_samples": 1,
                "negative_prompt": negative_prompt,
                "noise_schedule": "karras",
                "qualityToggle": True,
                "ucPreset": 0,
            }
        }
        
        # 根据模型调整参数
        if "nai-diffusion-4" in model:
            # NAI v4 模型特定参数
            payload["parameters"]["cfg_rescale"] = 0
            payload["parameters"]["noise_schedule"] = "karras"
        
        data = json.dumps(payload).encode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/zip, image/*",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        
        logger.info(f"{self.log_prefix} 发起 NovelAI 图片请求, model={model}, prompt={prompt[:80]}...")
        
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    response_data = response.read()
                    content_type = response.headers.get("Content-Type", "")
                    
                    # NovelAI 返回 zip 文件，需要解压
                    if "zip" in content_type or response_data[:4] == b'PK\x03\x04':
                        try:
                            with zipfile.ZipFile(BytesIO(response_data)) as zf:
                                # 获取 zip 中的第一个文件（图片）
                                file_list = zf.namelist()
                                if file_list:
                                    image_bytes = zf.read(file_list[0])
                                    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
                                    logger.info(f"{self.log_prefix} NovelAI 请求成功，图片大小: {len(image_bytes)} bytes")
                                    return True, base64_encoded
                                else:
                                    return False, "NovelAI 返回的 zip 文件为空"
                        except zipfile.BadZipFile:
                            return False, "NovelAI 返回的不是有效的 zip 文件"
                    
                    # 如果直接返回图片
                    elif content_type.startswith("image/") or response_data[:8].startswith(b'\x89PNG') or response_data[:2] == b'\xff\xd8':
                        base64_encoded = base64.b64encode(response_data).decode("utf-8")
                        logger.info(f"{self.log_prefix} NovelAI 请求成功，图片大小: {len(response_data)} bytes")
                        return True, base64_encoded
                    
                    else:
                        try:
                            error_text = response_data.decode("utf-8")[:500]
                            return False, f"NovelAI 返回未知格式: {error_text}"
                        except UnicodeDecodeError:
                            return False, f"NovelAI 返回未知格式 (Content-Type: {content_type})"
                else:
                    return False, f"NovelAI 请求失败 (状态码 {response.status})"
                    
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            logger.error(f"{self.log_prefix} NovelAI HTTP 错误: {e.code} - {error_body}")
            
            # 解析常见错误
            if e.code == 401:
                return False, "NovelAI 认证失败，请检查 API token"
            elif e.code == 402:
                return False, "NovelAI 配额不足，请充值 Anlas"
            elif e.code == 429:
                return False, "NovelAI 请求过于频繁，请稍后重试"
            else:
                return False, f"NovelAI HTTP 错误 {e.code}: {error_body}"
        except urllib.error.URLError as e:
            logger.error(f"{self.log_prefix} NovelAI 连接错误: {e.reason}")
            return False, f"连接错误: {e.reason}"
        except Exception as e:
            logger.error(f"{self.log_prefix} NovelAI 请求错误: {e!r}", exc_info=True)
            return False, str(e)

    def _make_http_image_request(
        self, 
        prompt: str, 
        model: str, 
        size: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        input_image_base64: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        发送HTTP请求生成图片（使用chat/completions端点）
        
        Args:
            prompt: 图片生成提示词
            model: 模型名称
            size: 图片尺寸（可选）
            base_url: API 基础 URL
            api_key: API 密钥
            input_image_base64: 输入图片的 base64 数据（用于图生图）
            
        Returns:
            Tuple[bool, str]: (是否成功, 图片URL/base64或错误信息)
        """
        import re
        
        # 使用传入的参数或从配置获取
        if base_url is None:
            base_url = self.get_config("api.base_url", "")
        if api_key is None:
            api_key = self.get_config("api.api_key", "")

        endpoint = f"{base_url.rstrip('/')}/chat/completions"

        # 构建消息内容
        if input_image_base64:
            # 图生图模式：使用多模态消息格式
            mime_type = get_image_mime_type(input_image_base64)
            message_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{input_image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            logger.info(f"{self.log_prefix} 发起图生图请求: {model}, Prompt: {prompt[:100]}...")
        else:
            # 文生图模式：纯文本消息
            message_content = prompt
            logger.info(f"{self.log_prefix} 发起文生图请求: {model}, Prompt: {prompt[:100]}...")

        # 使用chat completions格式
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message_content}],
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                response_body = response.read().decode("utf-8")
                
                if 200 <= response.status < 300:
                    response_data = json.loads(response_body)
                    
                    # 从chat completions响应中提取内容
                    content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # 尝试从content中提取图片URL
                    url_pattern = r'https?://[^\s\)\]\"\'<>]+'
                    urls = re.findall(url_pattern, content)
                    
                    image_url = None
                    for url in urls:
                        if any(ext in url.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']) or 'image' in url.lower():
                            image_url = url
                            break
                    
                    if not image_url and urls:
                        image_url = urls[0]
                    
                    if image_url:
                        return True, image_url
                    
                    # 如果content本身是base64
                    if content.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                        return True, content
                    
                    return False, f"API响应中未找到图片数据: {content[:200]}"
                else:
                    return False, f"API请求失败 (状态码 {response.status})"
                    
        except Exception as e:
            logger.error(f"{self.log_prefix} HTTP请求错误: {e!r}", exc_info=True)
            return False, str(e)


# ===== Command组件 =====

class DirectPicCommand:
    """直接生成图片的指令，跳过LLM提示词生成"""

    command_name = "direct_pic"
    command_description = "直接使用提供的prompt生成图片，不经过LLM处理。支持 /pic anime <prompt> 或 /pic edit <prompt> 指定风格"
    # 支持 /pic <prompt>, /pic anime <prompt>, /pic edit <prompt>
    command_pattern = r"^/pic\s+(?:(?P<style>anime|edit)\s+)?(?P<prompt>.+)$"

    def __init__(self, message, plugin_config: Optional[dict] = None):
        super().__init__(message, plugin_config)
        self.log_prefix = "[DirectPic]"

    @staticmethod
    def parse_style_from_prompt(prompt: str) -> Tuple[Optional[str], str]:
        """
        从 prompt 中解析风格前缀
        
        Args:
            prompt: 原始 prompt
            
        Returns:
            Tuple[Optional[str], str]: (style, remaining_prompt)
        """
        prompt = prompt.strip()
        lower_prompt = prompt.lower()
        
        # 检查是否以 anime 或 edit 开头
        for style in ["anime", "edit"]:
            if lower_prompt.startswith(style + " "):
                remaining = prompt[len(style):].strip()
                return style, remaining
        
        return None, prompt

    async def _extract_input_image(self) -> Optional[str]:
        """
        从当前消息中提取图片，用于图生图
        
        Returns:
            Optional[str]: 图片的 base64 数据，如果没有图片则返回 None
        """
        try:
            # 尝试从消息中获取图片
            if hasattr(self, 'message') and self.message:
                # 检查消息段中是否有图片
                if hasattr(self.message, 'message_segment'):
                    for seg in self.message.message_segment:
                        if hasattr(seg, 'type') and seg.type == 'image':
                            # 获取图片 URL 或 base64
                            if hasattr(seg, 'data'):
                                img_data = seg.data
                                # 如果是 URL，下载并转换
                                if isinstance(img_data, dict):
                                    img_url = img_data.get('url') or img_data.get('file')
                                    if img_url:
                                        if img_url.startswith('http'):
                                            success, result = await asyncio.to_thread(
                                                download_image_to_base64, img_url
                                            )
                                            if success:
                                                return result
                                        elif img_url.startswith('base64://'):
                                            return img_url[9:]  # 去掉 base64:// 前缀
                                # 如果直接是 base64
                                elif isinstance(img_data, str):
                                    if img_data.startswith('base64://'):
                                        return img_data[9:]
                                    elif img_data.startswith(('iVBORw', '/9j/', 'UklGR', 'R0lGOD')):
                                        return img_data
                
                # 尝试从 raw_message 中提取图片 URL
                if hasattr(self.message, 'raw_message'):
                    import re
                    # 匹配 CQ 码中的图片
                    cq_pattern = r'\[CQ:image[^\]]*url=([^\],]+)'
                    matches = re.findall(cq_pattern, str(self.message.raw_message))
                    if matches:
                        img_url = matches[0]
                        success, result = await asyncio.to_thread(
                            download_image_to_base64, img_url
                        )
                        if success:
                            return result
                    
                    # 匹配普通图片 URL
                    url_pattern = r'https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp)'
                    url_matches = re.findall(url_pattern, str(self.message.raw_message), re.IGNORECASE)
                    if url_matches:
                        success, result = await asyncio.to_thread(
                            download_image_to_base64, url_matches[0]
                        )
                        if success:
                            return result
            
            logger.debug(f"{self.log_prefix} 消息中未找到图片")
            return None
            
        except Exception as e:
            logger.error(f"{self.log_prefix} 提取输入图片失败: {e}", exc_info=True)
            return None

    async def _handle_image_result(self, result: str) -> Tuple[bool, Optional[str], bool]:
        """处理图片生成结果"""
        # 检查是否是Base64数据
        if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
            # 检查是否需要裁切
            crop_enabled = self.get_config("generation.crop_enabled", False)
            if crop_enabled:
                try:
                    image_bytes = base64.b64decode(result)
                    image_bytes = self._crop_image(image_bytes)
                    result = base64.b64encode(image_bytes).decode("utf-8")
                except Exception as e:
                    logger.error(f"{self.log_prefix} Base64图片裁切失败: {e}")
            
            send_success = await self.send_image(result)
            if send_success:
                logger.info(f"{self.log_prefix} 图片已发送")
                return True, None, True
            else:
                logger.error(f"{self.log_prefix} 图片生成成功但发送失败")
                await self.send_text("图片发送失败")
                return False, None, True
        else:
            # 是URL，需要下载
            image_url = result
            logger.info(f"{self.log_prefix} 下载图片: {image_url[:70]}...")
            
            try:
                encode_success, encode_result = await asyncio.to_thread(
                    self._download_and_encode_base64, image_url
                )
            except Exception as e:
                logger.error(f"{self.log_prefix} 下载图片失败: {e!r}", exc_info=True)
                encode_success = False
                encode_result = str(e)

            if encode_success:
                send_success = await self.send_image(encode_result)
                if send_success:
                    logger.info(f"{self.log_prefix} 图片已发送")
                    return True, None, True
                else:
                    logger.error(f"{self.log_prefix} 图片下载成功但发送失败")
                    await self.send_text("图片发送失败")
                    return False, None, True
            else:
                logger.error(f"{self.log_prefix} 下载图片失败: {encode_result}")
                await self.send_text(f"图片下载失败: {encode_result}")
                return False, None, True

    def _download_and_encode_base64(self, image_url: str) -> Tuple[bool, str]:
        """下载图片并编码为Base64，容忍 IncompleteRead"""
        from http.client import IncompleteRead

        try:
            req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status == 200:
                    try:
                        image_bytes = response.read()
                    except IncompleteRead as e:
                        logger.warning(f"{self.log_prefix} 下载图片 IncompleteRead，使用已读取的 {len(e.partial)} bytes")
                        image_bytes = e.partial

                    if not image_bytes:
                        return False, "下载的图片数据为空"

                    crop_enabled = self.get_config("generation.crop_enabled", False)
                    if crop_enabled:
                        image_bytes = self._crop_image(image_bytes)

                    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
                    return True, base64_encoded
                else:
                    return False, f"下载失败 (状态: {response.status})"
        except Exception as e:
            logger.error(f"{self.log_prefix} 下载图片错误: {e!r}", exc_info=True)
            return False, str(e)

    def _crop_image(self, image_bytes: bytes) -> bytes:
        """根据配置裁切图片"""
        try:
            from io import BytesIO
            from PIL import Image
            
            crop_position = self.get_config("generation.crop_position", "bottom")
            crop_pixels = self.get_config("generation.crop_pixels", 40)
            
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
            img_format = img.format or 'PNG'
            cropped_img.save(output, format=img_format)
            
            logger.info(f"{self.log_prefix} 已裁切图片{crop_position} {crop_pixels} 像素")
            return output.getvalue()
            
        except ImportError:
            return image_bytes
        except Exception as e:
            logger.error(f"{self.log_prefix} 图片裁切失败: {e}", exc_info=True)
            return image_bytes

    def _make_gradio_image_request(self, prompt: str) -> Tuple[bool, str]:
        """发送Gradio API请求生成图片"""
        base_url = self.get_config("api.base_url", "")
        resolution = self.get_config("generation.gradio_resolution", "1024x1024 ( 1:1 )")
        steps = self.get_config("generation.gradio_steps", 8)
        shift = self.get_config("generation.gradio_shift", 3)
        timeout = self.get_config("generation.gradio_timeout", 120)
        
        endpoint = f"{base_url.rstrip('/')}/gradio_api/call/generate"
        
        payload = {
            "data": [prompt, resolution, 42, steps, shift, True, []]
        }
        
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        
        logger.info(f"{self.log_prefix} 发起Gradio图片请求, Prompt: {prompt[:100]}...")
        
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                response_body = response.read().decode("utf-8")
                
                if 200 <= response.status < 300:
                    response_data = json.loads(response_body)
                    event_id = response_data.get("event_id")
                    
                    if not event_id:
                        return False, "未获取到event_id"
                    
                    logger.info(f"{self.log_prefix} 获取到event_id: {event_id}")
                else:
                    return False, f"POST请求失败 (状态码 {response.status})"
            
            result_endpoint = f"{base_url.rstrip('/')}/gradio_api/call/generate/{event_id}"
            
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    result_req = urllib.request.Request(result_endpoint, method="GET")
                    with urllib.request.urlopen(result_req, timeout=30) as result_response:
                        result_body = result_response.read().decode("utf-8")
                        
                        for line in result_body.split('\n'):
                            if line.startswith('data: '):
                                data_str = line[6:]
                                try:
                                    result_data = json.loads(data_str)
                                    
                                    if isinstance(result_data, list) and len(result_data) > 0:
                                        gallery = result_data[0]
                                        if isinstance(gallery, list) and len(gallery) > 0:
                                            first_image = gallery[0]
                                            if isinstance(first_image, dict):
                                                image_data = first_image.get("image", {})
                                                image_url = image_data.get("url")
                                                
                                                if image_url:
                                                    logger.info(f"{self.log_prefix} 获取到图片URL")
                                                    return True, image_url
                                
                                except json.JSONDecodeError:
                                    continue
                        
                        time.sleep(2)
                        
                except Exception as e:
                    logger.debug(f"{self.log_prefix} 轮询中: {e}")
                    time.sleep(2)
            
            return False, f"轮询超时（{timeout}秒）"
            
        except Exception as e:
            logger.error(f"{self.log_prefix} Gradio API请求错误: {e!r}", exc_info=True)
            return False, str(e)

    def _make_sd_api_request(
        self,
        prompt: str,
        base_url: str,
        api_key: str,
        sd_params: Optional[dict] = None,
    ) -> Tuple[bool, str]:
        """发送 SD API 请求生成图片"""
        endpoint = f"{base_url.rstrip('/')}/api/v1/generate_image"
        
        # 构建请求参数
        payload = {
            "prompt": prompt,
        }
        
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        
        logger.info(f"{self.log_prefix} 发起SD API图片请求, Prompt: {prompt[:100]}...")
        
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                response_body = response.read().decode("utf-8")
                
                if 200 <= response.status < 300:
                    response_data = json.loads(response_body)
                    
                    # 尝试从响应中提取图片
                    image_data = None
                    
                    if "image" in response_data:
                        image_data = response_data["image"]
                    elif "url" in response_data:
                        image_data = response_data["url"]
                    elif "images" in response_data and response_data["images"]:
                        first_img = response_data["images"][0]
                        if isinstance(first_img, str):
                            image_data = first_img
                        elif isinstance(first_img, dict):
                            image_data = first_img.get("url") or first_img.get("image") or first_img.get("base64")
                    elif "data" in response_data:
                        data_obj = response_data["data"]
                        if isinstance(data_obj, dict):
                            image_data = data_obj.get("image") or data_obj.get("url") or data_obj.get("image_url")
                        elif isinstance(data_obj, str):
                            image_data = data_obj
                    
                    if image_data:
                        logger.info(f"{self.log_prefix} SD API 返回图片成功")
                        return True, image_data
                    
                    if isinstance(response_data, str) and response_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                        return True, response_data
                    
                    return False, f"SD API 响应中未找到图片数据: {str(response_data)[:200]}"
                else:
                    return False, f"SD API 请求失败 (状态码 {response.status})"
                    
        except Exception as e:
            logger.error(f"{self.log_prefix} SD API 请求错误: {e!r}", exc_info=True)
            return False, str(e)

    def _make_regex_url_request(self, prompt: str, url_template: str) -> Tuple[bool, str]:
        """通过 URL 模板请求生图接口，使用 $1 占位符填充 prompt"""
        from http.client import IncompleteRead

        if not url_template or not url_template.strip():
            return False, "regex_url 未配置 URL 模板"

        encoded_prompt = urllib.parse.quote(prompt, safe="")
        if "$1" in url_template:
            endpoint = url_template.replace("$1", encoded_prompt)
        else:
            connector = "&" if "?" in url_template else "?"
            endpoint = f"{url_template}{connector}tag={encoded_prompt}"
        endpoint = _normalize_url_for_request(endpoint)

        headers = {
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        }
        logger.info(f"{self.log_prefix} 发起regex_url请求: {endpoint[:300]}")
        req = urllib.request.Request(endpoint, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()
                try:
                    response_body = response.read()
                except IncompleteRead as e:
                    logger.warning(f"{self.log_prefix} IncompleteRead: 已读 {len(e.partial)} bytes")
                    response_body = e.partial

                if not response_body:
                    return False, "regex_url 响应为空"

                if content_type.startswith("image/") or _looks_like_image_bytes(response_body[:32]):
                    base64_data = base64.b64encode(response_body).decode("utf-8")
                    return True, base64_data

                text = response_body.decode("utf-8", errors="ignore")
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        candidates = []
                        direct_url = data.get("url") or data.get("image_url")
                        if isinstance(direct_url, str):
                            candidates.append(direct_url)
                        if isinstance(data.get("data"), dict):
                            nested_url = data["data"].get("url") or data["data"].get("image_url")
                            if isinstance(nested_url, str):
                                candidates.append(nested_url)
                        if isinstance(data.get("data"), str):
                            candidates.append(data.get("data"))
                        for image_url in candidates:
                            if _looks_like_image_url(image_url) or _probe_url_is_image(image_url):
                                return True, image_url
                except Exception:
                    pass

                urls = re.findall(r'https?://[^\s\)\]\"\'<>]+', text)
                for url in urls:
                    if _looks_like_image_url(url) or _probe_url_is_image(url):
                        return True, url
                return False, f"regex_url 响应中未找到图片数据: {text[:200]}"
        except Exception as e:
            logger.error(f"{self.log_prefix} regex_url 请求错误: {e!r}", exc_info=True)
            return False, str(e)

    def _make_http_image_request(
        self, 
        prompt: str, 
        model: str, 
        size: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        input_image_base64: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        发送HTTP请求生成图片（使用chat/completions端点）
        
        Args:
            prompt: 图片生成提示词
            model: 模型名称
            size: 图片尺寸（可选）
            base_url: API 基础 URL
            api_key: API 密钥
            input_image_base64: 输入图片的 base64 数据（用于图生图）
            
        Returns:
            Tuple[bool, str]: (是否成功, 图片URL/base64或错误信息)
        """
        import re
        
        if base_url is None:
            base_url = self.get_config("api.base_url", "")
        if api_key is None:
            api_key = self.get_config("api.api_key", "")

        endpoint = f"{base_url.rstrip('/')}/chat/completions"

        # 构建消息内容
        if input_image_base64:
            # 图生图模式：使用多模态消息格式
            mime_type = get_image_mime_type(input_image_base64)
            message_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{input_image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            logger.info(f"{self.log_prefix} 发起图生图请求: {model}, Prompt: {prompt[:100]}...")
        else:
            # 文生图模式：纯文本消息
            message_content = prompt
            logger.info(f"{self.log_prefix} 发起文生图请求: {model}, Prompt: {prompt[:100]}...")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message_content}],
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                response_body = response.read().decode("utf-8")
                
                if 200 <= response.status < 300:
                    response_data = json.loads(response_body)
                    
                    content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    url_pattern = r'https?://[^\s\)\]\"\'<>]+'
                    urls = re.findall(url_pattern, content)
                    
                    image_url = None
                    for url in urls:
                        if any(ext in url.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']) or 'image' in url.lower():
                            image_url = url
                            break
                    
                    if not image_url and urls:
                        image_url = urls[0]
                    
                    if image_url:
                        return True, image_url
                    
                    if content.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                        return True, content
                    
                    return False, f"API响应中未找到图片数据: {content[:200]}"
                else:
                    return False, f"API请求失败 (状态码 {response.status})"
                    
        except Exception as e:
            logger.error(f"{self.log_prefix} HTTP请求错误: {e!r}", exc_info=True)
            return False, str(e)


# ===== 插件注册 =====

class _RuntimeBridgeMixin:
    """为旧版生图逻辑补齐 rdev 原生运行时上下文。"""

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
            await self.ctx.send.image(image_base64, stream_id)
            return True
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
        custom_model_name = str(self._config_get("llm.model_name", "") or "").strip()
        if not custom_model_name:
            return "planner"

        try:
            from src.services import llm_service

            available_models = llm_service.get_available_models()
            if custom_model_name in available_models:
                return custom_model_name

            # rdev 重构后的 ctx.llm.generate 接受的是任务名（planner/utils 等），
            # 旧版 LLM2PIC 配置里常保存具体模型名。这里把具体模型名映射回包含它的任务，
            # 避免把旧模型名原样传给运行时后触发“未找到模型配置”。
            for task_name, task_config in available_models.items():
                model_list = [str(item).strip() for item in getattr(task_config, "model_list", []) if str(item).strip()]
                if custom_model_name in model_list:
                    logger.info(f"[LLM2picBridge] 将旧版模型名 {custom_model_name} 映射到运行时任务 {task_name}")
                    return str(task_name)
        except Exception as exc:
            logger.warning(f"[LLM2picBridge] 解析 LLM 任务名失败，将回退 planner: {exc}")

        logger.warning(f"[LLM2picBridge] 配置的 LLM 模型/任务 {custom_model_name} 在当前运行时不可用，将回退 planner")
        return "planner"

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
        msg = result.get("message")
        if not isinstance(msg, dict):
            return None

        raw_message = msg.get("raw_message", [])
        if not isinstance(raw_message, list):
            return None

        return self._extract_image_from_segments(raw_message)

    @staticmethod
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
        return None

    async def _ctx_generate_prompt_with_style(
        self,
        *,
        user_request: str,
        chat_messages: str,
        persona: str,
        selfie_mode: bool,
        custom_system_prompt: str = "",
    ) -> Tuple[bool, str, Optional[str]]:
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

请根据以上信息，生成适合的图片提示词和风格判断。必须以 JSON 格式输出。"""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        target_model = await self._ctx_get_llm_model_name()
        try:
            result = await self.ctx.llm.generate(
                prompt=full_prompt,
                model=target_model,
                temperature=0.7,
            )
        except Exception as exc:
            logger.error("[LLM2picBridge] ctx.llm.generate 失败: %s", exc, exc_info=True)
            return False, str(exc), None

        result = _peel_envelope(result)
        if not isinstance(result, dict):
            return False, f"LLM 返回非 dict: {type(result).__name__}", None

        success = bool(result.get("success", False))
        response_text = str(result.get("response") or "").strip()
        if not success:
            return False, str(result.get("error") or "LLM生成失败"), None
        if not response_text:
            return False, "LLM返回空响应", None

        prompt, style = LLMOutputParser.parse(response_text)
        if prompt:
            return True, prompt, style
        return True, response_text, style


class _ActionRuntimeProxy(CustomPicAction):
    """承接旧 Action 逻辑，并把宿主能力映射到 rdev ctx。"""

    def __init__(
        self,
        runtime: _RuntimeBridgeMixin,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        action_data: dict[str, Any],
        session_message: Any = None,
    ) -> None:
        self._runtime = runtime
        self.plugin_config = plugin_config
        self._stream_id = stream_id
        self.action_data = action_data
        self.action_reasoning = ""
        self.cycle_timers = {}
        self.thinking_id = ""
        self.chat_stream = stream_id
        self.log_prefix = "[CustomPicAction]"
        self.message = session_message
        self.action_message = session_message
        self.chat_id = stream_id
        self.user_id = ""
        self.message_id = ""
        self.platform = ""
        self.group_id = ""
        self.group_name = ""
        self.user_nickname = ""
        self.is_group = False
        self.target_id = ""

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

    def _get_llm_model_config(self) -> str:
        return str(self.get_config("llm.model_name", "") or "").strip() or "planner"

    async def _generate_prompt_with_style(
        self,
        *,
        user_request: str,
        chat_messages: str,
        persona: str,
        selfie_mode: bool,
        custom_system_prompt: str,
    ) -> Tuple[bool, str, Optional[str]]:
        return await self._runtime._ctx_generate_prompt_with_style(
            user_request=user_request,
            chat_messages=chat_messages,
            persona=persona,
            selfie_mode=selfie_mode,
            custom_system_prompt=custom_system_prompt,
        )

    async def execute(self) -> Tuple[bool, Optional[str]]:
        logger.info(f"{self.log_prefix} 执行图片生成动作")

        original_description = str(self.action_data.get("description", "") or "").strip()
        selfie_mode = _normalize_bool(self.action_data.get("selfie_mode", False))
        use_input_image = _normalize_bool(self.action_data.get("use_input_image", False))

        input_image_base64 = None
        if use_input_image:
            input_image_base64 = await self._extract_input_image()
            if input_image_base64:
                logger.info(f"{self.log_prefix} 已提取输入图片，将进行图生图")
            else:
                logger.warning(f"{self.log_prefix} 未找到输入图片，将进行文生图")

        chat_messages_str = await self._get_recent_chat_messages()
        persona = await self._get_persona()

        if selfie_mode:
            logger.info(f"{self.log_prefix} 开始生成自拍图片...")
        else:
            logger.info(f"{self.log_prefix} 开始生成图片...")

        custom_system_prompt = str(self.get_config("llm.system_prompt", "") or "")
        success, generated_prompt, llm_style = await self._generate_prompt_with_style(
            user_request=original_description or "根据聊天内容生成一张合适的图片",
            chat_messages=chat_messages_str,
            persona=persona,
            selfie_mode=selfie_mode,
            custom_system_prompt=custom_system_prompt,
        )
        if not success:
            logger.error(f"{self.log_prefix} 生成提示词失败: {generated_prompt}")
            return False, f"提示词生成失败: {generated_prompt}"

        logger.info(f"{self.log_prefix} LLM生成的提示词: {generated_prompt[:200]}..., 风格: {llm_style}")

        style_router = StyleRouter(self.plugin_config)
        selected_style, model_config, route_reason = style_router.route(
            selfie_mode=selfie_mode,
            manual_style=None,
            llm_style=llm_style,
        )
        logger.info(f"{self.log_prefix} 风格路由结果: style={selected_style}, reason={route_reason}")

        final_prompt = self._build_final_prompt(generated_prompt, model_config)
        logger.info(f"{self.log_prefix} 最终提示词: {final_prompt[:200]}...")

        if model_config is None:
            logger.warning(f"{self.log_prefix} 没有配置风格模型，使用旧的 api 配置")
            api_type = str(self.get_config("api.api_type", "openai") or "openai")
            http_base_url = str(self.get_config("api.base_url", "") or "")
            http_api_key = str(self.get_config("api.api_key", "") or "")
            default_model = str(self.get_config("generation.default_model", "gpt-image-1") or "gpt-image-1")
            image_size = str(self.get_config("generation.default_size", "") or "")
            gradio_params = None
            sd_params = None
            novelai_params = None
        else:
            api_type = str(model_config.get("api_type", "openai") or "openai")
            http_base_url = str(model_config.get("base_url", "") or "")
            http_api_key = str(model_config.get("api_key", "") or "")
            default_model = str(model_config.get("model_name", "") or "")
            image_size = str(self.get_config("generation.default_size", "") or "")
            gradio_params = {
                "resolution": model_config.get("gradio_resolution", "1024x1024 ( 1:1 )"),
                "steps": model_config.get("gradio_steps", 8),
                "shift": model_config.get("gradio_shift", 3),
                "timeout": model_config.get("gradio_timeout", 120),
            }
            sd_params = {
                "negative_prompt": model_config.get("sd_negative_prompt", ""),
                "width": model_config.get("sd_width", 512),
                "height": model_config.get("sd_height", 512),
                "steps": model_config.get("sd_steps", 20),
                "cfg": model_config.get("sd_cfg", 7.0),
                "model_index": model_config.get("sd_model_index", 0),
                "seed": model_config.get("sd_seed", -1),
            }
            novelai_params = {
                "model": model_config.get("novelai_model", "nai-diffusion-4-5-full"),
                "width": model_config.get("novelai_width", 832),
                "height": model_config.get("novelai_height", 1216),
                "steps": model_config.get("novelai_steps", 28),
                "scale": model_config.get("novelai_scale", 5.0),
                "sampler": model_config.get("novelai_sampler", "k_euler"),
                "negative_prompt": model_config.get("novelai_negative_prompt", ""),
                "seed": model_config.get("novelai_seed", -1),
                "timeout": model_config.get("novelai_timeout", 120),
            }

        if api_type.lower() != "novelai" and not http_base_url:
            return False, f"{selected_style} 模型的 base_url 未配置"
        if api_type.lower() not in ("gradio", "regex_url") and not http_api_key.strip():
            return False, f"{selected_style} 模型的 API密钥未配置"

        try:
            if api_type.lower() == "gradio":
                request_success, result = await asyncio.to_thread(
                    self._make_gradio_image_request,
                    prompt=final_prompt,
                    base_url=http_base_url,
                    gradio_params=gradio_params,
                )
            elif api_type.lower() == "sd_api":
                request_success, result = await asyncio.to_thread(
                    self._make_sd_api_request,
                    prompt=final_prompt,
                    base_url=http_base_url,
                    api_key=http_api_key,
                    sd_params=sd_params,
                )
            elif api_type.lower() == "novelai":
                request_success, result = await asyncio.to_thread(
                    self._make_novelai_request,
                    prompt=final_prompt,
                    api_key=http_api_key,
                    novelai_params=novelai_params,
                )
            elif api_type.lower() == "regex_url":
                request_success, result = await asyncio.to_thread(
                    self._make_regex_url_request,
                    prompt=final_prompt,
                    url_template=http_base_url,
                )
            else:
                request_success, result = await asyncio.to_thread(
                    self._make_http_image_request,
                    prompt=final_prompt,
                    model=default_model,
                    size=image_size if image_size else None,
                    base_url=http_base_url,
                    api_key=http_api_key,
                    input_image_base64=input_image_base64,
                )
        except Exception as exc:
            logger.error(f"{self.log_prefix} 图片生成请求失败: {exc!r}", exc_info=True)
            return False, f"图片生成服务遇到问题: {str(exc)[:100]}"

        if request_success:
            return await self._handle_image_result(result)
        logger.error(f"{self.log_prefix} 图片生成失败: {result}")
        return False, f"图片生成失败: {result}"


class _CommandRuntimeProxy(DirectPicCommand):
    """承接旧命令逻辑，并把宿主能力映射到 rdev ctx。"""

    def __init__(
        self,
        runtime: _RuntimeBridgeMixin,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        matched_groups: dict[str, Any],
        session_message: Any = None,
    ) -> None:
        self._runtime = runtime
        self.plugin_config = plugin_config
        self._stream_id = stream_id
        self.matched_groups = matched_groups
        self.log_prefix = "[DirectPic]"
        self.message = session_message
        self.chat_id = stream_id

    def get_config(self, path: str, default: Any = None) -> Any:
        return self._runtime._config_get(path, default)

    async def send_text(self, text: str) -> bool:
        return await self._runtime._ctx_send_text(text, self._stream_id)

    async def send_image(self, image_base64: str) -> bool:
        return await self._runtime._ctx_send_image(image_base64, self._stream_id)


class LLM2PicPlugin(MaiBotPlugin, _RuntimeBridgeMixin):
    """LLM2pic 的 rdev 原生插件入口。"""

    async def on_load(self) -> None:
        self.ctx.logger.info("MaiBot_LLM2pic 原生适配插件已加载")

    async def on_unload(self) -> None:
        self.ctx.logger.info("MaiBot_LLM2pic 原生适配插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        del config_data
        self.ctx.logger.info("MaiBot_LLM2pic 配置更新: scope=%s version=%s", scope, version)

    @Action(
        "draw_picture",
        description=CustomPicAction.action_description,
        activation_type=ActivationType.ALWAYS,
        action_parameters=dict(CustomPicAction.action_parameters),
        action_require=list(CustomPicAction.action_require),
        associated_types=list(CustomPicAction.associated_types),
        parallel_action=bool(CustomPicAction.parallel_action),
        action_prompt=str(CustomPicAction.llm_judge_prompt),
    )
    async def handle_draw_picture(
        self,
        description: str = "",
        selfie_mode: bool = False,
        use_input_image: bool = False,
        stream_id: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str]:
        plugin_config = self.get_plugin_config_data()
        if not _normalize_bool(self._config_get("components.enable_image_generation", True)):
            return False, "图片生成功能未启用"

        # 先生成 prompt（快速，在 RPC 超时内完成）
        proxy = _ActionRuntimeProxy(
            self,
            plugin_config=plugin_config,
            stream_id=stream_id,
            action_data={
                "description": description,
                "selfie_mode": selfie_mode,
            },
            session_message=kwargs.get("message"),
        )

        # 获取聊天记录、生成提示词（都在30秒内）
        original_description = str(proxy.action_data.get("description", "") or "").strip()
        selfie_mode_bool = _normalize_bool(proxy.action_data.get("selfie_mode", False))

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
            return False, f"提示词生成失败: {generated_prompt}"

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
        return True, ""

    async def _background_generate_and_send(
        self,
        *,
        plugin_config: dict[str, Any],
        stream_id: str,
        generated_prompt: str,
        llm_style: Optional[str],
        selfie_mode_bool: bool,
        input_image_base64: Optional[str],
        proxy: _ActionRuntimeProxy,
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
        proxy: _ActionRuntimeProxy,
    ) -> None:
        """后台异步完成图片生成和发送（内部实现）。"""
        style_router = StyleRouter(plugin_config)
        selected_style, model_config, route_reason = style_router.route(
            selfie_mode=selfie_mode_bool,
            manual_style=None,
            llm_style=llm_style,
        )
        final_prompt = proxy._build_final_prompt(generated_prompt, model_config)

        if model_config is None:
            api_type = str(proxy.get_config("api.api_type", "openai") or "openai")
            http_base_url = str(proxy.get_config("api.base_url", "") or "")
            http_api_key = str(proxy.get_config("api.api_key", "") or "")
            default_model = str(proxy.get_config("generation.default_model", "gpt-image-1") or "gpt-image-1")
            image_size = str(proxy.get_config("generation.default_size", "") or "")
            gradio_params = None
            sd_params = None
            novelai_params = None
        else:
            api_type = str(model_config.get("api_type", "openai") or "openai")
            http_base_url = str(model_config.get("base_url", "") or "")
            http_api_key = str(model_config.get("api_key", "") or "")
            default_model = str(model_config.get("model_name", "") or "")
            image_size = str(proxy.get_config("generation.default_size", "") or "")
            gradio_params = {
                "resolution": model_config.get("gradio_resolution", "1024x1024 ( 1:1 )"),
                "steps": model_config.get("gradio_steps", 8),
                "shift": model_config.get("gradio_shift", 3),
                "timeout": model_config.get("gradio_timeout", 120),
            }
            sd_params = {
                "negative_prompt": model_config.get("sd_negative_prompt", ""),
                "width": model_config.get("sd_width", 512),
                "height": model_config.get("sd_height", 512),
                "steps": model_config.get("sd_steps", 20),
                "cfg": model_config.get("sd_cfg", 7.0),
                "model_index": model_config.get("sd_model_index", 0),
                "seed": model_config.get("sd_seed", -1),
            }
            novelai_params = {
                "model": model_config.get("novelai_model", "nai-diffusion-4-5-full"),
                "width": model_config.get("novelai_width", 832),
                "height": model_config.get("novelai_height", 1216),
                "steps": model_config.get("novelai_steps", 28),
                "scale": model_config.get("novelai_scale", 5.0),
                "sampler": model_config.get("novelai_sampler", "k_euler"),
                "negative_prompt": model_config.get("novelai_negative_prompt", ""),
                "seed": model_config.get("novelai_seed", -1),
                "timeout": model_config.get("novelai_timeout", 120),
            }

        if api_type.lower() != "novelai" and not http_base_url:
            await self._ctx_send_text("画图的 base_url 没配置，画不了", stream_id)
            return
        if api_type.lower() not in ("gradio", "regex_url") and not http_api_key.strip():
            await self._ctx_send_text("画图的 API 密钥没配，画不了", stream_id)
            return

        try:
            if api_type.lower() == "gradio":
                request_success, result = await asyncio.to_thread(
                    proxy._make_gradio_image_request,
                    prompt=final_prompt,
                    base_url=http_base_url,
                    gradio_params=gradio_params,
                )
            elif api_type.lower() == "sd_api":
                request_success, result = await asyncio.to_thread(
                    proxy._make_sd_api_request,
                    prompt=final_prompt,
                    base_url=http_base_url,
                    api_key=http_api_key,
                    sd_params=sd_params,
                )
            elif api_type.lower() == "novelai":
                request_success, result = await asyncio.to_thread(
                    proxy._make_novelai_request,
                    prompt=final_prompt,
                    api_key=http_api_key,
                    novelai_params=novelai_params,
                )
            elif api_type.lower() == "regex_url":
                request_success, result = await asyncio.to_thread(
                    proxy._make_regex_url_request,
                    prompt=final_prompt,
                    url_template=http_base_url,
                )
            else:
                request_success, result = await asyncio.to_thread(
                    proxy._make_http_image_request,
                    prompt=final_prompt,
                    model=default_model,
                    size=image_size if image_size else None,
                    base_url=http_base_url,
                    api_key=http_api_key,
                    input_image_base64=input_image_base64,
                )
        except Exception as exc:
            logger.error("[LLM2PicPlugin] 后台图片生成请求失败: %s", exc, exc_info=True)
            await self._ctx_send_text(f"画图出错了: {str(exc)[:80]}", stream_id)
            return

        if request_success:
            await self._background_handle_result(result, proxy, stream_id)
        else:
            await self._ctx_send_text(f"画图失败了: {str(result)[:80]}", stream_id)

    async def _background_handle_result(
        self,
        result: str,
        proxy: _ActionRuntimeProxy,
        stream_id: str,
    ) -> None:
        """后台处理生图结果：base64 直接发送，URL 下载后发送。"""
        if not stream_id:
            logger.error("[LLM2PicPlugin] stream_id 为空，无法发送图片")
            return

        if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
            crop_enabled = _normalize_bool(proxy.get_config("generation.crop_enabled", False))
            if crop_enabled:
                try:
                    import base64 as b64
                    image_bytes = b64.b64decode(result)
                    image_bytes = proxy._crop_image(image_bytes)
                    result = b64.b64encode(image_bytes).decode("utf-8")
                except Exception:
                    pass
            # 压缩大图片为 JPEG 以避免 QQ 发送失败
            result = _compress_image_if_needed(result)
            logger.info(f"[LLM2PicPlugin] 发送图片 base64 ({len(result)} chars) 到 stream_id={stream_id}")
            success = await self._ctx_send_image(result, stream_id)
            if not success:
                logger.error("[LLM2PicPlugin] 图片发送失败")
                await self._ctx_send_text("图片生成成功但发送失败", stream_id)
        else:
            try:
                encode_success, encode_result = await asyncio.to_thread(
                    proxy._download_and_encode_base64, result
                )
            except Exception:
                encode_success = False
                encode_result = "下载图片异常"
            if encode_success:
                encode_result = _compress_image_if_needed(encode_result)
                success = await self._ctx_send_image(encode_result, stream_id)
                if not success:
                    await self._ctx_send_text("图片生成成功但发送失败", stream_id)
            else:
                await self._ctx_send_text(f"图片下载失败: {encode_result}", stream_id)

    # ===== edit_picture Action =====

    @Action(
        "edit_picture",
        description="使用 GPT 图像模型生成或编辑图片。当用户发送图片并要求修改时进行图生图；当用户纯文字描述想要的图片时也可直接生成。",
        activation_type=ActivationType.ALWAYS,
        action_parameters={
            "description": "用户的图片需求描述，例如'变成动漫风格'、'画一只猫在月球上'、'把背景换成海边'",
        },
        action_require=[
            "当用户发送图片并要求修改/编辑/重绘/变换风格时使用",
            "当用户引用一张图片并描述修改需求时使用",
            "当用户要求使用 GPT 模型生成高质量图片时使用",
        ],
        associated_types=["text", "image"],
        parallel_action=False,
        action_prompt="""此动作让你能够使用 GPT 图像模型生成图片或编辑用户提供的图片。

【触发条件 - 图片编辑】
1. 用户发送了一张图片，同时附带文字描述修改需求
2. 用户引用/回复了一条包含图片的消息，并描述了修改需求
3. 用户要求对图片进行风格转换、元素添加/删除、重绘等操作

【触发条件 - 文生图】
1. 用户明确要求使用高质量/写实风格生成图片
2. 用户的需求更适合 GPT 图像模型而非二次元风格

【典型触发语句示例】
图片编辑：
- [图片] "把这张图变成动漫风格"
- [图片] "帮我把背景换成星空"
- [引用图片消息] "给这张图加上圣诞帽"
- [图片] "重绘这张图，画面更明亮一些"

文生图：
- "用GPT画一张写实的猫咪"
- "生成一张赛博朋克城市的图"

【禁止触发】
- 用户只是讨论或评价图片，没有修改/生成需求
- 用户要求自拍或二次元角色图（应使用 draw_picture）
- 前面聊天记录中你已经发过图片时，禁止再次生成
""",
    )
    async def handle_edit_picture(
        self,
        description: str = "",
        stream_id: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str]:
        plugin_config = self.get_plugin_config_data()

        # 检查 edit 模型是否配置
        style_router = StyleRouter(plugin_config)
        if not style_router.is_style_available("edit"):
            return False, "图片编辑功能未配置 edit 模型"

        # 后台异步执行，快速返回避免 RPC 超时
        asyncio.create_task(
            self._background_edit_picture(
                plugin_config=plugin_config,
                stream_id=stream_id,
                description=description,
            )
        )
        await self._ctx_send_text("正在编辑图片，请稍等...", stream_id)
        return True, ""

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
        custom_prompt_add = str(model_config.get("custom_prompt_add", "") or "").strip()
        if custom_prompt_add:
            final_prompt = f"{custom_prompt_add} {final_prompt}"

        mode_label = "图生图" if input_image_base64 else "文生图"
        logger.info(f"[EditPicture] {mode_label}: prompt={final_prompt[:100]}...")

        # 4. 调用 OpenAI chat/completions 端点
        api_type = str(model_config.get("api_type", "openai") or "openai")
        http_base_url = str(model_config.get("base_url", "") or "")
        http_api_key = str(model_config.get("api_key", "") or "")
        default_model = str(model_config.get("model_name", "") or "")

        if not http_base_url:
            await self._ctx_send_text("edit 模型的 base_url 未配置", stream_id)
            return
        if not http_api_key:
            await self._ctx_send_text("edit 模型的 API 密钥未配置", stream_id)
            return

        try:
            request_success, result = await asyncio.to_thread(
                self._edit_picture_request,
                prompt=final_prompt,
                model=default_model,
                base_url=http_base_url,
                api_key=http_api_key,
                input_image_base64=input_image_base64,
            )
        except Exception as exc:
            logger.error("[EditPicture] 请求失败: %s", exc, exc_info=True)
            await self._ctx_send_text(f"图片生成请求失败: {str(exc)[:80]}", stream_id)
            return

        if not request_success:
            await self._ctx_send_text(f"图片生成失败: {str(result)[:80]}", stream_id)
            return

        # 5. 处理结果并发送
        if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
            result = _compress_image_if_needed(result)
            success = await self._ctx_send_image(result, stream_id)
            if not success:
                await self._ctx_send_text("图片编辑成功但发送失败", stream_id)
        else:
            # result 是 URL，下载后发送
            try:
                dl_success, dl_result = await asyncio.to_thread(
                    download_image_to_base64, result
                )
            except Exception:
                dl_success = False
                dl_result = "下载图片异常"
            if dl_success:
                dl_result = _compress_image_if_needed(dl_result)
                success = await self._ctx_send_image(dl_result, stream_id)
                if not success:
                    await self._ctx_send_text("图片编辑成功但发送失败", stream_id)
            else:
                await self._ctx_send_text(f"编辑后图片下载失败: {dl_result}", stream_id)

    def _edit_picture_request(
        self,
        prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        input_image_base64: Optional[str],
    ) -> Tuple[bool, str]:
        """发送图片生成/编辑请求到 OpenAI chat/completions 端点。

        当 input_image_base64 不为空时为图生图，否则为纯文生图。
        """
        endpoint = f"{base_url.rstrip('/')}/chat/completions"

        if input_image_base64:
            # 图生图：多模态消息
            mime_type = get_image_mime_type(input_image_base64)
            message_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{input_image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        else:
            # 纯文生图：纯文本消息
            message_content = prompt

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message_content}],
        }

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

                if 200 <= response.status < 300:
                    response_data = json.loads(response_body)

                    # 从 chat completions 响应中提取内容
                    content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    # 尝试从 markdown 图片标签中提取 data URI 的 base64
                    data_uri_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                    data_uri_match = re.search(data_uri_pattern, content)
                    if data_uri_match:
                        return True, data_uri_match.group(1)

                    # 尝试从 content 中提取图片 URL
                    url_pattern = r'https?://[^\s\)\]\"\'<>]+'
                    urls = re.findall(url_pattern, content)

                    image_url = None
                    for url in urls:
                        if any(ext in url.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']) or 'image' in url.lower():
                            image_url = url
                            break

                    if not image_url and urls:
                        image_url = urls[0]

                    if image_url:
                        return True, image_url

                    # 如果 content 本身是 base64
                    if content.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                        return True, content

                    return False, f"API 响应中未找到图片数据: {content[:200]}"
                else:
                    return False, f"API 请求失败 (状态码 {response.status})"

        except Exception as e:
            logger.error(f"[EditPicture] HTTP 请求错误: {e!r}", exc_info=True)
            return False, str(e)

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
            # 空 prompt 快速返回
            proxy = _CommandRuntimeProxy(
                self,
                plugin_config=plugin_config,
                stream_id=stream_id,
                matched_groups=dict(matched_groups or {}),
                session_message=kwargs.get("message"),
            )
            return await proxy.execute()

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
            matched_groups={"prompt": raw_prompt, "style": manual_style or ""},
            session_message=session_message,
        )

        # 尝试提取输入图片
        input_image_base64 = await proxy._extract_input_image()

        style_router = StyleRouter(plugin_config)
        selected_style, model_config, route_reason = style_router.route(
            selfie_mode=False,
            manual_style=manual_style,
            llm_style=None,
        )

        if model_config is None:
            api_type = str(proxy.get_config("api.api_type", "openai") or "openai")
            http_base_url = str(proxy.get_config("api.base_url", "") or "")
            http_api_key = str(proxy.get_config("api.api_key", "") or "")
            default_model = str(proxy.get_config("generation.default_model", "gpt-image-1") or "gpt-image-1")
            image_size = str(proxy.get_config("generation.default_size", "") or "")
            gradio_params = None
            sd_params = None
            novelai_params = None
        else:
            api_type = str(model_config.get("api_type", "openai") or "openai")
            http_base_url = str(model_config.get("base_url", "") or "")
            http_api_key = str(model_config.get("api_key", "") or "")
            default_model = str(model_config.get("model_name", "") or "")
            image_size = str(proxy.get_config("generation.default_size", "") or "")
            gradio_params = {
                "resolution": model_config.get("gradio_resolution", "1024x1024 ( 1:1 )"),
                "steps": model_config.get("gradio_steps", 8),
                "shift": model_config.get("gradio_shift", 3),
                "timeout": model_config.get("gradio_timeout", 120),
            }
            sd_params = {
                "negative_prompt": model_config.get("sd_negative_prompt", ""),
                "width": model_config.get("sd_width", 512),
                "height": model_config.get("sd_height", 512),
                "steps": model_config.get("sd_steps", 20),
                "cfg": model_config.get("sd_cfg", 7.0),
                "model_index": model_config.get("sd_model_index", 0),
                "seed": model_config.get("sd_seed", -1),
            }
            novelai_params = {
                "model": model_config.get("novelai_model", "nai-diffusion-4-5-full"),
                "width": model_config.get("novelai_width", 832),
                "height": model_config.get("novelai_height", 1216),
                "steps": model_config.get("novelai_steps", 28),
                "scale": model_config.get("novelai_scale", 5.0),
                "sampler": model_config.get("novelai_sampler", "k_euler"),
                "negative_prompt": model_config.get("novelai_negative_prompt", ""),
                "seed": model_config.get("novelai_seed", -1),
                "timeout": model_config.get("novelai_timeout", 120),
            }

        if api_type.lower() != "novelai" and not http_base_url:
            await self._ctx_send_text("画图的 base_url 没配置，画不了", stream_id)
            return
        if api_type.lower() not in ("gradio", "regex_url") and not http_api_key.strip():
            await self._ctx_send_text("画图的 API 密钥没配，画不了", stream_id)
            return

        custom_prompt_add = ""
        if model_config and model_config.get("custom_prompt_add"):
            custom_prompt_add = model_config.get("custom_prompt_add", "")
        else:
            custom_prompt_add = str(proxy.get_config("generation.custom_prompt_add", "") or "")
        if custom_prompt_add and custom_prompt_add.strip():
            final_prompt = f"{custom_prompt_add.strip()}, {raw_prompt}"
        else:
            final_prompt = raw_prompt

        try:
            if api_type.lower() == "gradio":
                request_success, result = await asyncio.to_thread(
                    proxy._make_gradio_image_request,
                    prompt=final_prompt,
                    base_url=http_base_url,
                    gradio_params=gradio_params,
                )
            elif api_type.lower() == "sd_api":
                request_success, result = await asyncio.to_thread(
                    proxy._make_sd_api_request,
                    prompt=final_prompt,
                    base_url=http_base_url,
                    api_key=http_api_key,
                    sd_params=(sd_params if model_config else None),
                )
            elif api_type.lower() == "novelai":
                request_success, result = await asyncio.to_thread(
                    proxy._make_novelai_request,
                    prompt=final_prompt,
                    api_key=http_api_key,
                    novelai_params=(novelai_params if model_config else None),
                )
            elif api_type.lower() == "regex_url":
                request_success, result = await asyncio.to_thread(
                    proxy._make_regex_url_request,
                    prompt=final_prompt,
                    url_template=http_base_url,
                )
            else:
                request_success, result = await asyncio.to_thread(
                    proxy._make_http_image_request,
                    prompt=final_prompt,
                    model=default_model,
                    size=image_size if image_size else None,
                    base_url=http_base_url,
                    api_key=http_api_key,
                    input_image_base64=input_image_base64,
                )
        except Exception as exc:
            logger.error("[LLM2PicPlugin] 后台 /pic 请求失败: %s", exc, exc_info=True)
            await self._ctx_send_text(f"/pic 出错了: {str(exc)[:80]}", stream_id)
            return

        if request_success:
            # 复用 _background_handle_result 逻辑
            if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):
                crop_enabled = _normalize_bool(proxy.get_config("generation.crop_enabled", False))
                if crop_enabled:
                    try:
                        import base64 as b64_img
                        img_bytes = b64_img.b64decode(result)
                        img_bytes = proxy._crop_image(img_bytes)
                        result = b64_img.b64encode(img_bytes).decode("utf-8")
                    except Exception:
                        pass
                await self._ctx_send_image(result, stream_id)
            else:
                try:
                    encode_success, encode_result = await asyncio.to_thread(
                        proxy._download_and_encode_base64, result
                    )
                except Exception:
                    encode_success = False
                    encode_result = "下载图片异常"
                if encode_success:
                    await self._ctx_send_image(encode_result, stream_id)
                else:
                    await self._ctx_send_text(f"图片下载失败: {encode_result}", stream_id)
        else:
            await self._ctx_send_text(f"/pic 失败了: {str(result)[:80]}", stream_id)


def create_plugin() -> LLM2PicPlugin:
    """rdev Runner 原生插件工厂。"""
    return LLM2PicPlugin()
