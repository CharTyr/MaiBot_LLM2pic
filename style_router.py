"""
风格路由器和 LLM 输出解析器
"""

import json
from typing import List, Optional, Tuple

from src.common.logger import get_logger

logger = get_logger("MaiBot_LLM2pic")


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

## 场景补全规则
当用户没有明确指定地点、背景、时间、天气或光线时，你必须根据主体、动作、情绪和聊天上下文补充一个具体且自然的场景。
- 场景应包含背景/地点 + 氛围/时间/光线，例如 bedroom, cafe, city street, classroom, sunset, soft lighting
- 不要输出空泛的 background、beautiful scene、nice place
- 用户已经指定场景时，严格保留用户场景，不要替换
- 自拍/肖像请求没有指定场景时，补充适合角色当前状态的自然环境

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


def _get_endpoint_config(style_config: dict, api_type: str) -> dict:
    endpoint_config = style_config.get(api_type)
    return endpoint_config if isinstance(endpoint_config, dict) else {}


def _endpoint_value(style_config: dict, endpoint_config: dict, key: str, default: object = "") -> object:
    if key in endpoint_config:
        return endpoint_config[key]
    return style_config.get(key, default)


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

        api_type = str(style_config.get("api_type", "openai") or "openai").lower().replace("-", "_")
        endpoint_config = _get_endpoint_config(style_config, api_type)
        return {
            "api_type": style_config.get("api_type", "openai"),
            "base_url": _endpoint_value(style_config, endpoint_config, "base_url", ""),
            "api_key": _endpoint_value(style_config, endpoint_config, "api_key", ""),
            "model_name": _endpoint_value(style_config, endpoint_config, "model_name", ""),
            "size": _endpoint_value(style_config, endpoint_config, "size", style_config.get("size", "")),
            "custom_prompt_add": _endpoint_value(style_config, endpoint_config, "custom_prompt_add", ""),
            # Gradio 参数
            "gradio_resolution": _endpoint_value(style_config, endpoint_config, "resolution", style_config.get("gradio_resolution", "1024x1024 ( 1:1 )")),
            "gradio_steps": _endpoint_value(style_config, endpoint_config, "steps", style_config.get("gradio_steps", 8)),
            "gradio_shift": _endpoint_value(style_config, endpoint_config, "shift", style_config.get("gradio_shift", 3)),
            "gradio_timeout": _endpoint_value(style_config, endpoint_config, "timeout", style_config.get("gradio_timeout", 120)),
            # SD API 参数
            "sd_negative_prompt": _endpoint_value(style_config, endpoint_config, "negative_prompt", style_config.get("sd_negative_prompt", "")),
            "sd_width": _endpoint_value(style_config, endpoint_config, "width", style_config.get("sd_width", 512)),
            "sd_height": _endpoint_value(style_config, endpoint_config, "height", style_config.get("sd_height", 512)),
            "sd_steps": _endpoint_value(style_config, endpoint_config, "steps", style_config.get("sd_steps", 20)),
            "sd_cfg": _endpoint_value(style_config, endpoint_config, "cfg", style_config.get("sd_cfg", 7.0)),
            "sd_model_index": _endpoint_value(style_config, endpoint_config, "model_index", style_config.get("sd_model_index", 0)),
            "sd_seed": _endpoint_value(style_config, endpoint_config, "seed", style_config.get("sd_seed", -1)),
            # NovelAI API 参数
            "novelai_model": _endpoint_value(style_config, endpoint_config, "model", style_config.get("novelai_model", "nai-diffusion-4-5-full")),
            "novelai_width": _endpoint_value(style_config, endpoint_config, "width", style_config.get("novelai_width", 832)),
            "novelai_height": _endpoint_value(style_config, endpoint_config, "height", style_config.get("novelai_height", 1216)),
            "novelai_steps": _endpoint_value(style_config, endpoint_config, "steps", style_config.get("novelai_steps", 28)),
            "novelai_scale": _endpoint_value(style_config, endpoint_config, "scale", style_config.get("novelai_scale", 5.0)),
            "novelai_sampler": _endpoint_value(style_config, endpoint_config, "sampler", style_config.get("novelai_sampler", "k_euler")),
            "novelai_negative_prompt": _endpoint_value(style_config, endpoint_config, "negative_prompt", style_config.get("novelai_negative_prompt", "")),
            "novelai_seed": _endpoint_value(style_config, endpoint_config, "seed", style_config.get("novelai_seed", -1)),
            "novelai_timeout": _endpoint_value(style_config, endpoint_config, "timeout", style_config.get("novelai_timeout", 120)),
            # NewAPI NAI 参数
            "newapi_nai_negative_prompt": _endpoint_value(style_config, endpoint_config, "negative_prompt", style_config.get("newapi_nai_negative_prompt", "")),
            "newapi_nai_size": _endpoint_value(style_config, endpoint_config, "size", style_config.get("newapi_nai_size", "portrait")),
            "newapi_nai_steps": _endpoint_value(style_config, endpoint_config, "steps", style_config.get("newapi_nai_steps", 23)),
            "newapi_nai_scale": _endpoint_value(style_config, endpoint_config, "scale", style_config.get("newapi_nai_scale", 5)),
            "newapi_nai_sampler": _endpoint_value(style_config, endpoint_config, "sampler", style_config.get("newapi_nai_sampler", "k_euler_ancestral")),
            "newapi_nai_seed": _endpoint_value(style_config, endpoint_config, "seed", style_config.get("newapi_nai_seed", -1)),
            "newapi_nai_image_format": _endpoint_value(style_config, endpoint_config, "image_format", style_config.get("newapi_nai_image_format", "png")),
            "newapi_nai_max_tokens": _endpoint_value(style_config, endpoint_config, "max_tokens", style_config.get("newapi_nai_max_tokens", 100000)),
            "newapi_nai_timeout": _endpoint_value(style_config, endpoint_config, "timeout", style_config.get("newapi_nai_timeout", 180)),
            "newapi_nai_retry_attempts": _endpoint_value(style_config, endpoint_config, "retry_attempts", style_config.get("newapi_nai_retry_attempts", 3)),
            "newapi_nai_proxy_mode": _endpoint_value(style_config, endpoint_config, "proxy_mode", style_config.get("newapi_nai_proxy_mode", "auto")),
            "newapi_nai_quality_toggle": _endpoint_value(style_config, endpoint_config, "quality_toggle", style_config.get("newapi_nai_quality_toggle", True)),
            "newapi_nai_auto_smea": _endpoint_value(style_config, endpoint_config, "auto_smea", style_config.get("newapi_nai_auto_smea", False)),
            "newapi_nai_variety_boost": _endpoint_value(style_config, endpoint_config, "variety_boost", style_config.get("newapi_nai_variety_boost", False)),
            "newapi_nai_extra_params": _endpoint_value(style_config, endpoint_config, "extra_params", style_config.get("newapi_nai_extra_params", {})),
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
        """
        if not llm_output:
            return "", None

        llm_output = llm_output.strip()

        # 尝试解析 JSON
        try:
            data = json.loads(llm_output)

            if not isinstance(data, dict):
                logger.warning("[LLMOutputParser] JSON 解析结果不是字典，使用原始输出")
                return llm_output, None

            prompt = data.get("prompt", "")
            style = data.get("style", None)
            validated_style = LLMOutputParser.validate_style(style)

            logger.debug(f"[LLMOutputParser] JSON 解析成功: prompt={prompt[:50]}..., style={validated_style}")
            return prompt, validated_style

        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON
            json_match = LLMOutputParser._extract_json_from_text(llm_output)
            if json_match:
                try:
                    data = json.loads(json_match)

                    if not isinstance(data, dict):
                        logger.warning("[LLMOutputParser] 提取的 JSON 不是字典，使用原始输出")
                        return llm_output, None

                    prompt = data.get("prompt", "")
                    style = data.get("style", None)
                    validated_style = LLMOutputParser.validate_style(style)
                    logger.debug(f"[LLMOutputParser] 从文本中提取 JSON 成功: prompt={prompt[:50]}..., style={validated_style}")
                    return prompt, validated_style
                except json.JSONDecodeError:
                    pass

            # 回退：使用整个输出作为 prompt
            logger.warning("[LLMOutputParser] LLM 输出不是有效 JSON，使用原始输出作为 prompt")
            return llm_output, None

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[str]:
        """从文本中提取 JSON 字符串"""
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
