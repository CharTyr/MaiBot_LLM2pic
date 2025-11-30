"""
MaiBot_LLM2pic - MaiBot图片生成插件

使用LLM根据聊天记录和人设生成符合需求的prompt，然后调用图片生成API
"""

import asyncio
import json
import urllib.request
import base64
import traceback
import time
from typing import List, Tuple, Type, Optional

from src.plugin_system import (
    BasePlugin,
    BaseAction,
    register_plugin,
    ComponentInfo,
    ActionActivationType,
    ConfigField,
    llm_api,
    message_api,
)
from src.plugin_system.base.component_types import ChatMode, CommandInfo
from src.plugin_system.base.base_command import BaseCommand
from src.chat.message_receive.message import MessageRecv
from src.config.config import global_config, model_config
from src.common.logger import get_logger

logger = get_logger("MaiBot_LLM2pic")


# ===== 默认提示词模板 =====

DEFAULT_SYSTEM_PROMPT = """你是一位专业的AI绘画提示词生成专家。你的任务是根据用户的请求和聊天上下文，生成高质量的英文图片生成提示词，并判断适合的绘画风格。

## 你的角色设定
{persona}

## 输出格式
你必须以 JSON 格式输出，包含两个字段：
- prompt: 英文图片生成提示词
- style: 绘画风格，只能是 "anime"（二次元/动漫风格）或 "real"（写实/真实风格）

## 风格判断规则
- anime（二次元）：动漫角色、游戏角色、虚拟人物、可爱风格、日系风格、卡通风格
- real（写实）：真实人物、照片风格、写实场景、自然风景、真实物品

## 提示词规则
1. 使用逗号分隔的英文关键词格式
2. 关键词顺序：人物/主体 -> 外貌特征 -> 服装 -> 动作/姿势 -> 表情 -> 背景/场景
3. 对于角色请求，使用角色的罗马音名称并补充作品名称，如 rem (re zero)
4. 单人构图时添加 solo 标签
5. 不要添加质量词如 masterpiece, best quality 等（系统会自动添加）
6. 不要添加任何NSFW内容

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

用户请求: "画一张真实的风景照"
输出: {{"prompt": "beautiful landscape, mountain, lake, sunset, golden hour, realistic photography", "style": "real"}}

用户请求: "自拍"
输出: {{"prompt": "1girl, solo, selfie, front-facing camera, close-up shot, smile, casual clothes, indoor background", "style": "anime"}}"""


# ===== 风格路由器 =====

class StyleRouter:
    """风格路由器，根据各种条件决定使用哪个模型"""
    
    VALID_STYLES = ["anime", "real"]
    
    def __init__(self, config: dict):
        """
        初始化风格路由器
        
        Args:
            config: 插件配置字典
        """
        self.config = config
        self.default_style = config.get("generation", {}).get("default_style", "anime")
        self.anime_config = self._extract_model_config("anime")
        self.real_config = self._extract_model_config("real")
        
        logger.debug(f"[StyleRouter] 初始化完成: default_style={self.default_style}, "
                    f"anime_enabled={self.anime_config is not None}, "
                    f"real_enabled={self.real_config is not None}")
    
    def _extract_model_config(self, style: str) -> Optional[dict]:
        """
        提取指定风格的模型配置
        
        Args:
            style: 风格名称 ("anime" 或 "real")
            
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
        if self.real_config:
            available_configs.append(("real", self.real_config))
        
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
                # anime 未配置，回退到 real
                logger.warning("[StyleRouter] selfie_mode=True 但 anime 未配置，回退到 real")
                return "real", self.real_config, "selfie_mode_fallback_to_real"
        
        # 2. 手动指定风格
        if manual_style:
            manual_style_lower = manual_style.lower().strip()
            if manual_style_lower in self.VALID_STYLES:
                config = self.anime_config if manual_style_lower == "anime" else self.real_config
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
                config = self.anime_config if llm_style_lower == "anime" else self.real_config
                if config:
                    logger.debug(f"[StyleRouter] LLM 判断风格: {llm_style_lower}")
                    return llm_style_lower, config, "llm_style"
                else:
                    logger.warning(f"[StyleRouter] LLM 判断的风格 {llm_style_lower} 未配置，使用默认风格")
        
        # 4. 使用默认风格
        default_config = self.anime_config if self.default_style == "anime" else self.real_config
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
        elif style == "real":
            return self.real_config is not None
        return False
    
    def get_available_styles(self) -> List[str]:
        """获取所有可用的风格列表"""
        styles = []
        if self.anime_config:
            styles.append("anime")
        if self.real_config:
            styles.append("real")
        return styles


# ===== LLM 输出解析器 =====

class LLMOutputParser:
    """LLM 输出解析器，用于解析 JSON 格式的 LLM 输出"""
    
    VALID_STYLES = ["anime", "real"]
    
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


# ===== Prompt生成器 =====

class PromptGenerator:
    """使用LLM生成图片提示词"""

    @staticmethod
    async def generate_prompt(
        user_request: str,
        chat_messages: str,
        persona: str,
        selfie_mode: bool,
        model_config_to_use,
        custom_system_prompt: str = "",
    ) -> Tuple[bool, str]:
        """
        使用LLM生成图片提示词
        
        Args:
            user_request: 用户的绘图请求
            chat_messages: 最近的聊天记录
            persona: 人设信息
            selfie_mode: 是否为自拍模式
            model_config_to_use: 使用的模型配置
            custom_system_prompt: 自定义系统提示词（留空则使用默认）
            
        Returns:
            Tuple[bool, str]: (是否成功, 生成的提示词或错误信息)
        """
        # 使用自定义提示词或默认提示词
        base_prompt = custom_system_prompt.strip() if custom_system_prompt else DEFAULT_SYSTEM_PROMPT
        system_prompt = base_prompt.format(persona=persona)
        
        mode_hint = ""
        if selfie_mode:
            mode_hint = "\n【自拍模式】请以你的角色身份生成一张自拍照的提示词。"
        
        user_prompt = f"""## 最近的聊天记录
{chat_messages}

## 用户的绘图请求
{user_request}
{mode_hint}

请根据以上信息，生成适合的图片提示词。只输出提示词，不要有任何其他内容。"""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            success, response, reasoning, model_name = await llm_api.generate_with_model(
                prompt=full_prompt,
                model_config=model_config_to_use,
                request_type="custom_pic_plugin.prompt_generation",
                temperature=0.7,
                max_tokens=500,
            )
            
            if success and response:
                # 清理响应，移除可能的多余内容
                cleaned_response = response.strip()
                # 移除可能的引号包裹
                if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
                    cleaned_response = cleaned_response[1:-1]
                if cleaned_response.startswith("'") and cleaned_response.endswith("'"):
                    cleaned_response = cleaned_response[1:-1]
                    
                logger.info(f"[PromptGenerator] LLM生成提示词成功，使用模型: {model_name}")
                return True, cleaned_response
            else:
                logger.error(f"[PromptGenerator] LLM生成失败: {response}")
                return False, response or "LLM生成失败"
                
        except Exception as e:
            logger.error(f"[PromptGenerator] 生成提示词时出错: {e}", exc_info=True)
            return False, str(e)

    @staticmethod
    async def generate_prompt_with_style(
        user_request: str,
        chat_messages: str,
        persona: str,
        selfie_mode: bool,
        model_config_to_use,
        custom_system_prompt: str = "",
    ) -> Tuple[bool, str, Optional[str]]:
        """
        使用LLM生成图片提示词和风格判断
        
        Args:
            user_request: 用户的绘图请求
            chat_messages: 最近的聊天记录
            persona: 人设信息
            selfie_mode: 是否为自拍模式
            model_config_to_use: 使用的模型配置
            custom_system_prompt: 自定义系统提示词（留空则使用默认）
            
        Returns:
            Tuple[bool, str, Optional[str]]: (是否成功, 生成的提示词或错误信息, 风格)
        """
        # 使用自定义提示词或默认提示词
        base_prompt = custom_system_prompt.strip() if custom_system_prompt else DEFAULT_SYSTEM_PROMPT
        system_prompt = base_prompt.format(persona=persona)
        
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
        
        try:
            success, response, reasoning, model_name = await llm_api.generate_with_model(
                prompt=full_prompt,
                model_config=model_config_to_use,
                request_type="custom_pic_plugin.prompt_generation",
                temperature=0.7,
                max_tokens=500,
            )
            
            if success and response:
                # 使用 LLMOutputParser 解析响应
                prompt, style = LLMOutputParser.parse(response)
                
                if prompt:
                    logger.info(f"[PromptGenerator] LLM生成提示词成功，使用模型: {model_name}, 风格: {style}")
                    return True, prompt, style
                else:
                    # 如果解析后 prompt 为空，使用原始响应
                    logger.warning(f"[PromptGenerator] 解析后 prompt 为空，使用原始响应")
                    return True, response.strip(), style
            else:
                logger.error(f"[PromptGenerator] LLM生成失败: {response}")
                return False, response or "LLM生成失败", None
                
        except Exception as e:
            logger.error(f"[PromptGenerator] 生成提示词时出错: {e}", exc_info=True)
            return False, str(e), None


# ===== Action组件 =====

class CustomPicAction(BaseAction):
    """生成一张图片并发送"""

    # 激活设置
    activation_type = ActionActivationType.LLM_JUDGE
    mode_enable = ChatMode.ALL
    parallel_action = False

    # 动作基本信息
    action_name = "draw_picture"
    action_description = "可以根据特定的描述，生成并发送一张图片，当没有特定描述时可以参考聊天记录，你可以立刻画好，不用等待"

    # LLM判定提示词
    llm_judge_prompt = """
此动作让你能够生成并发送图片，用于回应群友想要"看到"某些视觉内容的请求。

【触发条件】当群友想要看到以下内容时使用：
1. 你当前的状态/环境/正在做的事（自拍、你在哪、你在干嘛）
2. 你拍的照片/摄影作品（发张你拍的照片、看看你的摄影）
3. 你正在吃/喝/用的东西（你在吃什么、给我看看）
4. 你画的画/创作的图（画一张、帮我画个）
5. 某个具体场景/角色/事物的图片（我想看看...的样子）

【典型触发语句示例】
- "自拍/来张自拍/发张照片看看"
- "你现在在哪/在干嘛，发张图看看"
- "你在吃什么/喝什么，给我看看"
- "发张你拍的照片/看看你的摄影作品"
- "画一张.../帮我画个..."
- "我想看看...长什么样"

【禁止触发】
- 纯文字聊天、问答、讨论（不涉及"看图"需求）
- 只是提到图片相关词汇但不是要求生成
- 讨论或评价已经存在的图片
- 用户明确表示不需要图片
- 并不是对你提出的看图需求
- 前面聊天记录中你已经发过图片时，禁止再次生成并发送图片
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
        "如果最近的消息内你发过图片请不要选择此动作"
    ]
    associated_types = ["text", "image"]

    async def execute(self) -> Tuple[bool, Optional[str]]:
        """执行图片生成动作"""
        logger.info(f"{self.log_prefix} 执行图片生成动作")

        # 获取用户请求
        original_description = self.action_data.get("description", "")
        if not original_description:
            original_description = ""
        original_description = original_description.strip()
        
        # 检查自拍模式
        selfie_mode = self.action_data.get("selfie_mode", False)
        if isinstance(selfie_mode, str):
            selfie_mode = selfie_mode.lower() in ['true', '1', 'yes', 'on']

        # 获取聊天记录
        chat_messages_str = await self._get_recent_chat_messages()
        
        # 获取人设信息
        persona = self._get_persona()
        
        # 获取LLM模型配置
        llm_model_config = self._get_llm_model_config()
        
        # 记录日志
        if selfie_mode:
            logger.info(f"{self.log_prefix} 开始生成自拍图片...")
        else:
            logger.info(f"{self.log_prefix} 开始生成图片...")
        
        # 获取自定义系统提示词
        custom_system_prompt = self.get_config("llm.system_prompt", "")
        
        # 使用LLM生成提示词（带风格判断）
        success, generated_prompt, llm_style = await PromptGenerator.generate_prompt_with_style(
            user_request=original_description or "根据聊天内容生成一张合适的图片",
            chat_messages=chat_messages_str,
            persona=persona,
            selfie_mode=selfie_mode,
            model_config_to_use=llm_model_config,
            custom_system_prompt=custom_system_prompt,
        )
        
        if not success:
            logger.error(f"{self.log_prefix} 生成提示词失败: {generated_prompt}")
            return False, f"提示词生成失败: {generated_prompt}"
        
        logger.info(f"{self.log_prefix} LLM生成的提示词: {generated_prompt[:200]}..., 风格: {llm_style}")

        # 创建风格路由器并决定使用哪个模型
        style_router = StyleRouter(self.config)
        selected_style, model_config, route_reason = style_router.route(
            selfie_mode=selfie_mode,
            manual_style=None,  # Action 模式不支持手动指定风格
            llm_style=llm_style,
        )
        
        logger.info(f"{self.log_prefix} 风格路由结果: style={selected_style}, reason={route_reason}")
        
        # 构建最终提示词（使用风格特定的附加提示词）
        final_prompt = self._build_final_prompt(generated_prompt, model_config)
        logger.info(f"{self.log_prefix} 最终提示词: {final_prompt[:200]}...")
        
        # 检查是否有可用的模型配置
        if model_config is None:
            # 回退到旧的配置方式（向后兼容）
            logger.warning(f"{self.log_prefix} 没有配置风格模型，使用旧的 api 配置")
            api_type = self.get_config("api.api_type", "openai")
            http_base_url = self.get_config("api.base_url")
            http_api_key = self.get_config("api.api_key", "")
            default_model = self.get_config("generation.default_model", "gpt-image-1")
            image_size = self.get_config("generation.default_size", "")
            
            # 检查 base_url 是否配置
            if not http_base_url:
                logger.error(f"{self.log_prefix} API配置缺失: base_url 未配置")
                return False, "API base_url 未配置"

            # OpenAI 格式需要检查 api_key，Gradio 格式不需要
            if api_type.lower() != "gradio":
                if not http_api_key or http_api_key == "YOUR_API_KEY_HERE" or not http_api_key.strip():
                    logger.error(f"{self.log_prefix} API密钥未配置")
                    return False, "API密钥未配置"
        else:
            # 使用风格模型配置
            api_type = model_config.get("api_type", "openai")
            http_base_url = model_config.get("base_url", "")
            http_api_key = model_config.get("api_key", "")
            default_model = model_config.get("model_name", "")
            image_size = self.get_config("generation.default_size", "")
            
            # 检查 base_url 是否配置
            if not http_base_url:
                logger.error(f"{self.log_prefix} {selected_style} 模型的 base_url 未配置")
                return False, f"{selected_style} 模型的 base_url 未配置"

            # OpenAI 格式需要检查 api_key，Gradio 格式不需要
            if api_type.lower() != "gradio":
                if not http_api_key or not http_api_key.strip():
                    logger.error(f"{self.log_prefix} {selected_style} 模型的 API密钥未配置")
                    return False, f"{selected_style} 模型的 API密钥未配置"

        try:
            if api_type.lower() == "gradio":
                # 使用风格模型的 Gradio 参数
                gradio_params = None
                if model_config:
                    gradio_params = {
                        "resolution": model_config.get("gradio_resolution", "1024x1024 ( 1:1 )"),
                        "steps": model_config.get("gradio_steps", 8),
                        "shift": model_config.get("gradio_shift", 3),
                        "timeout": model_config.get("gradio_timeout", 120),
                    }
                success, result = await asyncio.to_thread(
                    self._make_gradio_image_request,
                    prompt=final_prompt,
                    base_url=http_base_url,
                    gradio_params=gradio_params,
                )
            elif api_type.lower() == "sd_api":
                # 使用 SD API
                sd_params = None
                if model_config:
                    sd_params = {
                        "negative_prompt": model_config.get("sd_negative_prompt", ""),
                        "width": model_config.get("sd_width", 512),
                        "height": model_config.get("sd_height", 512),
                        "steps": model_config.get("sd_steps", 20),
                        "cfg": model_config.get("sd_cfg", 7.0),
                        "model_index": model_config.get("sd_model_index", 0),
                        "seed": model_config.get("sd_seed", -1),
                    }
                success, result = await asyncio.to_thread(
                    self._make_sd_api_request,
                    prompt=final_prompt,
                    base_url=http_base_url,
                    api_key=http_api_key,
                    sd_params=sd_params,
                )
            else:
                success, result = await asyncio.to_thread(
                    self._make_http_image_request,
                    prompt=final_prompt,
                    model=default_model,
                    size=image_size if image_size else None,
                    base_url=http_base_url,
                    api_key=http_api_key,
                )
        except Exception as e:
            logger.error(f"{self.log_prefix} 图片生成请求失败: {e!r}", exc_info=True)
            success = False
            result = f"图片生成服务遇到问题: {str(e)[:100]}"

        if success:
            return await self._handle_image_result(result)
        else:
            logger.error(f"{self.log_prefix} 图片生成失败: {result}")
            return False, f"图片生成失败: {result}"

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
        """下载图片并编码为Base64"""
        try:
            with urllib.request.urlopen(image_url, timeout=60) as response:
                if response.status == 200:
                    image_bytes = response.read()
                    
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
                        # 有些 API 返回 {"data": {"image": "..."}}
                        data_obj = response_data["data"]
                        if isinstance(data_obj, dict):
                            image_data = data_obj.get("image") or data_obj.get("url")
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

    def _make_http_image_request(
        self, 
        prompt: str, 
        model: str, 
        size: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Tuple[bool, str]:
        """发送HTTP请求生成图片（使用chat/completions端点）"""
        import re
        
        # 使用传入的参数或从配置获取
        if base_url is None:
            base_url = self.get_config("api.base_url", "")
        if api_key is None:
            api_key = self.get_config("api.api_key", "")

        endpoint = f"{base_url.rstrip('/')}/chat/completions"

        # 使用chat completions格式
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        logger.info(f"{self.log_prefix} 发起图片请求: {model}, Prompt: {prompt[:100]}...")

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

class DirectPicCommand(BaseCommand):
    """直接生成图片的指令，跳过LLM提示词生成"""

    command_name = "direct_pic"
    command_description = "直接使用提供的prompt生成图片，不经过LLM处理。支持 /pic anime <prompt> 或 /pic real <prompt> 指定风格"
    # 支持 /pic <prompt>, /pic anime <prompt>, /pic real <prompt>
    command_pattern = r"^/pic\s+(?:(?P<style>anime|real)\s+)?(?P<prompt>.+)$"

    def __init__(self, message: MessageRecv, plugin_config: Optional[dict] = None):
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
        
        # 检查是否以 anime 或 real 开头
        for style in ["anime", "real"]:
            if lower_prompt.startswith(style + " "):
                remaining = prompt[len(style):].strip()
                return style, remaining
        
        return None, prompt

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        """执行直接图片生成"""
        # 获取用户输入的prompt和风格
        raw_prompt = self.matched_groups.get("prompt", "").strip()
        manual_style = self.matched_groups.get("style")
        
        if not raw_prompt:
            await self.send_text("请提供图片描述，例如: /pic a cute cat 或 /pic anime 一个可爱的女孩")
            return True, None, True
        
        # 如果正则没有匹配到风格，尝试从 prompt 中解析
        if not manual_style:
            manual_style, raw_prompt = self.parse_style_from_prompt(raw_prompt)
        
        logger.info(f"{self.log_prefix} 收到直接生图指令，style: {manual_style}, prompt: {raw_prompt[:100]}...")
        
        # 创建风格路由器
        style_router = StyleRouter(self.config)
        
        # 路由到对应的模型
        selected_style, model_config, route_reason = style_router.route(
            selfie_mode=False,
            manual_style=manual_style,
            llm_style=None,
        )
        
        logger.info(f"{self.log_prefix} 风格路由结果: style={selected_style}, reason={route_reason}")
        
        # 检查是否有可用的模型配置
        if model_config is None:
            # 回退到旧的配置方式（向后兼容）
            logger.warning(f"{self.log_prefix} 没有配置风格模型，使用旧的 api 配置")
            api_type = self.get_config("api.api_type", "openai")
            http_base_url = self.get_config("api.base_url")
            http_api_key = self.get_config("api.api_key", "")
            default_model = self.get_config("generation.default_model", "gpt-image-1")
            
            if not http_base_url:
                await self.send_text("API配置错误：base_url 未配置")
                return False, None, True

            if api_type.lower() not in ["gradio"]:
                if not http_api_key or http_api_key == "YOUR_API_KEY_HERE" or not http_api_key.strip():
                    await self.send_text("API配置错误：api_key 未配置")
                    return False, None, True
            
            gradio_params = None
            sd_params = None
        else:
            # 使用风格模型配置
            api_type = model_config.get("api_type", "openai")
            http_base_url = model_config.get("base_url", "")
            http_api_key = model_config.get("api_key", "")
            default_model = model_config.get("model_name", "")
            
            # 检查配置
            if not http_base_url:
                # 检查手动指定的风格是否可用
                if manual_style and not style_router.is_style_available(manual_style):
                    available = style_router.get_available_styles()
                    await self.send_text(f"{manual_style} 风格未配置。可用风格: {', '.join(available) if available else '无'}")
                    return False, None, True
                await self.send_text(f"{selected_style} 模型的 base_url 未配置")
                return False, None, True

            if api_type.lower() not in ["gradio", "sd_api"]:
                if not http_api_key or not http_api_key.strip():
                    await self.send_text(f"{selected_style} 模型的 API密钥未配置")
                    return False, None, True
            elif api_type.lower() == "sd_api":
                if not http_api_key or not http_api_key.strip():
                    await self.send_text(f"{selected_style} 模型的 API密钥未配置")
                    return False, None, True
            
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

        # 优先使用风格特定的附加提示词，否则使用全局附加提示词
        custom_prompt_add = ""
        if model_config and model_config.get("custom_prompt_add"):
            custom_prompt_add = model_config.get("custom_prompt_add", "")
        else:
            custom_prompt_add = self.get_config("generation.custom_prompt_add", "")
        
        if custom_prompt_add and custom_prompt_add.strip():
            final_prompt = f"{custom_prompt_add.strip()}, {raw_prompt}"
        else:
            final_prompt = raw_prompt
        
        logger.info(f"{self.log_prefix} 最终提示词: {final_prompt[:200]}...")

        # 获取图片尺寸配置
        image_size = self.get_config("generation.default_size", "")

        try:
            if api_type.lower() == "gradio":
                success, result = await asyncio.to_thread(
                    self._make_gradio_image_request,
                    prompt=final_prompt,
                    base_url=http_base_url,
                    gradio_params=gradio_params,
                )
            elif api_type.lower() == "sd_api":
                success, result = await asyncio.to_thread(
                    self._make_sd_api_request,
                    prompt=final_prompt,
                    base_url=http_base_url,
                    api_key=http_api_key,
                    sd_params=sd_params if model_config else None,
                )
            else:
                success, result = await asyncio.to_thread(
                    self._make_http_image_request,
                    prompt=final_prompt,
                    model=default_model,
                    size=image_size if image_size else None,
                    base_url=http_base_url,
                    api_key=http_api_key,
                )
        except Exception as e:
            logger.error(f"{self.log_prefix} 图片生成请求失败: {e!r}", exc_info=True)
            await self.send_text(f"图片生成失败: {str(e)[:100]}")
            return False, None, True

        if success:
            return await self._handle_image_result(result)
        else:
            logger.error(f"{self.log_prefix} 图片生成失败: {result}")
            await self.send_text(f"图片生成失败: {result}")
            return False, None, True

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
        """下载图片并编码为Base64"""
        try:
            with urllib.request.urlopen(image_url, timeout=60) as response:
                if response.status == 200:
                    image_bytes = response.read()
                    
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
                            image_data = data_obj.get("image") or data_obj.get("url")
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

    def _make_http_image_request(
        self, prompt: str, model: str, size: Optional[str] = None
    ) -> Tuple[bool, str]:
        """发送HTTP请求生成图片"""
        import re
        
        base_url = self.get_config("api.base_url", "")
        api_key = self.get_config("api.api_key", "")

        endpoint = f"{base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        logger.info(f"{self.log_prefix} 发起图片请求: {model}, Prompt: {prompt[:100]}...")

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

@register_plugin
class CustomPicPlugin(BasePlugin):
    """使用LLM生成提示词的图片生成插件"""
    
    plugin_name = "MaiBot_LLM2pic"
    plugin_version = "3.1.0"
    plugin_author = "Ptrel"
    enable_plugin = True
    dependencies: List[str] = []
    python_dependencies: List[str] = []
    config_file_name = "config.toml"
    
    config_section_descriptions = {
        "plugin": "插件基本配置",
        "api": "图片生成API配置",
        "generation": "图片生成参数配置",
        "llm": "LLM模型配置（用于生成提示词）",
        "components": "组件启用配置",
    }

    config_schema = {
        "plugin": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用插件"
            ),
        },
        "generation": {
            "default_style": ConfigField(
                type=str,
                default="anime",
                description="默认风格：anime（二次元）或 real（写实）"
            ),
            "custom_prompt_add": ConfigField(
                type=str,
                default="",
                description="全局附加提示词（会添加到LLM生成的提示词前面，可留空）"
            ),
            "crop_enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用图片裁切（用于去除AI生成的水印）"
            ),
            "crop_position": ConfigField(
                type=str,
                default="bottom",
                description="裁切位置：top（顶部）、bottom（底部）、left（左侧）、right（右侧）"
            ),
            "crop_pixels": ConfigField(
                type=int,
                default=40,
                description="裁切像素数"
            ),
        },
        "anime": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用 anime（二次元）风格的独立模型配置。如果不启用，将使用 [api] 节的配置"
            ),
            "api_type": ConfigField(
                type=str,
                default="gradio",
                description="API类型：openai、gradio 或 sd_api"
            ),
            "base_url": ConfigField(
                type=str,
                default="https://tongyi-mai-z-image-turbo.hf.space",
                description="API 基础 URL"
            ),
            "api_key": ConfigField(
                type=str,
                default="",
                description="API 密钥（Gradio 可留空）",
                required=False
            ),
            "model_name": ConfigField(
                type=str,
                default="",
                description="模型名称（OpenAI 格式需要）"
            ),
            "custom_prompt_add": ConfigField(
                type=str,
                default="",
                description="该风格专用的附加提示词（留空则使用全局 generation.custom_prompt_add）"
            ),
            "gradio_resolution": ConfigField(
                type=str,
                default="1024x1024 ( 1:1 )",
                description="Gradio 图片分辨率"
            ),
            "gradio_steps": ConfigField(
                type=int,
                default=8,
                description="Gradio 推理步数"
            ),
            "gradio_shift": ConfigField(
                type=int,
                default=3,
                description="Gradio 时间偏移参数"
            ),
            "gradio_timeout": ConfigField(
                type=int,
                default=120,
                description="Gradio 轮询超时时间（秒）"
            ),
            "sd_negative_prompt": ConfigField(
                type=str,
                default="",
                description="SD API 负面提示词"
            ),
            "sd_width": ConfigField(
                type=int,
                default=512,
                description="SD API 图像宽度 (64-2048)"
            ),
            "sd_height": ConfigField(
                type=int,
                default=512,
                description="SD API 图像高度 (64-2048)"
            ),
            "sd_steps": ConfigField(
                type=int,
                default=20,
                description="SD API 生成步数 (1-50)"
            ),
            "sd_cfg": ConfigField(
                type=float,
                default=7.0,
                description="SD API CFG引导强度 (1-10)"
            ),
            "sd_model_index": ConfigField(
                type=int,
                default=0,
                description="SD API 模型索引"
            ),
            "sd_seed": ConfigField(
                type=int,
                default=-1,
                description="SD API 随机种子，-1为随机"
            ),
        },
        "real": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用 real（写实）模型"
            ),
            "api_type": ConfigField(
                type=str,
                default="openai",
                description="API类型：openai、gradio 或 sd_api"
            ),
            "base_url": ConfigField(
                type=str,
                default="",
                description="API 基础 URL"
            ),
            "api_key": ConfigField(
                type=str,
                default="",
                description="API 密钥",
                required=False
            ),
            "model_name": ConfigField(
                type=str,
                default="gpt-image-1",
                description="模型名称"
            ),
            "custom_prompt_add": ConfigField(
                type=str,
                default="",
                description="该风格专用的附加提示词（留空则使用全局 generation.custom_prompt_add）"
            ),
            "gradio_resolution": ConfigField(
                type=str,
                default="1024x1024 ( 1:1 )",
                description="Gradio 图片分辨率"
            ),
            "gradio_steps": ConfigField(
                type=int,
                default=8,
                description="Gradio 推理步数"
            ),
            "gradio_shift": ConfigField(
                type=int,
                default=3,
                description="Gradio 时间偏移参数"
            ),
            "gradio_timeout": ConfigField(
                type=int,
                default=120,
                description="Gradio 轮询超时时间（秒）"
            ),
            "sd_negative_prompt": ConfigField(
                type=str,
                default="",
                description="SD API 负面提示词"
            ),
            "sd_width": ConfigField(
                type=int,
                default=512,
                description="SD API 图像宽度 (64-2048)"
            ),
            "sd_height": ConfigField(
                type=int,
                default=512,
                description="SD API 图像高度 (64-2048)"
            ),
            "sd_steps": ConfigField(
                type=int,
                default=20,
                description="SD API 生成步数 (1-50)"
            ),
            "sd_cfg": ConfigField(
                type=float,
                default=7.0,
                description="SD API CFG引导强度 (1-10)"
            ),
            "sd_model_index": ConfigField(
                type=int,
                default=0,
                description="SD API 模型索引"
            ),
            "sd_seed": ConfigField(
                type=int,
                default=-1,
                description="SD API 随机种子，-1为随机"
            ),
        },
        "llm": {
            "model_name": ConfigField(
                type=str,
                default="",
                description="用于生成提示词的LLM模型名称（留空则使用系统默认的planner模型）"
            ),
            "system_prompt": ConfigField(
                type=str,
                default="",
                description="调用LLM生成提示词时的系统提示词（留空则使用默认提示词）。支持 {persona} 占位符用于插入人设信息。"
            ),
            "context_message_limit": ConfigField(
                type=int,
                default=20,
                description="传递给LLM的聊天记录条数上限（1-100）"
            ),
            "context_time_minutes": ConfigField(
                type=int,
                default=30,
                description="获取聊天记录的时间范围（分钟）"
            ),
        },
        "components": {
            "enable_image_generation": ConfigField(
                type=bool, 
                default=True, 
                description="是否启用图片生成Action（LLM智能触发）"
            ),
            "enable_direct_pic_command": ConfigField(
                type=bool,
                default=True,
                description="是否启用 /pic 指令（直接透传prompt到生图API）"
            ),
        },
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        """返回插件包含的组件列表"""
        components = []
        if self.get_config("components.enable_image_generation", True):
            components.append((CustomPicAction.get_action_info(), CustomPicAction))
        if self.get_config("components.enable_direct_pic_command", True):
            components.append((DirectPicCommand.get_command_info(), DirectPicCommand))
        return components
