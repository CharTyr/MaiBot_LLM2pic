_CONFIG_VERSION = "4.2.0"


"""
LLM2Pic 插件配置模型。

所有 PluginConfigBase 子类从 plugin.py 抽出，便于维护。
"""

from typing import Any, Literal
from maibot_sdk import Field, PluginConfigBase

from .wd14_client import DEFAULT_ENDPOINT as WD14_DEFAULT_ENDPOINT


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


class Wd14Config(PluginConfigBase):
    """WD14 反推配置。"""

    __ui_label__ = "WD14 反推"
    __ui_icon__ = "scan-image"
    __ui_order__ = 7

    enabled: bool = Field(default=True, description="是否启用引用图片 WD14 反推")
    endpoint: str = Field(default=WD14_DEFAULT_ENDPOINT, description="WD14 tagger endpoint URL")
    threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="tag 置信度阈值")
    timeout: float = Field(default=60.0, ge=5.0, le=300.0, description="反推请求超时时间（秒）")
    max_image_size: int = Field(default=1024, ge=128, le=4096, description="反推前缩放图片最大边长，避免超大图超时")


class ComponentsConfig(PluginConfigBase):
    """组件启用配置。"""

    __ui_label__ = "组件"
    __ui_icon__ = "toggle-right"
    __ui_order__ = 6

    enable_image_generation: bool = Field(default=True, description="是否启用图片生成 Tool")
    enable_direct_pic_command: bool = Field(default=True, description="是否启用 /pic 指令")


class GitHubConfig(PluginConfigBase):
    """GitHub 自动上传配置。生成的图片发送成功后异步上传到指定仓库。"""

    __ui_label__ = "GitHub 上传"
    __ui_icon__ = "upload-cloud"
    __ui_order__ = 9

    enabled: bool = Field(default=False, description="是否启用自动上传到 GitHub")
    token: str = Field(default="", description="GitHub Personal Access Token（contents:write 权限）；留空时从 anime.regex_url.base_url 的 git_token 参数回退提取")
    owner: str = Field(default="CharTyr", description="目标仓库 owner")
    repo: str = Field(default="my-images", description="目标仓库名")
    path_prefix: str = Field(default="images", description="仓库内路径前缀，按日期分文件夹：<prefix>/<YYYY-MM-DD>/<文件名>")
    branch: str = Field(default="main", description="目标分支")
    commit_message: str = Field(default="", description="自定义 commit message（留空使用默认）")


class LLM2PicPluginConfig(PluginConfigBase):
    """LLM2PIC 插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    generation_guard: GenerationGuardConfig = Field(default_factory=GenerationGuardConfig)
    anime: AnimeConfig = Field(default_factory=AnimeConfig)
    edit: EditConfig = Field(default_factory=EditConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    tag_retriever: TagRetrieverConfig = Field(default_factory=TagRetrieverConfig)
    wd14: Wd14Config = Field(default_factory=Wd14Config)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
