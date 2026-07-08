_CONFIG_VERSION = "4.2.0"


"""
LLM2Pic 插件配置模型。

所有 PluginConfigBase 子类从 plugin.py 抽出，便于维护。
"""

from typing import Any, Literal
from maibot_sdk import Field, PluginConfigBase

from .wd14_client import DEFAULT_ENDPOINT as WD14_DEFAULT_ENDPOINT


class PluginSectionConfig(PluginConfigBase):
    """总开关与配置版本。关闭后插件不响应 /pic 与画图 Tool。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="关闭后整个 LLM2Pic 插件停用（/pic、draw_picture、edit_picture 均不可用）。")
    config_version: str = Field(default=_CONFIG_VERSION, description="配置格式版本号，由插件迁移使用；一般勿手改。")




class RefImageConfig(PluginConfigBase):
    """参考图模式默认参数（/pic i2i|char-ref|vibe 或 draw_picture 显式参考图时生效；日常文生图不受影响）。"""

    __ui_label__ = "参考图"
    __ui_icon__ = "images"
    __ui_order__ = 2

    i2i_strength: float = Field(
        default=0.7,
        ge=0.01,
        le=0.99,
        description="i2i 图生图：结构保留强度。越低越像原图姿势/构图，越高越按新 prompt 重画。常用 0.6–0.8 只借骨架不借画风。",
    )
    i2i_noise: float = Field(
        default=0.0,
        ge=0.0,
        le=0.99,
        description="i2i 注入噪声量。0 最稳；略增可减轻与原图粘连，过大易崩。",
    )
    char_ref_type: Literal["character", "style", "character&style"] = Field(
        default="character",
        description="char-ref 参考类型：character 锁角色、style 借画风、character&style 两者兼有。",
    )
    char_ref_fidelity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="char-ref 对参考图特征的忠实度，越高越贴近参考角色/风格。",
    )
    char_ref_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="char-ref 参考强度，与 fidelity 配合调节参考图影响力。",
    )
    vibe_info_extracted: float = Field(
        default=0.4,
        ge=0.01,
        le=1.0,
        description="vibe 从参考图提取氛围/笔触信息的强度。偏低可避免把参考图角色五官硬迁过来。",
    )
    vibe_strength: float = Field(
        default=0.3,
        ge=0.01,
        le=1.0,
        description="vibe 迁移到成图上的强度。越大画风越像参考，过大易糊脸。",
    )
    vibe_global_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="vibe 全局权重系数，微调 vibe 在整张图上的占比。",
    )
    min_image_size: int = Field(
        default=256,
        ge=64,
        le=4096,
        description="附图最短边低于此像素时跳过参考图模式，自动退回普通文生图，避免无效或超时。",
    )


class GenerationConfig(PluginConfigBase):
    """所有出图路径共用的默认风格、全局 tag 前缀与可选裁切。"""

    __ui_label__ = "生成"
    __ui_icon__ = "wand-sparkles"
    __ui_order__ = 1

    default_style: Literal["anime", "edit"] = Field(default="anime", description="Planner 或 /pic 未指定 anime/edit 时使用的风格路由。自拍模式仍强制 anime。")
    custom_prompt_add: str = Field(default="", description="拼到最终正向 prompt 最前的全局 tag（如画师串、质量词）。各端点下的 custom_prompt_add 会覆盖本项。")
    crop_enabled: bool = Field(default=False, description="发送前是否裁掉图片边缘（常用于去掉底部水印条）。")
    crop_position: Literal["top", "bottom", "left", "right"] = Field(default="bottom", description="裁切条所在边：top/bottom/left/right，配合 crop_pixels 使用。")
    crop_pixels: int = Field(default=40, ge=0, description="从指定边裁掉的像素宽度/高度。")

    ref_image: RefImageConfig = Field(
        default_factory=RefImageConfig,
        description="i2i / char-ref / vibe 参考图默认参数（用户显式指定参考图时读取）。",
    )


class GenerationGuardConfig(PluginConfigBase):
    """限制 Planner 自动调用 draw_picture 的频率与并发，防止刷屏与误触。"""

    __ui_label__ = "出图保护"
    __ui_icon__ = "shield"
    __ui_order__ = 3

    enabled: bool = Field(default=True, description="总开关：关闭后不再做节流/锁/负向意图拦截（不推荐在公开群关闭）。")
    pending_lock_enabled: bool = Field(default=True, description="同一群/私聊同时只允许一个生图后台任务，避免并发抢 API 与重复出图。")
    negative_intent_block_enabled: bool = Field(default=True, description="用户说「别画」「不要图」等时，拦截 Planner 主动 draw_picture（/pic 命令不受影响）。")
    explicit_request_min_interval_seconds: int = Field(default=30, ge=0, description="用户明确要图时，同一聊天流两次 draw_picture 的最短间隔（秒），0 表示不限制。")
    proactive_min_interval_seconds: int = Field(default=240, ge=0, description="Planner 未经用户明确要求就画图时的最短间隔（秒），建议 ≥120 防刷图。")


class RegexUrlEndpointConfig(PluginConfigBase):
    """regex_url 端点参数。"""

    base_url: str = Field(default="", description="简易出图 URL：$1 替换为 URL 编码后的完整 prompt。适合自建图床/代理。")
    custom_prompt_add: str = Field(default="{{{masterpiece,best quality}}},", description="仅本端点生效的正向 tag 前缀，会覆盖 [generation] 的 custom_prompt_add。")


class OpenAIEndpointConfig(PluginConfigBase):
    """OpenAI 兼容端点参数。"""

    base_url: str = Field(default="", description="OpenAI 兼容接口根地址（含 /v1）。用于 DALL·E 类或 edit 图生图。")
    api_key: str = Field(default="", description="该端点 API Key，WebUI 中以密码框显示。", json_schema_extra={"input_type": "password"})
    model_name: str = Field(default="", description="图像模型 ID（如 gpt-image-1、dall-e-3）。")
    size: str = Field(default="", description="生成尺寸（如 1024x1024），留空则用接口默认。")
    custom_prompt_add: str = Field(default="", description="该端点专用正向提示词前缀")


class GradioEndpointConfig(PluginConfigBase):
    """Gradio 端点参数。"""

    base_url: str = Field(default="https://tongyi-mai-z-image-turbo.hf.space", description="Hugging Face Gradio Space 根 URL，用于 Z-Image 等远程推理。")
    custom_prompt_add: str = Field(default="", description="该端点专用正向提示词前缀")
    resolution: str = Field(default="1024x1024 ( 1:1 )", description="Gradio 下拉中的分辨率选项字符串，需与 Space 支持项一致。")
    steps: int = Field(default=8, ge=1, le=50, description="采样/推理步数。步数越多通常越慢、细节可能更好。")
    shift: int = Field(default=3, description="Gradio Z-Image 的 shift 参数，影响噪声调度，一般保持默认。")
    timeout: int = Field(default=120, ge=1, description="等待 Gradio 任务完成的最长时间，超时则报失败。")


class SdApiEndpointConfig(PluginConfigBase):
    """Stable Diffusion API 端点参数。"""

    base_url: str = Field(default="", description="Automatic1111 / Forge 等 WebUI 的 txt2img API 地址。")
    api_key: str = Field(default="", description="API 密钥", json_schema_extra={"input_type": "password"})
    custom_prompt_add: str = Field(default="", description="该端点专用正向提示词前缀")
    negative_prompt: str = Field(default="", description="该端点使用的 negative prompt。")
    width: int = Field(default=832, ge=1, description="出图宽度（像素）。")
    height: int = Field(default=1216, ge=1, description="出图高度（像素）。")
    steps: int = Field(default=28, ge=1, description="推理步数")
    cfg: float = Field(default=7, description="Classifier-Free Guidance，越大越贴 prompt，过大易过饱和。")
    model_index: int = Field(default=9, ge=0, description="SD WebUI 多模型列表中的索引下标。")
    seed: int = Field(default=-1, description="固定种子可复现同一张图；-1 为每次随机。")


class NovelAIEndpointConfig(PluginConfigBase):
    """NovelAI 官方 API 端点参数。"""

    api_key: str = Field(default="", description="NovelAI 官方 API 的 Bearer Token。", json_schema_extra={"input_type": "password"})
    custom_prompt_add: str = Field(default="", description="该端点专用正向提示词前缀")
    model: str = Field(default="nai-diffusion-4-5-full", description="如 nai-diffusion-4-5-full。")
    width: int = Field(default=832, ge=1, description="图片宽度")
    height: int = Field(default=1216, ge=1, description="图片高度")
    steps: int = Field(default=28, ge=1, le=50, description="推理步数")
    scale: float = Field(default=5.0, description="CFG / scale，控制 prompt 遵循程度。")
    sampler: str = Field(default="k_euler", description="如 k_euler、k_euler_ancestral。")
    negative_prompt: str = Field(default="", description="负向提示词")
    seed: int = Field(default=-1, description="随机种子，-1 表示随机")
    timeout: int = Field(default=120, ge=1, description="单次 HTTP 请求最长等待时间。")


class NewApiNaiEndpointConfig(PluginConfigBase):
    """NewAPI NAI 端点参数。"""

    base_url: str = Field(default="", description="NewAPI 网关根地址（OpenAI 兼容 /v1），用于 NAI 代理出图。")
    api_key: str = Field(default="", description="NewAPI 的 API Key。", json_schema_extra={"input_type": "password"})
    model_name: str = Field(default="nai-diffusion-4-5-full", description="NewAPI 上登记的 NAI 模型名，如 nai-diffusion-4-5-full。")
    custom_prompt_add: str = Field(default="{{{masterpiece,best quality}}},", description="拼在 LLM 生成 tag 之前的画师串/质量词（NAI 常用 {{{artist}}}, masterpiece 等）。")
    negative_prompt: str = Field(default="lowres, bad anatomy, bad hands, text, watermark", description="负向提示词")
    size: str = Field(default="portrait", description="画幅：portrait/landscape/square 或宽x高。与 NAI 常用竖图/横图一致。")
    steps: int = Field(default=23, ge=1, le=28, description="NAI 步数，NewAPI 通道通常上限 28。")
    scale: float = Field(default=5, description="提示词引导强度")
    sampler: str = Field(default="k_euler_ancestral", description="采样器")
    seed: int = Field(default=-1, description="随机种子，-1 表示随机")
    image_format: Literal["png", "webp"] = Field(default="png", description="API 返回 png 或 webp。")
    max_tokens: int = Field(default=100000, ge=1, description="NewAPI 计费/预算相关上限，按网关说明填写。")
    timeout: int = Field(default=180, ge=1, description="请求超时时间（秒）")
    retry_attempts: int = Field(default=3, ge=1, le=5, description="限流或短暂故障时的自动重试次数。")
    proxy_mode: Literal["auto", "inherit", "direct"] = Field(default="auto", description="出图 HTTP 代理：auto 自动、inherit 继承 MaiBot、direct 直连。")
    quality_toggle: bool = Field(default=True, description="对应 NAI qualityToggle，开启可略提质量（视模型/额度）。")
    auto_smea: bool = Field(default=False, description="NAI autoSmea 增强开关。")
    variety_boost: bool = Field(default=False, description="NAI variety_boost，增加画面多样性。")
    extra_params: dict[str, Any] = Field(default_factory=dict, description="高级：合并进请求体的键值对，勿乱填以免 API 报错。")


class AnimeConfig(PluginConfigBase):
    """二次元主通道：选择 regex_url / NewAPI NAI / OpenAI / Gradio / SD / 官方 NAI 之一。"""

    __ui_label__ = "Anime"
    __ui_icon__ = "image"
    __ui_order__ = 2

    enabled: bool = Field(default=True, description="关闭后无法走 anime 路由（含自拍与默认文生图）。")
    api_type: Literal["regex_url", "newapi_nai", "openai", "gradio", "sd_api", "novelai"] = Field(
        default="gradio",
        description="实际调用的后端：推荐 newapi_nai；其余为备用或测试。",
    )
    regex_url: RegexUrlEndpointConfig = Field(default_factory=RegexUrlEndpointConfig, description="当 api_type=regex_url 时展开的配置。")
    newapi_nai: NewApiNaiEndpointConfig = Field(default_factory=NewApiNaiEndpointConfig, description="当 api_type=newapi_nai 时展开的配置（主路径）。")
    openai: OpenAIEndpointConfig = Field(default_factory=OpenAIEndpointConfig, description="当 api_type=openai 时展开的配置。")
    gradio: GradioEndpointConfig = Field(default_factory=GradioEndpointConfig, description="当 api_type=gradio 时展开的配置。")
    sd_api: SdApiEndpointConfig = Field(default_factory=SdApiEndpointConfig, description="当 api_type=sd_api 时展开的配置。")
    novelai: NovelAIEndpointConfig = Field(default_factory=NovelAIEndpointConfig, description="当 api_type=novelai 时展开的配置（直连官方）。")


class EditConfig(PluginConfigBase):
    """改图/P 图通道：OpenAI 兼容图生图或文生图，供 edit_picture 与 /pic edit。"""

    __ui_label__ = "Edit"
    __ui_icon__ = "image-plus"
    __ui_order__ = 4

    enabled: bool = Field(default=False, description="关闭后 edit_picture 与 /pic edit 不可用。")
    api_type: Literal["openai"] = Field(default="openai", description="edit 仅支持 openai 类接口。")
    openai: OpenAIEndpointConfig = Field(default_factory=OpenAIEndpointConfig, description="OpenAI 兼容端点参数")


class LlmConfig(PluginConfigBase):
    """把用户自然语言转成 Danbooru tag 的 LLM：模型、上下文、SFW 与排序策略。"""

    __ui_label__ = "LLM"
    __ui_icon__ = "brain"
    __ui_order__ = 5

    model_name: str = Field(default="", description="留空走 MaiBot 任务模型；填写则为具体模型或任务名（见插件 model 回退逻辑）。")
    context_message_limit: int = Field(default=20, ge=1, le=100, description="写 prompt 时带入的最近消息条数上限，用于补全场景/指代。")
    context_time_minutes: int = Field(default=30, ge=1, le=1440, description="只拉取最近 N 分钟内的消息作为上下文。")
    prompt_mode: Literal["legacy", "danbooru"] = Field(default="danbooru", description="danbooru：结构化 tag + 规则（推荐）；legacy：旧版 JSON prompt。")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="越低 tag 越稳，越高越发散。建议 0.1–0.3。")
    danbooru_sfw_mode: bool = Field(default=True, description="true：默认用 SFW 规则模板并过滤擦边 tag；/pic nsfw 或 nsfw_allowed=true 单次放开。")
    enforce_tag_order: bool = Field(default=True, description="true：把 1girl/镜头词/year 等按习惯前置/后置，利于 NAI 构图；false 保持 LLM 原顺序。")
    selfie_appearance_policy: Literal["auto", "never", "keep"] = Field(default="auto", description="auto：未描述外貌时去掉 persona 外貌 tag 防乱脸；never 总是去掉；keep 总是保留。")
    system_prompt: str = Field(default="", description="追加到 Danbooru 生成器后的本地规则（OC、东雪莲、禁止雪景联想等）。支持多行。", json_schema_extra={"ui_type": "textarea", "rows": 8})


class TagRetrieverConfig(PluginConfigBase):
    """写 prompt 前从 Danbooru 语料检索候选 tag，提高 tag 名准确度。"""

    __ui_label__ = "Tag 检索"
    __ui_icon__ = "tags"
    __ui_order__ = 6

    enabled: bool = Field(default=True, description="关闭则仅靠 LLM 记忆，不查在线/本地 tag 库。")
    mode: Literal["online", "local"] = Field(default="online", description="online：HF/API 语义检索；local：本地索引。")
    api_url: str = Field(default="https://sakizuki-danboorusearch.hf.space/api", description="Danbooru 语义搜索服务根 URL。")
    timeout: float = Field(default=90.0, ge=1.0, le=300.0, description="单次检索请求超时（秒）。")
    search_limit: int = Field(default=30, ge=1, le=500, description="一次查询最多返回的 tag 候选数。")
    search_top_k: int = Field(default=5, ge=1, le=50, description="对用户描述每个片段各召回多少相关 tag。")
    related_limit: int = Field(default=20, ge=0, le=200, description="根据种子 tag 扩展共现 tag 的数量上限。")
    related_seed_count: int = Field(default=8, ge=1, le=50, description="用于共现扩展的初始种子 tag 个数。")
    show_nsfw: bool = Field(default=False, description="true 时候选池可含 R18 tag（仍受 danbooru_sfw_mode 与本次 nsfw 开关约束）。")
    popularity_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="越大越偏向 Danbooru 高热 tag，0 纯语义相似。")
    fallback_local: bool = Field(default=True, description="建议开启，避免 API 挂了就完全没有候选 tag。")
    top_k: int = Field(default=20, ge=1, le=200, description="local 模式下一次返回的 tag 数上限。")
    min_score: float = Field(default=0.3, ge=0.0, le=1.0, description="低于此相似度的本地 tag 会被丢弃。")


class Wd14Config(PluginConfigBase):
    """用户发图或回复引用图时，用 WD14 反推 Danbooru tag 辅助写 prompt。"""

    __ui_label__ = "WD14 反推"
    __ui_icon__ = "scan-image"
    __ui_order__ = 7

    enabled: bool = Field(default=True, description="关闭则不发图/引用图时跳过 WD14（参考图 i2i 仍可用图，只是少 tag 融合）。")
    endpoint: str = Field(default=WD14_DEFAULT_ENDPOINT, description="WD14 Tagger 推理服务地址（如自建或公共 endpoint）。")
    threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="高于此置信度的 tag 才会写入参考信息，过低噪声多、过高 tag 少。")
    timeout: float = Field(default=60.0, ge=5.0, le=300.0, description="WD14 HTTP 请求超时。")
    max_image_size: int = Field(default=1024, ge=128, le=4096, description="发送 WD14 前把长边缩到此值以内，减轻超时与内存。")


class ComponentsConfig(PluginConfigBase):
    """单独开关 Planner 画图 Tool 与 /pic 命令，便于只开一种入口。"""

    __ui_label__ = "组件"
    __ui_icon__ = "toggle-right"
    __ui_order__ = 8

    enable_image_generation: bool = Field(default=True, description="关闭后 Planner 看不到 draw_picture（/pic 仍可用，除非也关 direct_pic）。")
    enable_direct_pic_command: bool = Field(default=True, description="关闭后群内 /pic 不响应；支持回复引用图 + /pic i2i|char-ref|vibe|nsfw。")


class GitHubConfig(PluginConfigBase):
    """出图成功后异步上传到 GitHub 仓库留档（不阻塞发群）。"""

    __ui_label__ = "GitHub 上传"
    __ui_icon__ = "upload-cloud"
    __ui_order__ = 9

    enabled: bool = Field(default=False, description="开启且 token 有效时，发图成功后后台 upload。")
    token: str = Field(default="", description="PAT，需 repo contents:write。留空可尝试从 regex_url 的 git_token 解析（若配置了）。")
    owner: str = Field(default="CharTyr", description="GitHub 用户名或组织名。")
    repo: str = Field(default="my-images", description="存放图片的仓库名称。")
    path_prefix: str = Field(default="images", description="对象路径前缀，实际为 prefix/日期/文件名.png。")
    branch: str = Field(default="main", description="提交到的分支，通常 main。")
    commit_message: str = Field(default="", description="留空则用插件默认说明；可写进 commit 便于网页展示 prompt。")


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
