"""
图片生成客户端基类。

定义 GenerationContext（统一上下文）和 ImageClient ABC。
所有具体 API 客户端（newapi_nai / gradio / sd_api 等）继承 ImageClient。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationContext:
    """传给 ImageClient 的统一上下文。

    包含所有 NAI API 支持的参数字段。
    不需要的参考图模式字段保持 None / 空值，client 只取需要的。
    """

    # ── 基础参数 ──
    prompt: str = ""
    negative_prompt: str = ""
    size: tuple[int, int] = (832, 1216)
    model: str = "nai-diffusion-4-5-full"
    steps: int = 28
    scale: float = 5.0
    sampler: str = "k_euler_ancestral"
    seed: int = -1              # -1 = 随机
    image_format: str = "png"   # png | webp

    # ── 高级参数（NAI 4/4.5）──
    variety_boost: bool = False
    cfg_rescale: Optional[float] = None
    noise_schedule: Optional[str] = None    # karras | exponential | polyexponential
    quality_toggle: bool = False            # NAI 质量开关（旧字段名 qualityToggle）
    auto_smea: bool = False                # NAI SMEA 开关（旧字段名 autoSmea）

    # ── 多角色 ──
    characters: Optional[list[dict]] = None
    use_coords: bool = False

    # ── 参考图模式 ──
    ref_mode: str = "none"     # none | i2i | char_ref | vibe | inpaint

    # i2i（图生图）
    i2i_image: Optional[str] = None        # data URI 或纯 base64
    i2i_strength: float = 0.7
    i2i_noise: float = 0.0

    # char-ref（角色参考，V4.5 限定）
    char_ref_image: Optional[str] = None   # data URI 或纯 base64
    char_ref_type: str = "character"       # character | style | character&style
    char_ref_fidelity: float = 1.0
    char_ref_strength: float = 1.0

    # vibe / controlnet（风格迁移）
    vibe_images: Optional[list[dict]] = None
    vibe_global_strength: float = 1.0

    # inpaint（局部重绘，预留）
    inpaint_image: Optional[str] = None
    inpaint_mask: Optional[str] = None
    inpaint_strength: float = 1.0

    # ── 请求控制 ──
    max_tokens: int = 100000   # OpenAI max_tokens = Anlas × 10000
    timeout: int = 180
    retry_attempts: int = 3
    proxy_mode: str = "auto"   # auto | inherit | direct

    # ── 扩展参数（legacy 端点兼容）──
    extra_params: Optional[dict] = None


@dataclass
class GenerationResult:
    """ImageClient.generate() 的返回值。"""

    success: bool
    image_base64: str = ""     # 纯 base64（不含 data: 前缀）
    seed: int = -1
    error: str = ""
    vibe_cache_ids: list[dict] = field(default_factory=list)
    raw_content: str = ""      # 原始 message.content（调试用）


class ImageClient(ABC):
    """所有图片 API 客户端的抽象基类。

    子类实现 generate()，从 GenerationContext 取需要的字段构造 API payload。
    """

    @abstractmethod
    async def generate(self, ctx: GenerationContext) -> GenerationResult:
        """执行图片生成请求，返回 GenerationResult。

        实现要点：
        - 互斥校验（i2i vs inpaint，controlnet vs character_references）
        - 构造 API payload
        - 发请求 + 重试
        - 解析响应，提取 image_base64 + seed + vibe_cache_ids
        """
        ...


# ── 互斥校验工具 ──

def validate_ref_mutex(ctx: GenerationContext) -> None:
    """校验参考图模式的互斥规则。

    Raises:
        ValueError: 如果违反互斥规则。
    """
    # 互斥组 A: i2i vs inpaint
    has_i2i = ctx.ref_mode == "i2i" and ctx.i2i_image
    has_inpaint = ctx.ref_mode == "inpaint" and ctx.inpaint_image
    if has_i2i and has_inpaint:
        raise ValueError("i2i and inpaint are mutually exclusive")

    # 互斥组 B: controlnet (vibe) vs character_references (char_ref)
    has_vibe = ctx.ref_mode == "vibe" and ctx.vibe_images
    has_char_ref = ctx.ref_mode == "char_ref" and ctx.char_ref_image
    if has_vibe and has_char_ref:
        raise ValueError("controlnet and character_references are mutually exclusive")


def calc_max_tokens(ctx: GenerationContext) -> int:
    """根据参考图模式计算 max_tokens 预算。

    Returns:
        max_tokens 值（1 anlas = 10000 tokens）。
    """
    if ctx.ref_mode == "char_ref":
        return 60000   # 5 anlas + 余量
    if ctx.ref_mode == "vibe":
        return 40000   # 3 anlas + 附加费 + 余量
    # txt2img / i2i / inpaint = 0 anlas（但 max_tokens 仍需要合理值）
    return ctx.max_tokens