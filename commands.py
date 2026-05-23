"""/pic 指令元数据。"""

from .image_clients import ImageClientMixin


class DirectPicCommand(ImageClientMixin):
    """直接生成图片的指令元数据和共享图片能力。"""

    command_name = "direct_pic"
    command_description = "直接使用提供的prompt生成图片，不经过LLM处理。支持 /pic anime <prompt> 或 /pic edit <prompt> 指定风格"
    command_pattern = r"^/pic\s+(?:(?P<style>anime|edit)\s+)?(?P<prompt>.+)$"
