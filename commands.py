"""/pic 指令元数据。"""

from .image_clients import ImageClientMixin


class DirectPicCommand(ImageClientMixin):
    """直接生成图片的指令元数据和共享图片能力。"""

    command_name = "direct_pic"
    command_description = "使用自然语言描述生成图片，会先转写为 Danbooru tags。支持 /pic nsfw <描述> 或 /pic anime <描述> 指定模式/风格"
    command_pattern = r"^/pic\s+(?:(?P<nsfw>[Nn][Ss][Ff][Ww])\s+)?(?:(?P<style>anime|edit)\s+)?(?P<prompt>.+)$"
