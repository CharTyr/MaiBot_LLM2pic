"""/pic 指令元数据。"""

from .image_clients import ImageClientMixin


class DirectPicCommand(ImageClientMixin):
    """直接生成图片的指令元数据和共享图片能力。"""

    command_name = "direct_pic"
    command_description = "使用自然语言描述生成图片，会先转写为 Danbooru tags。可选前缀：nsfw（NSFW模式）、i2i/char-ref/vibe（参考图模式，需附图）、anime/edit（风格）。例: /pic i2i 照这个姿势画；支持回复引用图片消息（正文含 /pic 即可）"
    command_pattern = r"/pic\s+(?:(?P<nsfw>[Nn][Ss][Ff][Ww])\s+)?(?:(?P<ref>i2i|char-ref|char_ref|vibe)\s+)?(?:(?P<style>anime|edit)\s+)?(?P<prompt>.+)$"
