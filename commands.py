"""/pic 指令元数据（纯数据，不继承出图客户端）。"""


class DirectPicCommand:
    """直接生成图片的指令元数据。"""

    command_name = "direct_pic"
    command_description = (
        "使用自然语言描述生成图片，会先转写为 Danbooru tags。"
        "可选前缀（顺序任意）：nsfw、i2i/char-ref/vibe、anime/edit。"
        "例: /pic i2i 照这个姿势画；/pic i2i nsfw ... 与 /pic nsfw i2i ... 均可；"
        "支持回复引用图片消息（正文含 /pic 即可）"
    )
    # 只抓 /pic 后整段 body；nsfw/ref/style 在 plugin 里任意序解析。
    # 勿用「多个具名分组 + 量词交替」：Python 会用最后一次交替覆盖，冲掉先前 nsfw。
    command_pattern = r"/pic\s+(?P<body>.+)$"
