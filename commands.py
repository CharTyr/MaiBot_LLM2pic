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


class ReverseTagCommand:
    """纯 WD14 反推：只回 tag，不出图。"""

    command_name = "reverse_tag"
    command_description = (
        "WD14 反推图片 Danbooru tags，只回文本不出图。"
        "用法：回复一张图片发 /tags；或本条附图 + /tags。"
        "可选 /tags detail 放宽置信度阈值（仍不显示 conf）。"
    )
    # 勿加 ^：回复引用时正文前会有 [回复了…] 前缀（同 /pic 坑）
    command_pattern = r"/tags(?:\s+(?P<body>\S.*))?$"

