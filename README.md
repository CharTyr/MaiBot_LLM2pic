# MaiBot_LLM2pic - 智能图片生成插件

使用 LLM 根据聊天上下文自动生成提示词，调用图片生成 API。支持文生图和图片编辑。

## 依赖

- Python 3.11+
- Pillow（可选，用于图片裁切和缩放）：`pip install Pillow`

## 快速开始

```bash
cp config.example.toml config.toml
```

编辑 `config.toml`，根据你的 API 类型配置。

## 双工具架构

插件提供两个独立的工具，LLM 会根据场景自动选择：

| 工具 | 用途 | 模型配置 |
|------|------|----------|
| `draw_picture` | 纯文生图（二次元/动漫风格） | `[anime]` |
| `edit_picture` | 图片编辑 + GPT 文生图 | `[edit]` |

### draw_picture

从零生成图片，走 anime 模型（NovelAI / Gradio / SD API 等）。

触发场景：
- "画一张猫咪"
- "来张自拍"
- "我想看看你在干嘛"

### edit_picture

编辑用户提供的图片，或使用 GPT 图像模型生成图片。走 edit 模型（OpenAI chat/completions 格式）。

触发场景：
- [图片] "把这张图变成动漫风格"
- [引用图片] "给这张图加上圣诞帽"
- "用GPT画一张写实的风景"

图片获取方式：
1. 用户消息中直接包含图片
2. 用户引用/回复了一条包含图片的消息

如果没有找到图片，会退化为纯文生图模式。

## 支持的 API 类型

### 1. Gradio（推荐，免费）

```toml
[anime]
enabled = true
api_type = "gradio"
base_url = "https://tongyi-mai-z-image-turbo.hf.space"
api_key = ""  # 留空

gradio_resolution = "1024x1024 ( 1:1 )"
gradio_steps = 8
gradio_shift = 3
gradio_timeout = 120
```

### 2. OpenAI 格式（用于 edit 模型）

```toml
[edit]
enabled = true
api_type = "openai"
base_url = "https://api.openai.com/v1"
api_key = "sk-xxx"
model_name = "gpt-image-2"
```

OpenAI 格式使用 chat/completions 端点，支持多模态输入（图片+文字）。

### 3. SD API

```toml
[anime]
enabled = true
api_type = "sd_api"
base_url = "https://sd.exacg.cc"  # 不带 /api/v1/generate_image
api_key = "你的密钥"

sd_negative_prompt = ""
sd_width = 512
sd_height = 512
sd_steps = 20
sd_cfg = 7.0
sd_model_index = 0
sd_seed = -1
```

### 4. NovelAI

```toml
[anime]
enabled = true
api_type = "novelai"
api_key = "你的Bearer Token"

novelai_model = "nai-diffusion-4-5-full"
novelai_width = 832
novelai_height = 1216
novelai_steps = 28
novelai_scale = 5.0
```

## /pic 命令

```
/pic <prompt>           # 文生图，使用默认风格
/pic anime <prompt>     # 文生图，强制 anime 模型
/pic edit <prompt>      # 使用 edit 模型生成（支持图生图）
```

## 常用配置

### 附加提示词

```toml
[generation]
custom_prompt_add = "masterpiece, best quality"
```

### 图片裁切（去水印）

```toml
[generation]
crop_enabled = true
crop_position = "bottom"  # top/bottom/left/right
crop_pixels = 40
```

### LLM 配置

```toml
[llm]
model_name = ""  # 留空使用系统默认
context_message_limit = 20
context_time_minutes = 30
system_prompt = ""  # 留空使用默认，支持 {persona} 占位符
```

## 常见问题

**Q: 生成很慢/超时？**
- 增加 `gradio_timeout`
- 减少 `gradio_steps`

**Q: 想提高质量？**
- 增加 `gradio_steps`（如 20）
- 在 `custom_prompt_add` 加质量词

**Q: 图片编辑不工作？**
- 确保 `[edit]` 配置了支持多模态输入的 OpenAI 格式 API
- 确保最近5分钟内有图片消息（直接发送或引用）
- 检查 Pillow 是否安装（用于图片缩放预处理）

## 许可证

与 MaiBot 主项目相同
