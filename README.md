# MaiBot_LLM2pic - 智能图片生成插件

使用 LLM 根据聊天上下文自动生成提示词，调用图片生成 API。支持文生图和图生图。

## 依赖

- Python 3.11+
- Pillow（可选，用于图片裁切）：`pip install Pillow`

## 快速开始

```bash
cp config.example.toml config.toml
```

编辑 `config.toml`，根据你的 API 类型配置。

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

### 2. OpenAI 格式（支持图生图）

```toml
[anime]
enabled = true
api_type = "openai"
base_url = "https://api.openai.com/v1"
api_key = "sk-xxx"
model_name = "dall-e-3"
```

OpenAI 格式使用 chat/completions 端点，支持多模态输入，可以进行图生图。

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

## 双模型配置

插件支持 `[anime]` 和 `[real]` 两个独立模型，LLM 会自动判断使用哪个：

- `anime`：二次元/动漫风格（自拍模式强制使用）
- `real`：写实/真实风格

## 使用方式

### 自动触发（文生图）

聊天中提到画图相关内容会自动触发：
- "画一张猫咪"
- "来张自拍"

### 自动触发（图生图）

发送图片并要求修改时会自动触发：
- [图片] "把这张图变成动漫风格"
- [图片] "帮我修改一下这张图"

### /pic 命令

```
/pic <prompt>           # 文生图，使用默认风格
/pic anime <prompt>     # 文生图，强制二次元
/pic real <prompt>      # 文生图，强制写实
[图片] /pic <prompt>    # 图生图，基于发送的图片生成
```

## 图生图说明

图生图功能仅支持 OpenAI 格式的 API（api_type = "openai"）。

使用方式：
1. 发送一张图片
2. 在同一条消息中使用 `/pic <描述>` 命令
3. 或者让 LLM 自动判断（发送图片并说"把这张图..."）

图生图会将图片和文字描述一起发送给 API，API 会根据描述对图片进行修改或重绘。

## 常用配置

### 附加提示词

```toml
[generation]
custom_prompt_add = "masterpiece, best quality"
```

代码会自动在后面加逗号，不需要手动加。

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

**Q: 图生图不工作？**
- 确保使用的是 OpenAI 格式的 API（api_type = "openai"）
- 确保 API 支持多模态输入（如 GPT-4V、Gemini 等）
- Gradio 和 SD API 目前不支持图生图

## 许可证

与 MaiBot 主项目相同
