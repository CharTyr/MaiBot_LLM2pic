# 快速开始指南

## 使用 Z-Image-Turbo（推荐）

Z-Image-Turbo 是通义万相推出的免费图片生成模型，托管在 HuggingFace Space 上，无需 API 密钥即可使用。

### 1. 创建配置文件

```bash
cd MaiBot/plugins/MaiBot_LLM2pic
cp config.example.toml config.toml
```

### 2. 编辑配置

打开 `config.toml`，确保以下配置：

```toml
[api]
api_type = "gradio"
base_url = "https://tongyi-mai-z-image-turbo.hf.space"
api_key = ""  # 留空即可

[generation]
gradio_resolution = "1024x1024 ( 1:1 )"
gradio_steps = 8
gradio_shift = 3
gradio_timeout = 120
```

### 3. 测试 API

```bash
source ../../venv/bin/activate
python test_gradio_api.py
```

如果看到 "✓ 测试成功！"，说明配置正确。

### 4. 启动 MaiBot

重启 MaiBot，插件会自动加载。

### 5. 使用插件

在聊天中发送：
- "画一只猫"
- "自拍"
- "我想看看樱花的样子"

## 可用的分辨率选项

```toml
gradio_resolution = "512x512 ( 1:1 )"      # 小图，快速
gradio_resolution = "1024x1024 ( 1:1 )"    # 方形，推荐
gradio_resolution = "1024x1536 ( 2:3 )"    # 竖图
gradio_resolution = "1536x1024 ( 3:2 )"    # 横图
```

## 调整生成质量

```toml
# 快速模式（低质量）
gradio_steps = 4

# 平衡模式（推荐）
gradio_steps = 8

# 高质量模式（慢）
gradio_steps = 20
```

## 常见问题

### Q: 生成图片很慢？
A: HuggingFace Space 是免费服务，可能需要排队。可以尝试：
1. 减少 `gradio_steps`（如设为 4）
2. 增加 `gradio_timeout`（如设为 180）

### Q: 提示 "轮询超时"？
A: 增加 `gradio_timeout` 的值，或者稍后再试。

### Q: 想使用其他 API？
A: 修改 `api_type = "openai"`，并配置相应的 `base_url` 和 `api_key`。

## 进阶配置

### 自定义提示词前缀

```toml
[generation]
custom_prompt_add = "masterpiece, best quality, highly detailed"
```

### 启用图片裁切（去水印）

```toml
[generation]
crop_enabled = true
crop_position = "bottom"  # 裁切底部
crop_pixels = 40          # 裁切 40 像素
```

### 使用特定 LLM 模型生成提示词

```toml
[llm]
model_name = "gpt-4"  # 使用 GPT-4 生成提示词
```

## 测试图片

测试生成的图片会保存在项目根目录的 `md_pic/` 文件夹中。
