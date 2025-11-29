# MaiBot_LLM2pic - 智能图片生成插件

使用 LLM 根据聊天记录和人设生成符合需求的 prompt，然后调用图片生成 API。

## 目录

- [功能特性](#功能特性)
- [支持的 API 类型](#支持的-api-类型)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [测试](#测试)
- [工作原理](#工作原理)
- [常见问题](#常见问题)
- [进阶配置](#进阶配置)
- [更新日志](#更新日志)
- [技术文档](#技术文档)

## 功能特性

- 🤖 **智能提示词生成**：使用 LLM 根据聊天上下文自动生成高质量的图片提示词
- 🎨 **多 API 支持**：支持 OpenAI 格式和 Gradio 格式的图片生成 API
- 🖼️ **自拍模式**：可以根据角色人设生成自拍照片
- ✂️ **图片裁切**：支持自动裁切图片边缘（用于去除水印）
- 🔧 **高度可配置**：丰富的配置选项，满足不同需求

## 支持的 API 类型

### 1. OpenAI 格式（api_type = "openai"）

支持所有兼容 OpenAI `/v1/chat/completions` 格式的图片生成 API，包括：
- **OpenAI DALL-E**：官方图片生成服务
- **Grok Image Generation**：X.AI 的图片生成服务
- 其他兼容 OpenAI 格式的服务

### 2. Gradio 格式（api_type = "gradio"）

支持 Gradio 应用的图片生成 API，特别是 HuggingFace Space 上的模型，例如：
- **Z-Image-Turbo**：`https://tongyi-mai-z-image-turbo.hf.space`（通义万相，免费无需密钥）
- 其他基于 Gradio 的图片生成应用

## 快速开始

### 方式一：使用 Z-Image-Turbo（推荐，免费）

Z-Image-Turbo 是通义万相推出的免费图片生成模型，托管在 HuggingFace Space 上，无需 API 密钥即可使用。

#### 1. 创建配置文件

```bash
cd MaiBot/plugins/MaiBot_LLM2pic
cp config.example.toml config.toml
```

#### 2. 编辑配置

打开 `config.toml`，确保以下配置：

```toml
[plugin]
enabled = true

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

#### 3. 测试 API

```bash
source ../../venv/bin/activate
python tests/test_gradio_api.py
```

如果看到 "✓ 测试成功！"，说明配置正确。测试图片会保存在 `md_pic/` 目录。

#### 4. 启动 MaiBot

重启 MaiBot，插件会自动加载。

### 方式二：使用 OpenAI DALL-E

```toml
[api]
api_type = "openai"
base_url = "https://api.openai.com/v1"
api_key = "sk-your-api-key-here"

[generation]
default_model = "dall-e-3"
default_size = "1024x1024"
```

### 方式三：使用 Grok Image

```toml
[api]
api_type = "openai"
base_url = "https://api.x.ai/v1"
api_key = "xai-your-api-key-here"

[generation]
default_model = "grok-2-image"
```

## 配置说明

### 完整配置示例

```toml
[plugin]
enabled = true

[api]
# API类型：openai 或 gradio
api_type = "gradio"
# API基础URL
base_url = "https://tongyi-mai-z-image-turbo.hf.space"
# API密钥（Gradio可留空）
api_key = ""

[generation]
# OpenAI格式专用
default_model = "gpt-image-1"
default_size = ""

# 全局附加提示词
custom_prompt_add = ""

# 图片裁切（去水印）
crop_enabled = false
crop_position = "bottom"  # top/bottom/left/right
crop_pixels = 40

# Gradio格式专用
gradio_resolution = "1024x1024 ( 1:1 )"
gradio_steps = 8
gradio_shift = 3
gradio_timeout = 120

[llm]
# 用于生成提示词的LLM模型
model_name = ""
# 自定义系统提示词
system_prompt = ""

[components]
enable_image_generation = true
```

### 配置项详解

#### API 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `api_type` | string | `"openai"` | API 类型：`openai` 或 `gradio` |
| `base_url` | string | - | API 基础 URL |
| `api_key` | string | - | API 密钥（Gradio 可留空） |

#### 图片生成参数

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `default_model` | string | `"gpt-image-1"` | OpenAI 格式的模型名称 |
| `default_size` | string | `""` | OpenAI 格式的图片尺寸 |
| `custom_prompt_add` | string | `""` | 全局附加提示词 |
| `crop_enabled` | bool | `false` | 是否启用图片裁切 |
| `crop_position` | string | `"bottom"` | 裁切位置：top/bottom/left/right |
| `crop_pixels` | int | `40` | 裁切像素数 |

#### Gradio 专用参数

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `gradio_resolution` | string | `"1024x1024 ( 1:1 )"` | 图片分辨率 |
| `gradio_steps` | int | `8` | 推理步数（4-50，越大质量越好但越慢） |
| `gradio_shift` | int | `3` | 时间偏移参数 |
| `gradio_timeout` | int | `120` | 轮询超时时间（秒） |

**可用的分辨率选项：**
- `"512x512 ( 1:1 )"` - 小图，快速
- `"1024x1024 ( 1:1 )"` - 方形，推荐
- `"1024x1536 ( 2:3 )"` - 竖图
- `"1536x1024 ( 3:2 )"` - 横图

**推理步数建议：**
- `4` - 快速模式（低质量）
- `8` - 平衡模式（推荐）
- `20` - 高质量模式（慢）

#### LLM 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model_name` | string | `""` | 用于生成提示词的 LLM 模型（留空使用默认） |
| `system_prompt` | string | `""` | 自定义系统提示词（留空使用默认） |

## 使用示例

### 普通绘图

在聊天中发送以下消息触发图片生成：

- "画一张猫咪的图片"
- "帮我画个日落"
- "我想看看樱花的样子"
- "画一个女孩在雨中"

### 自拍模式

当用户要求自拍时，插件会以角色身份生成自拍照：

- "自拍"
- "来张自拍"
- "发张照片看看"
- "你现在在哪，发张图看看"

### 触发条件

插件会在以下情况下自动触发：

1. 用户想看你当前的状态/环境/正在做的事
2. 用户想看你拍的照片/摄影作品
3. 用户想看你正在吃/喝/用的东西
4. 用户想看你画的画/创作的图
5. 用户想看某个具体场景/角色/事物的图片

## 测试

### 测试 Gradio API

```bash
cd MaiBot/plugins/MaiBot_LLM2pic
source ../../venv/bin/activate
python tests/test_gradio_api.py
```

### 集成测试

```bash
python tests/test_integration.py
```

测试图片将保存到 `md_pic/` 文件夹中。

## 工作原理

### 完整流程

1. **触发检测**
   - LLM 判断用户消息是否需要生成图片
   - 识别是否为自拍模式

2. **提示词生成**
   - 获取最近的聊天记录（30分钟内，最多20条）
   - 获取角色人设信息
   - 使用 LLM 根据上下文生成英文提示词
   - 清理和优化提示词

3. **API 调用**
   - **OpenAI 格式**：
     - 直接调用 `/chat/completions` 端点
     - 从响应中提取图片 URL 或 Base64
   - **Gradio 格式**：
     - POST 请求获取 event_id
     - GET 请求轮询结果（SSE 格式）
     - 解析响应获取图片 URL

4. **图片处理**
   - 下载图片（如果是 URL）
   - 可选：裁切图片边缘（去水印）
   - 编码为 Base64

5. **发送图片**
   - 将图片发送到聊天

### 技术架构

```
用户消息
    ↓
LLM 判定是否需要生成图片
    ↓
获取聊天记录 + 人设
    ↓
LLM 生成英文提示词
    ↓
    ├─→ OpenAI API → 图片URL/Base64
    └─→ Gradio API → POST(event_id) → GET(轮询) → 图片URL
    ↓
下载 + 裁切（可选）
    ↓
发送图片
```

## 常见问题

### Q: 生成图片很慢？

**A:** HuggingFace Space 是免费服务，可能需要排队。可以尝试：
1. 减少 `gradio_steps`（如设为 4）
2. 增加 `gradio_timeout`（如设为 180）
3. 选择非高峰时段使用

### Q: 提示 "轮询超时"？

**A:** 
1. 增加 `gradio_timeout` 的值（如 180 或 240）
2. 检查网络连接
3. 稍后再试（可能服务器繁忙）

### Q: 想使用其他 API？

**A:** 修改配置：
```toml
[api]
api_type = "openai"  # 改为 openai
base_url = "你的API地址"
api_key = "你的API密钥"
```

### Q: 如何提高图片质量？

**A:** 
1. 增加 `gradio_steps`（如设为 20）
2. 使用更高的分辨率
3. 在 `custom_prompt_add` 中添加质量词：
   ```toml
   custom_prompt_add = "masterpiece, best quality, highly detailed"
   ```

### Q: 如何去除图片水印？

**A:** 启用图片裁切：
```toml
[generation]
crop_enabled = true
crop_position = "bottom"  # 根据水印位置调整
crop_pixels = 40          # 根据水印大小调整
```

### Q: 生成的提示词不够好？

**A:** 
1. 使用更强的 LLM 模型：
   ```toml
   [llm]
   model_name = "gpt-4"
   ```
2. 自定义系统提示词：
   ```toml
   [llm]
   system_prompt = "你的自定义提示词..."
   ```

### Q: 需要安装额外依赖吗？

**A:** 
- 基础功能：无需额外依赖
- 图片裁切：需要安装 PIL/Pillow
  ```bash
  pip install Pillow
  ```

## 进阶配置

### 自定义提示词前缀

在所有生成的提示词前添加固定内容：

```toml
[generation]
custom_prompt_add = "masterpiece, best quality, highly detailed, 8k resolution"
```

### 自定义 LLM 系统提示词

完全自定义提示词生成逻辑：

```toml
[llm]
system_prompt = """你是一位专业的AI绘画提示词生成专家。

## 你的角色设定
{persona}

## 输出规则
1. 只输出纯英文提示词
2. 使用逗号分隔的关键词格式
3. 关键词顺序：主体 -> 特征 -> 动作 -> 背景 -> 风格

请根据用户请求生成提示词。"""
```

注意：`{persona}` 占位符会被自动替换为角色人设。

### 多分辨率配置

根据不同场景使用不同分辨率：

```toml
# 人物肖像
gradio_resolution = "1024x1536 ( 2:3 )"

# 风景照片
gradio_resolution = "1536x1024 ( 3:2 )"

# 通用场景
gradio_resolution = "1024x1024 ( 1:1 )"
```

### 性能优化

快速生成模式（牺牲质量换取速度）：

```toml
[generation]
gradio_steps = 4
gradio_timeout = 60
```

高质量模式（牺牲速度换取质量）：

```toml
[generation]
gradio_steps = 20
gradio_timeout = 180
```

## 更新日志

### [3.0.0] - 2025-11-29

#### 新增功能
- ✨ **Gradio API 支持**：新增对 Gradio 格式 API 的支持，可以调用 HuggingFace Space 上的图片生成模型
- 🎨 **Z-Image-Turbo 集成**：完整支持通义万相 Z-Image-Turbo 模型
- 🔧 **API 类型配置**：新增 `api_type` 配置项，支持 `openai` 和 `gradio` 两种格式
- 📊 **Gradio 专用参数**：新增 `gradio_resolution`、`gradio_steps`、`gradio_shift`、`gradio_timeout` 配置项

#### 技术改进
- 🔄 **双 API 架构**：实现了 OpenAI 格式和 Gradio 格式的双 API 支持
- 🔍 **SSE 解析**：实现了 Gradio Server-Sent Events (SSE) 响应的解析
- ⏱️ **轮询机制**：实现了 Gradio API 的 POST + GET 轮询机制
- 🧪 **测试脚本**：添加了完整的测试脚本（`test_gradio_api.py` 和 `test_integration.py`）

#### 文档更新
- 📝 添加了完整的 `README.md` 文档
- 📋 添加了 `config.example.toml` 配置示例
- 📖 添加了 API 调用文档（`Zimagedoc-curl.md` 和 `zimagedoc-mcp.md`）
- 📚 添加了 `QUICKSTART.md` 快速开始指南
- 📜 添加了 `CHANGELOG.md` 更新日志

#### 向后兼容
- ✅ 完全兼容现有的 OpenAI 格式配置
- ✅ 默认使用 OpenAI 格式，不影响现有用户

#### 测试结果
- ✅ Gradio API 调用测试通过
- ✅ 图片下载和保存测试通过
- ✅ 集成测试通过
- ✅ 代码语法检查通过

### [2.x] - 之前版本

#### 功能
- LLM 智能提示词生成
- OpenAI 格式 API 支持
- 自拍模式
- 图片裁切功能
- 人设集成

## 技术文档

### Gradio API 调用流程

#### 1. POST 请求获取 event_id

```bash
curl -X POST https://tongyi-mai-z-image-turbo.hf.space/gradio_api/call/generate \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      "a cute cat",           # prompt
      "1024x1024 ( 1:1 )",   # resolution
      42,                     # seed
      8,                      # steps
      3,                      # shift
      true,                   # random_seed
      []                      # gallery_images
    ]
  }'
```

响应：
```json
{"event_id": "ae677b5f085a43e5bcce120534a6ac40"}
```

#### 2. GET 请求轮询结果

```bash
curl -N https://tongyi-mai-z-image-turbo.hf.space/gradio_api/call/generate/{event_id}
```

响应（SSE 格式）：
```
event: complete
data: [[{"image": {"url": "https://...", ...}}, ...], "305049", 305049]
```

### OpenAI API 调用流程

```bash
curl -X POST https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "dall-e-3",
    "messages": [{"role": "user", "content": "a cute cat"}]
  }'
```

### 代码结构

```
MaiBot_LLM2pic/
├── plugin.py              # 主插件代码
├── config.example.toml    # 配置示例
├── config.toml            # 配置文件（需自行创建，已在.gitignore中）
├── README.md              # 本文档
├── LICENSE                # 许可证
├── docs/                  # 文档目录
│   ├── CHANGELOG.md       # 更新日志
│   ├── QUICKSTART.md      # 快速开始
│   ├── Zimagedoc-curl.md  # curl 调用文档
│   └── zimagedoc-mcp.md   # MCP 调用文档
└── tests/                 # 测试目录（已在.gitignore中）
    ├── test_gradio_api.py # Gradio API 测试
    └── test_integration.py# 集成测试
```

### 核心类和方法

- `PromptGenerator`: 提示词生成器
  - `generate_prompt()`: 使用 LLM 生成提示词

- `CustomPicAction`: 图片生成动作
  - `execute()`: 执行图片生成
  - `_make_gradio_image_request()`: Gradio API 调用
  - `_make_http_image_request()`: OpenAI API 调用
  - `_handle_image_result()`: 处理图片结果

- `CustomPicPlugin`: 插件主类
  - `get_plugin_components()`: 返回插件组件

## 依赖

- Python 3.11+
- MaiBot 插件系统
- PIL/Pillow（可选，用于图片裁切）

## 许可证

与 MaiBot 主项目相同

## 作者

CharTyr

## 贡献

欢迎提交 Issue 和 Pull Request！

## 支持

如有问题，请在 MaiBot 项目中提交 Issue。
