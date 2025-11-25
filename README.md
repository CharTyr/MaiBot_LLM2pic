# MaiBot_LLM2pic

MaiBot 图片生成插件 - 使用 LLM 根据聊天记录和人设智能生成提示词，然后调用图片生成 API。

## 功能特点

- 🤖 **智能提示词生成**：使用 LLM（默认 planner 模型）根据聊天上下文和角色人设自动生成图片提示词
- 🎨 **自然语言生图**：适配 OpenAI gpt-image、Grok grok-imagine 等自然语言生图模型(Openai兼容格式)
- 📸 **多场景支持**：自拍、摄影作品、画图等多种触发场景
- ✂️ **图片裁切**：可选裁切图片边缘，去除 AI 生成的水印
- ⚙️ **高度可配置**：支持自定义 LLM 提示词、模型选择等

## 安装

1. 将插件文件夹放入 MaiBot 的 `plugins` 目录
2. 复制 `config.example.toml` 为 `config.toml`
3. 编辑 `config.toml`，填入你的 API 配置

## 配置说明

### API 配置 `[api]`

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `base_url` | 图片生成 API 地址 | `https://api.openai.com/v1` |
| `api_key` | API 密钥 | - |

### 图片生成配置 `[generation]`

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `default_model` | 生图模型名称 | `gpt-image-1` |
| `default_size` | 图片尺寸（留空自动） | `""` |
| `crop_enabled` | 是否启用裁切 | `false` |
| `crop_position` | 裁切位置 (top/bottom/left/right) | `bottom` |
| `crop_pixels` | 裁切像素数 | `40` |
| `custom_prompt_add` | 全局附加提示词 | `""` |

### LLM 配置 `[llm]`

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `model_name` | LLM 模型名称（留空使用 planner） | `""` |
| `system_prompt` | 自定义系统提示词，支持 `{persona}` 占位符 | `""` |

## 触发条件

插件通过 LLM 判断是否触发，典型场景包括：

- 要求自拍/发照片
- 想看角色当前状态/环境
- 想看角色拍的摄影作品
- 想看角色在吃/喝什么
- 要求画图/生成图片

## 依赖

- MaiBot 主程序
- Pillow（用于图片裁切，MaiBot 已包含）

## 许可证

MIT License
