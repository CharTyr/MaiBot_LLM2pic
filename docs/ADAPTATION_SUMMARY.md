# Z-Image-Turbo API 适配总结

## 适配目标

为 MaiBot_LLM2pic 插件添加对 HuggingFace Space 上 Gradio 格式 API 的支持，特别是通义万相的 Z-Image-Turbo 模型。

## 完成的工作

### 1. 代码修改

#### 1.1 配置架构扩展 (`plugin.py`)

**新增配置项：**
- `api.api_type`: API 类型选择（`openai` 或 `gradio`）
- `generation.gradio_resolution`: Gradio API 图片分辨率
- `generation.gradio_steps`: 推理步数
- `generation.gradio_shift`: 时间偏移参数
- `generation.gradio_timeout`: 轮询超时时间

**修改位置：** `CustomPicPlugin.config_schema`

#### 1.2 新增 Gradio API 调用方法

**新方法：** `CustomPicAction._make_gradio_image_request()`

**实现逻辑：**
1. POST 请求到 `/gradio_api/call/generate` 获取 `event_id`
2. GET 请求到 `/gradio_api/call/generate/{event_id}` 轮询结果
3. 解析 SSE (Server-Sent Events) 格式响应
4. 提取图片 URL 并返回

**关键技术点：**
- SSE 响应解析（`event: complete` 和 `data:` 行）
- JSON 数据提取（从嵌套结构中获取图片 URL）
- 轮询机制（带超时控制）

#### 1.3 API 调用路由

**修改方法：** `CustomPicAction.execute()`

**改动：**
```python
# 根据 api_type 选择不同的调用方法
if api_type.lower() == "gradio":
    success, result = await asyncio.to_thread(
        self._make_gradio_image_request,
        prompt=final_prompt,
    )
else:
    success, result = await asyncio.to_thread(
        self._make_http_image_request,
        prompt=final_prompt,
        model=default_model,
        size=image_size if image_size else None,
    )
```

### 2. 测试脚本

#### 2.1 基础 API 测试 (`test_gradio_api.py`)

**功能：**
- 测试 Gradio API 的完整调用流程
- 下载并保存测试图片到 `md_pic/`
- 验证 API 响应格式

**测试结果：** ✅ 通过

#### 2.2 集成测试 (`test_integration.py`)

**功能：**
- 测试插件中的 `_make_gradio_image_request` 方法
- 模拟真实的插件环境
- 验证配置读取和方法调用

**测试结果：** ✅ 通过

### 3. 文档

#### 3.1 README.md
- 完整的功能介绍
- 支持的 API 类型说明
- 配置示例
- 使用指南

#### 3.2 QUICKSTART.md
- Z-Image-Turbo 快速开始指南
- 常见问题解答
- 进阶配置示例

#### 3.3 config.example.toml
- 完整的配置示例
- 详细的配置项说明
- Gradio 和 OpenAI 两种格式的示例

#### 3.4 CHANGELOG.md
- 版本更新记录
- 新功能列表
- 技术改进说明

#### 3.5 API 文档
- `Zimagedoc-curl.md`: cURL 调用文档
- `zimagedoc-mcp.md`: MCP 调用文档

## 技术亮点

### 1. 双 API 架构设计

通过 `api_type` 配置实现了两种 API 格式的无缝切换：
- **OpenAI 格式**：传统的 `/chat/completions` 端点
- **Gradio 格式**：Gradio 应用的 `/gradio_api/call/*` 端点

### 2. SSE 响应解析

实现了对 Gradio Server-Sent Events 格式的解析：
```
event: complete
data: [[{"image": {"url": "..."}}], "seed", seed_int]
```

### 3. 轮询机制

实现了带超时控制的轮询机制，确保在 API 响应慢时不会无限等待。

### 4. 向后兼容

所有改动都保持了向后兼容：
- 默认 `api_type = "openai"`
- 现有配置无需修改即可继续使用
- 新增配置项都有合理的默认值

## 测试验证

### 测试环境
- Python 3.11.2
- MaiBot 虚拟环境
- Linux 系统

### 测试结果

| 测试项 | 状态 | 说明 |
|--------|------|------|
| Gradio API 调用 | ✅ | 成功生成图片 |
| 图片下载 | ✅ | 成功下载并保存 |
| 集成测试 | ✅ | 插件方法正常工作 |
| 代码语法检查 | ✅ | 无语法错误 |

### 生成的测试图片

保存在 `md_pic/` 目录：
1. `z-image-test-cat.png` (1.4MB) - 初始 API 测试
2. `gradio-test-sunset.png` (1.1MB) - 完整流程测试
3. `integration-test-robot.png` (972KB) - 集成测试

## 使用示例

### 配置 Z-Image-Turbo

```toml
[api]
api_type = "gradio"
base_url = "https://tongyi-mai-z-image-turbo.hf.space"
api_key = ""

[generation]
gradio_resolution = "1024x1024 ( 1:1 )"
gradio_steps = 8
gradio_shift = 3
```

### 运行测试

```bash
cd MaiBot/plugins/MaiBot_LLM2pic
source ../../venv/bin/activate
python test_gradio_api.py
```

## 文件清单

### 修改的文件
- `plugin.py` - 核心插件代码

### 新增的文件
- `test_gradio_api.py` - API 测试脚本
- `test_integration.py` - 集成测试脚本
- `README.md` - 完整文档
- `QUICKSTART.md` - 快速开始指南
- `config.example.toml` - 配置示例
- `CHANGELOG.md` - 更新日志
- `ADAPTATION_SUMMARY.md` - 本文档

### 文档文件（参考）
- `Zimagedoc-curl.md` - cURL API 文档
- `zimagedoc-mcp.md` - MCP API 文档

## 后续建议

### 1. 性能优化
- 考虑添加图片缓存机制
- 实现并发请求限制

### 2. 功能扩展
- 支持更多 Gradio 模型
- 添加图片风格预设
- 支持批量生成

### 3. 用户体验
- 添加生成进度提示
- 支持生成失败重试
- 添加图片质量评分

## 总结

本次适配成功为 MaiBot_LLM2pic 插件添加了 Gradio API 支持，使其能够调用 HuggingFace Space 上的免费图片生成模型。适配过程中保持了良好的代码结构和向后兼容性，并提供了完整的测试和文档。

**版本：** v3.0.0  
**适配日期：** 2024-11-29  
**适配者：** Kiro AI Assistant
