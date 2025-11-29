# 更新日志

## [3.0.0] - 2024-11-29

### 新增功能
- ✨ **Gradio API 支持**：新增对 Gradio 格式 API 的支持，可以调用 HuggingFace Space 上的图片生成模型
- 🎨 **Z-Image-Turbo 集成**：完整支持通义万相 Z-Image-Turbo 模型
- 🔧 **API 类型配置**：新增 `api_type` 配置项，支持 `openai` 和 `gradio` 两种格式
- 📊 **Gradio 专用参数**：新增 `gradio_resolution`、`gradio_steps`、`gradio_shift`、`gradio_timeout` 配置项

### 技术改进
- 🔄 **双 API 架构**：实现了 OpenAI 格式和 Gradio 格式的双 API 支持
- 🔍 **SSE 解析**：实现了 Gradio Server-Sent Events (SSE) 响应的解析
- ⏱️ **轮询机制**：实现了 Gradio API 的 POST + GET 轮询机制
- 🧪 **测试脚本**：添加了完整的测试脚本（`test_gradio_api.py` 和 `test_integration.py`）

### 文档更新
- 📝 添加了 `README.md` 详细使用文档
- 📋 添加了 `config.example.toml` 配置示例
- 📖 添加了 API 调用文档（`Zimagedoc-curl.md` 和 `zimagedoc-mcp.md`）

### 向后兼容
- ✅ 完全兼容现有的 OpenAI 格式配置
- ✅ 默认使用 OpenAI 格式，不影响现有用户

### 测试结果
- ✅ Gradio API 调用测试通过
- ✅ 图片下载和保存测试通过
- ✅ 集成测试通过
- ✅ 代码语法检查通过

## [2.x] - 之前版本

### 功能
- LLM 智能提示词生成
- OpenAI 格式 API 支持
- 自拍模式
- 图片裁切功能
- 人设集成
