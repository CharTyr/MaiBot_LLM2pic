# MaiBot_LLM2pic 开发文档

## 架构概览

```
plugin.py
├── StyleRouter          # 风格路由器，决定使用 anime/real 模型
├── LLMOutputParser      # 解析 LLM 返回的 JSON 格式输出
├── PromptGenerator      # 调用 LLM 生成图片提示词
├── CustomPicAction      # Action 组件，LLM 智能触发图片生成
├── DirectPicCommand     # Command 组件，/pic 指令直接生图
└── CustomPicPlugin      # 插件主类，注册组件和配置
```

## 核心类说明

### StyleRouter

风格路由器，根据优先级决定使用哪个模型：

```
selfie_mode > manual_style > llm_style > default_style
```

```python
router = StyleRouter(self.plugin_config)
style, model_config, reason = router.route(
    selfie_mode=False,
    manual_style="anime",  # 手动指定
    llm_style=None,
)
```

### LLMOutputParser

解析 LLM 返回的 JSON：

```python
success, prompt, style = LLMOutputParser.parse(llm_response)
# prompt: "1girl, solo, ..."
# style: "anime" 或 "real"
```

### PromptGenerator

调用 LLM 生成提示词：

```python
success, prompt, style = await PromptGenerator.generate_prompt_with_style(
    chat_messages="最近的聊天记录",
    user_request="画一只猫",
    persona="角色人设",
    selfie_mode=False,
    custom_system_prompt="",  # 可选自定义提示词
    model_name="",  # 可选指定模型
)
```

### CustomPicAction

继承 `BaseAction`，通过 LLM 判断是否触发。

关键属性：
- `activation_type = ActionActivationType.LLM_JUDGE`
- `action_name = "draw_picture"`
- `llm_judge_prompt`: 触发条件描述

关键方法：
- `execute()`: 执行图片生成流程
- `_make_gradio_image_request()`: Gradio API 调用
- `_make_sd_api_request()`: SD API 调用
- `_make_http_image_request()`: OpenAI 格式 API 调用
- `_handle_image_result()`: 处理图片结果（下载/裁切/编码）

### DirectPicCommand

继承 `BaseCommand`，通过正则匹配触发。

```python
command_pattern = r"^/pic\s+(?:(?P<style>anime|real)\s+)?(?P<prompt>.+)$"
```

支持：
- `/pic <prompt>`
- `/pic anime <prompt>`
- `/pic real <prompt>`

## 支持的 API 类型

### 1. Gradio (`api_type = "gradio"`)

流程：POST 获取 event_id → GET 轮询 SSE 结果

```python
def _make_gradio_image_request(self, prompt, base_url, gradio_params):
    # POST /gradio_api/call/generate
    # GET /gradio_api/call/generate/{event_id}
```

### 2. SD API (`api_type = "sd_api"`)

直接 POST 请求，返回 `image_url`。

```python
def _make_sd_api_request(self, prompt, base_url, api_key, sd_params):
    # POST /api/v1/generate_image
    # 响应: {"data": {"image_url": "..."}}
```

注意事项：
- 需要添加 `User-Agent` 头绕过 Cloudflare
- 响应字段是 `data.image_url`

### 3. OpenAI (`api_type = "openai"`)

兼容 OpenAI `/chat/completions` 格式。

```python
def _make_http_image_request(self, prompt, model, size, base_url, api_key):
    # POST /chat/completions
```

## 配置获取

使用 `self.get_config()` 或 `self.plugin_config`：

```python
# 获取单个配置
api_type = self.get_config("anime.api_type", "openai")

# 传递给 StyleRouter
router = StyleRouter(self.plugin_config)
```

## 添加新 API 类型

1. 在 `StyleRouter._extract_model_config()` 添加新参数
2. 在 `CustomPicAction.execute()` 添加新的 API 调用分支
3. 实现 `_make_xxx_request()` 方法
4. 在 `DirectPicCommand.execute()` 同步添加
5. 更新 `CustomPicPlugin.get_config_fields()` 添加配置字段

## 图片处理流程

```
API 返回 → 判断类型（base64/URL）→ 下载（如果是URL）→ 裁切（可选）→ Base64 编码 → 发送
```

```python
async def _handle_image_result(self, success, result):
    if result.startswith(("iVBORw", "/9j/")):  # base64
        # 直接使用
    else:  # URL
        # 下载并编码
        encode_success, base64_data = await asyncio.to_thread(
            self._download_and_encode_base64, result
        )
```

## 常见问题

### Cloudflare 403

添加 User-Agent 头：

```python
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...",
}
```

### self.config vs self.plugin_config

MaiBot 插件系统使用 `self.plugin_config`，不是 `self.config`。

### 花括号转义

`system_prompt` 中如果有 `{{{xxx}}}`，不能用 `.format()`，要用 `.replace()`：

```python
# 错误
system_prompt = base_prompt.format(persona=persona)

# 正确
system_prompt = base_prompt.replace("{persona}", persona)
```

## 测试

```bash
# 测试 SD API
curl -X POST "https://sd.exacg.cc/api/v1/generate_image" \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cute cat", "width": 512, "height": 512}'
```
