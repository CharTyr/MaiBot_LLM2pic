# LLM2PIC 插件现状

## 背景

LLM2PIC（MaiBot_LLM2pic）是 MaiBot 的图片生成插件，通过 LLM 根据聊天上下文和人设生成生图 API 的提示词，然后调用图片生成 API 生成图片并发送到聊天中。

原仓库：https://github.com/CharTyr/MaiBot_LLM2pic

## 当前架构

已完全裁剪为 **rdev 原生架构**，旧版 `src.plugin_system` 兼容层已全部移除。

| 组件 | 说明 |
|------|------|
| `LLM2PicPlugin` (`MaiBotPlugin`) | rdev 原生入口，通过 `create_plugin()` 工厂加载 |
| `@Action("draw_picture")` | LLM 智能触发 → 内部转为 Tool 组件 |
| `@Command("direct_pic")` | `/pic` 指令直接生图 |
| `_ActionRuntimeProxy(CustomPicAction)` | 代理层，将旧 `CustomPicAction` 的方法桥接到 rdev ctx |
| `_CommandRuntimeProxy(DirectPicCommand)` | 代理层，将旧 `DirectPicCommand` 的方法桥接到 rdev ctx |
| `_RuntimeBridgeMixin` | 提供 `ctx.send`、`ctx.message`、`ctx.llm`、`ctx.config` 等运行时能力封装 |
| `CustomPicAction` | **纯数据+方法类**，保存 action 元数据和生图 API 调用方法 |
| `DirectPicCommand` | **纯数据+方法类**，保存 command 元数据和生图 API 调用方法 |
| `StyleRouter` | 风格路由器，anime/real 双模型调度 |
| `LLMOutputParser` | 解析 LLM 返回的 JSON 格式风格+prompt |

### 已删除的旧代码

- `src.plugin_system` 系 import（`BasePlugin`、`BaseAction`、`BaseCommand`、`register_plugin` 等）
- `src.chat.message_receive.message` import（`MessageRecv`）
- `src.config.config` import（`global_config`、`model_config`）
- `CustomPicPlugin` 类（旧 `@register_plugin` 入口）
- `PromptGenerator` 类（旧 LLM 调用方式，被 `_ctx_generate_prompt_with_style` 替代）
- `CustomPicAction.execute` 旧实现（被 `_ActionRuntimeProxy.execute` 覆盖）
- `CustomPicAction._get_recent_chat_messages`、`_get_persona`、`_get_llm_model_config`
- `DirectPicCommand.execute` 旧实现（被 `_CommandRuntimeProxy` 覆盖）
- `_LLM_JUDGE` 常量、`ChatMode` 引用

## 长耗时问题的解决

### 问题根因

SDK 对 Action/Tool 的 RPC 调用有 **30 秒硬超时**，但生图 API（如 `std.loliyc.com`）的图片生成通常需要 60~180 秒。

### 方案（插件侧，不动主程序）

将 `handle_draw_picture` 和 `handle_direct_pic`（`/pic` 命令）改为**快速返回 + 后台异步**模式：

```
请求进入 → 生成 prompt（<30s）→ 返回 (True, "")
                                  ↓
                         asyncio.create_task
                                  ↓
                         调生图 API（可 >30s）
                                  ↓
                         下载/解码图片
                                  ↓
                         ctx.send.image 发送结果
                         ctx.send.text 发送错误
```

### 重要约束

- **不要动主程序**。主程序的超时配置（`component_query.py`、`supervisor.py` 等）不在插件改动范围。
- 后台 task 通过 `self.ctx.send.image` / `self.ctx.send.text` 发送消息。

## 支持的 API 类型

| api_type | 说明 | 当前使用 |
|----------|------|----------|
| `regex_url` | URL 模板模式，用 `$1` 占位 prompt | **anime 模型在用** |
| `openai` | OpenAI 兼容 `/chat/completions` | 兼容备用 |
| `gradio` | HuggingFace Space Gradio 格式 | 兼容备用 |
| `sd_api` | Stable Diffusion API 格式 | 兼容备用 |
| `novelai` | NovelAI 官方 API | 兼容备用 |

## 配置

配置文件：`config.toml`（同目录），模板：`config.example.toml`

双模型风格：`[anime]`（二次元，自拍强制使用）和 `[real]`（写实，默认未启用）。

## 修改时注意事项

1. **仅改插件文件**，不碰 `src/` 主程序
2. `CustomPicAction` 和 `DirectPicCommand` 必须保留——`_ActionRuntimeProxy` / `_CommandRuntimeProxy` 继承它们复用生图 API 方法（`_make_regex_url_request`、`_crop_image` 等）和元数据（`action_description`、`command_pattern` 等）
3. 所有发送消息/图片通过 `_RuntimeBridgeMixin` 的 `_ctx_*` 方法进行，封装了 `self.ctx.send.*`、`self.ctx.message.*` 等 rdev 原生 API
4. 不要再引入 `src.plugin_system` 或 `src.config.config` 依赖
