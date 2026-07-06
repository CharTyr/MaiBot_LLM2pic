"""将生成的图片上传到 GitHub 指定仓库。

使用 GitHub Contents API（https://docs.github.com/rest/repos/contents），
通过标准库 urllib 发送 PUT 请求，把 base64 图片写入仓库。

设计目标：
- 不阻塞主生图/发送流程，由调用方以 ``asyncio.create_task`` 调度。
- 失败仅记录日志，不影响图片发送。
- 路径规则：``<path_prefix>/<YYYY-MM-DD>/<时间戳>.<ext>``，按日期分文件夹。
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Tuple

from src.common.logger import get_logger

logger = get_logger("MaiBot_LLM2pic")

# base64 magic 前缀 -> (扩展名, mime)
_MAGIC_PREFIXES: Tuple[Tuple[str, str, str], ...] = (
    ("iVBORw", "png", "image/png"),
    ("/9j/", "jpg", "image/jpeg"),
    ("UklGR", "webp", "image/webp"),
    ("R0lGOD", "gif", "image/gif"),
)


def _guess_ext(image_base64: str) -> str:
    """根据 base64 前缀推断图片扩展名，默认 png。"""
    head = (image_base64 or "")[:8]
    for prefix, ext, _ in _MAGIC_PREFIXES:
        if head.startswith(prefix):
            return ext
    return "png"


def _build_path(path_prefix: str, image_base64: str) -> str:
    """构造仓库内路径：``<prefix>/<YYYY-MM-DD>/<时间戳>_<随机>.<ext>``。"""
    prefix = (path_prefix or "images").strip("/")
    date_str = time.strftime("%Y-%m-%d", time.localtime())
    ext = _guess_ext(image_base64)
    ts = int(time.time())
    import random
    suffix = f"{random.randint(0, 0xFFFFFFFF):08x}"
    filename = f"{ts}_{suffix}.{ext}"
    return f"{prefix}/{date_str}/{filename}"


def _upload_to_github_sync(
    image_base64: str,
    *,
    token: str,
    owner: str,
    repo: str,
    path_prefix: str,
    branch: str,
    commit_message: str,
) -> Tuple[bool, str]:
    """同步上传单张图片到 GitHub 仓库，返回 (success, message)。"""
    if not token:
        return False, "github.token 未配置"
    if not owner or not repo:
        return False, "github.owner/repo 未配置"

    path = _build_path(path_prefix, image_base64)
    # Contents API 要求 content 为 base64 编码后的文件内容（字符串）
    # image_base64 已经是 base64 字符串，可直接使用
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{urllib.parse.quote(path, safe='/')}"

    payload = {
        "message": commit_message or f"upload image {path}",
        "content": image_base64,
    }
    if branch:
        payload["branch"] = branch

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "MaiBot-LLM2pic",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        return False, f"HTTP {exc.code}: {detail[:200]}"
    except Exception as exc:
        return False, f"{exc!r}"

    if status in (200, 201):
        return True, path
    return False, f"意外状态码 {status}: {raw[:200].decode('utf-8', errors='ignore')}"


def _extract_git_token_from_regex_url(get_config) -> str:
    """从 ``anime.regex_url.base_url`` 的 ``git_token`` 查询参数回退提取 token。

    regex_url 端点历史上有 ``git_token=<token>&git_repo=CharTyr/my-images`` 参数，
    复用同一 token 避免重复配置。若 regex_url 未配置或无该参数，返回空字符串。
    """
    try:
        base_url = str(get_config("anime.regex_url.base_url", "") or "")
        if not base_url:
            return ""
        parsed = urllib.parse.urlparse(base_url)
        if not parsed.query:
            return ""
        params = urllib.parse.parse_qs(parsed.query)
        values = params.get("git_token") or []
        return str(values[0]).strip() if values else ""
    except Exception:
        return ""


def _extract_git_repo_from_regex_url(get_config) -> Tuple[str, str]:
    """从 ``anime.regex_url.base_url`` 的 ``git_repo`` 参数回退提取 (owner, repo)。"""
    try:
        base_url = str(get_config("anime.regex_url.base_url", "") or "")
        if not base_url:
            return "", ""
        parsed = urllib.parse.urlparse(base_url)
        if not parsed.query:
            return "", ""
        params = urllib.parse.parse_qs(parsed.query)
        values = params.get("git_repo") or []
        if not values:
            return "", ""
        repo_str = str(values[0]).strip()
        if "/" not in repo_str:
            return "", ""
        owner, repo = repo_str.split("/", 1)
        return owner.strip(), repo.strip()
    except Exception:
        return "", ""


async def upload_image_to_github(image_base64: str, *, get_config, prompt: str = "") -> None:
    """异步上传入口：读取配置并上传，失败仅记录日志。

    :param image_base64: 纯 base64 字符串（无 data: 前缀）
    :param get_config: 可调用对象，``get_config(path, default)`` 读取插件配置
    :param prompt: 生成该图片时使用的提示词/tag，作为 commit message 保存，
                   前端可据此展示该图的 tag 信息

    配置回退顺序：
    - ``github.token`` 为空时，从 ``anime.regex_url.base_url`` 的 ``git_token`` 参数提取
    - ``github.owner``/``github.repo`` 为默认值时，从 ``git_repo`` 参数提取
    """
    try:
        enabled = bool(get_config("github.enabled", False))
        if not enabled:
            return
        token = str(get_config("github.token", "") or "")
        # 回退：从 regex_url 的 git_token 参数复用同一 token
        if not token:
            token = _extract_git_token_from_regex_url(get_config)

        owner = str(get_config("github.owner", "CharTyr") or "CharTyr")
        repo = str(get_config("github.repo", "my-images") or "my-images")
        # 回退：若仍为默认值，尝试从 regex_url 的 git_repo 参数提取
        if owner == "CharTyr" and repo == "my-images":
            fb_owner, fb_repo = _extract_git_repo_from_regex_url(get_config)
            if fb_owner and fb_repo:
                owner, repo = fb_owner, fb_repo

        path_prefix = str(get_config("github.path_prefix", "images") or "images")
        branch = str(get_config("github.branch", "main") or "main")
        # commit message 优先用 prompt（tag），其次用配置的自定义 message
        commit_message = prompt or str(get_config("github.commit_message", "") or "")

        import asyncio

        success, message = await asyncio.to_thread(
            _upload_to_github_sync,
            image_base64,
            token=token,
            owner=owner,
            repo=repo,
            path_prefix=path_prefix,
            branch=branch,
            commit_message=commit_message,
        )
        if success:
            logger.info(f"[GitHubUploader] 已上传: {message}")
        else:
            logger.warning(f"[GitHubUploader] 上传失败: {message}")
    except Exception as exc:
        logger.warning(f"[GitHubUploader] 上传异常: {exc!r}", exc_info=True)