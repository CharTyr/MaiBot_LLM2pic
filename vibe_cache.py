"""
Vibe Transfer cache_id 复用。

API 文档 §20.3.1：每次 vibe 编码成功后，网关返回 cache_id（22字符 URL-safe base64）。
下次同一张图 + 同 model + 同 info_extracted 的请求可改用 cache_id 代替图片字节，
跳过上传与编码计费（省 1 anlas 附加费）。

缓存 key: (image_sha256, model, info_extracted量化到0.01粒度)
缓存 value: cache_id 字符串
存储: SQLite，插件 data 目录下
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from src.common.logger import get_logger

logger = get_logger("MaiBot_LLM2pic")

# Cache 过期时间：7天（服务端 cache_id 可能更短，但过期后 API 返回 400 时自动清）
_CACHE_TTL_SECONDS = 7 * 24 * 3600


def _get_db_path() -> Path:
    """获取插件 data 目录下的 vibe_cache.db 路径。"""
    # MaiBot 插件 data 目录约定
    base = Path(os.environ.get("MAIBOT_DATA_DIR", ""))
    if not base:
        # 兜底：用插件目录下的 data
        base = Path(__file__).resolve().parent.parent / "data"
    db_dir = base / "plugins" / "chartyr.maibot-llm2pic"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "vibe_cache.db"


def _quantize_info_extracted(value: float) -> float:
    """量化 info_extracted 到 0.01 粒度（API 文档要求）。"""
    return round(float(value), 2)


def _image_sha256(image_data: str) -> str:
    """计算图片数据的 SHA-256（支持 data URI 和纯 base64）。"""
    import base64
    # 去掉 data:image/...;base64, 前缀
    raw = image_data
    if "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        decoded = base64.b64decode(raw)
    except Exception:
        decoded = raw.encode("utf-8")
    return hashlib.sha256(decoded).hexdigest()


class VibeCache:
    """Vibe Transfer cache_id 本地缓存。"""

    def __init__(self):
        self._db_path = _get_db_path()
        self._init_db()

    def _init_db(self):
        """初始化 SQLite 表。"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vibe_cache (
                image_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                info_extracted REAL NOT NULL,
                cache_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (image_hash, model, info_extracted)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vibe_lookup
            ON vibe_cache(image_hash, model, info_extracted)
        """)
        conn.commit()
        conn.close()

    def lookup(
        self,
        image_data: str,
        model: str,
        info_extracted: float,
    ) -> Optional[str]:
        """查缓存。命中返回 cache_id，未命中返回 None。"""
        img_hash = _image_sha256(image_data)
        ie = _quantize_info_extracted(info_extracted)
        model_lower = str(model or "").lower().strip()

        conn = sqlite3.connect(str(self._db_path))
        row = conn.execute(
            "SELECT cache_id, created_at FROM vibe_cache "
            "WHERE image_hash=? AND model=? AND info_extracted=?",
            (img_hash, model_lower, ie),
        ).fetchone()
        conn.close()

        if row is None:
            return None

        cache_id, created_at = row
        if time.time() - created_at > _CACHE_TTL_SECONDS:
            logger.debug(f"[VibeCache] expired: hash={img_hash[:12]}, age={int(time.time() - created_at)}s")
            self._delete(img_hash, model_lower, ie)
            return None

        logger.info(f"[VibeCache] hit: hash={img_hash[:12]}, model={model_lower}, ie={ie}")
        return cache_id

    def store(
        self,
        image_data: str,
        model: str,
        info_extracted: float,
        cache_id: str,
    ) -> None:
        """存缓存。"""
        if not cache_id:
            return
        img_hash = _image_sha256(image_data)
        ie = _quantize_info_extracted(info_extracted)
        model_lower = str(model or "").lower().strip()

        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            "INSERT OR REPLACE INTO vibe_cache "
            "(image_hash, model, info_extracted, cache_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (img_hash, model_lower, ie, cache_id, time.time()),
        )
        conn.commit()
        conn.close()
        logger.info(f"[VibeCache] stored: hash={img_hash[:12]}, cache_id={cache_id[:8]}...")

    def _delete(self, img_hash: str, model: str, info_extracted: float) -> None:
        """删除过期条目。"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            "DELETE FROM vibe_cache "
            "WHERE image_hash=? AND model=? AND info_extracted=?",
            (img_hash, model, info_extracted),
        )
        conn.commit()
        conn.close()

    def cleanup_expired(self) -> int:
        """清理所有过期条目。返回清理数量。"""
        cutoff = time.time() - _CACHE_TTL_SECONDS
        conn = sqlite3.connect(str(self._db_path))
        cur = conn.execute(
            "DELETE FROM vibe_cache WHERE created_at < ?",
            (cutoff,),
        )
        count = cur.rowcount
        conn.commit()
        conn.close()
        if count > 0:
            logger.info(f"[VibeCache] cleaned {count} expired entries")
        return count