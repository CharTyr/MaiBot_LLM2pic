# -*- coding: utf-8 -*-
"""
Character tag validator — runtime Danbooru verification for role tags.

Why this exists
---------------
LLM-generated prompts often contain *wrong* character tags:
  - "zoe (palworld)"            -> canonical is "zoe_rayne_(palworld)"
  - "gothic girl riding cat"    -> natural language, NAI doesn't understand
The semantic tag retriever (DanbooruOnline) only *suggests* candidates; it
never verifies that a character tag actually exists in Danbooru.

This module adds the missing "look it up" step after the LLM generates tags:
  1. extract character-looking tags from the prompt
  2. batch-verify each against Danbooru autocomplete (category=4 = character)
  3. rewrite wrong tags to their canonical form
  4. for low-post-count characters, inject appearance anchors (hair/eye color)
     pulled from a real post's tag_string

Design rules
------------
- NEVER blocks generation: any failure (timeout, network, parse) -> return the
  prompt unchanged.
- Only touches tags that look like character references (`name (series)` or a
  bare word that autocomplete resolves to category=4). General tags like
  "smile", "black hair" are left alone.
- Cheap: one autocomplete call per distinct character tag (sequential, max a
  handful per prompt), one posts.json call only for low-data characters.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.common.logger import get_logger

logger = get_logger("MaiBot_LLM2pic")

# Danbooru autocomplete + posts endpoints (public API, direct connection)
_AUTOCOMPLETE_URL = "https://danbooru.donmai.us/autocomplete.json"
_POSTS_URL = "https://danbooru.donmai.us/posts.json"
_TIMEOUT = 8.0
# Danbooru sits behind Cloudflare with a quirky bot policy: browser-ish UAs
# get 403-challenged, plain `curl/x.y.z` UAs pass. Verified 2026-07-31.
_UA = "curl/8.7.1"

# Characters with fewer posts than this need explicit appearance anchors.
_ANCHOR_THRESHOLD = 300
# Keep at most this many anchor tags per character.
_MAX_ANCHOR_TAGS = 4

# A character tag: "name_(series)" or "name (series)", optionally wrapped in
# NAI weight syntax: {{{name (series)}}}, [name_(series)], {name (series)} ...
# We match the bare word first (no spaces) then handle the (series) part.
_CHAR_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_WEIGHT_WRAP_RE = re.compile(r"^[\[\]{}()]*|[\[\]{}()]*$")

# Hair/eye color tags worth anchoring.
_COLOR_HINT_RE = re.compile(
    r"\b("
    r"black|white|brown|blonde|blond|gray|grey|silver|gold|golden|red|orange|"
    r"pink|purple|blue|green|teal|cyan|aqua|lavender|violet|indigo|"
    r"multicolored|two-tone|gradient"
    r")_hair\b"
    r"|\b("
    r"black|white|brown|amber|red|orange|pink|purple|blue|green|teal|cyan|"
    r"aqua|lavender|violet|indigo|gold|golden|gray|grey|silver|heterochromia"
    r")_eyes\b",
    re.IGNORECASE,
)

# Recognize a character-tag token inside a prompt.
# Handles: zoe_rayne_(palworld), zoe rayne (palworld), {{{zoe_rayne (palworld)}}}
_CHAR_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])"                       # no preceding tag char
    r"([A-Za-z0-9_]+(?:\s*\(\s*[A-Za-z0-9_ ]+\s*\))?)"   # name (series)?
    r"(?![A-Za-z0-9_])"                        # no trailing tag char
)


def _strip_weight_wrappers(tag: str) -> str:
    """Remove NAI weight syntax: {{{azuma_seren}}} -> azuma_seren."""
    t = tag.strip()
    while t and t[0] in "[{(" and t[-1] in "]})":
        inner = t[1:-1]
        if inner and inner != t:
            t = inner
        else:
            break
    return t.strip()


def _normalize(tag: str) -> str:
    """Canonical form: lowercase, spaces -> underscores."""
    return re.sub(r"\s+", "_", _strip_weight_wrappers(tag).strip()).lower()


def extract_character_tokens(prompt: str) -> List[str]:
    """Return distinct character-tag tokens (raw, wrappers stripped, normalized).

    Conservative: only tokens matching `name (series)` shape OR bare words that
    are not obviously general tags (contains no spaces other than series parens).
    """
    seen: List[str] = []
    seen_set = set()
    for m in _CHAR_TOKEN_RE.finditer(prompt):
        raw = m.group(1)
        token = _strip_weight_wrappers(raw).strip()
        norm = _normalize(token)
        if not norm:
            continue
        # Bare single words are suspicious (could be general tags like "smile").
        # Only keep bare words when they look like proper names or known
        # character-style tokens (contains underscore or series parens).
        has_series = "(" in norm
        if not has_series and "_" not in norm:
            continue
        if norm not in seen_set:
            seen_set.add(norm)
            seen.append(norm)
    return seen


async def _autocomplete(query: str, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """Query Danbooru autocomplete; return raw items."""
    try:
        resp = await client.get(
            _AUTOCOMPLETE_URL,
            params={
                "search[query]": query,
                "search[type]": "tag_query",
                "limit": 8,
            },
            headers={"User-Agent": _UA},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
    except Exception as e:
        logger.warning(f"[TagValidator] autocomplete failed for '{query}': {e}")
    return []


def _tag_key(tag: str) -> str:
    """Normalize a tag for comparison: lowercase, drop separators.

    rem_(re:zero) == rem_(re_zero) == rem (re zero) — punctuation-insensitive.
    """
    return re.sub(r"[\s:_\-\.]", "", tag.lower())


async def _resolve_character(query: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Resolve a tag query to its canonical character entry, or None.

    Strategy (avoids autocomplete's fuzzy mis-matches):
      1. split query into (name, series) — "zoe_(palworld)" -> ("zoe", "palworld")
      2. exact match on full query, punctuation-insensitive — this handles the
         common case (autocomplete returns rem_(re:zero) for rem_(re_zero))
      3. else query autocomplete with the bare name, keep only candidates
         whose name starts with "<bare name>_" or equals it (zoe_rayne ✓,
         hange_zoe ✗, remilia_scarlet ✗)
      4. if series present, verify candidate+series combo via posts API
         (e.g. "zoe_rayne palworld" has posts) — prevents cross-character picks
      5. pick highest-post_count verified candidate; else None
    """
    name_part = query
    series_part: Optional[str] = None
    m = re.match(r"^(.+?)_?\(([^)]+)\)$", query)
    if m:
        name_part = m.group(1).strip().strip("_")
        series_part = m.group(2).strip().strip("_")

    # Step 2: exact full-query match (punctuation-insensitive)
    items = await _autocomplete(query, client)
    if items:
        q_key = _tag_key(query)
        for it in items:
            if not isinstance(it, dict):
                continue
            value = str(it.get("value") or "")
            if _tag_key(value) == q_key and it.get("category") == 4:
                return it

    # Step 3: bare-name search, strict prefix match on token boundary
    items = await _autocomplete(name_part, client)
    chars: List[Dict[str, Any]] = []
    n = name_part.lower().replace(" ", "_")
    for it in items:
        if not isinstance(it, dict) or it.get("category") != 4:
            continue
        value = str(it.get("value") or "")
        v = value.lower().replace(" ", "_")
        # zoe_rayne ✓ (starts zoe_), hange_zoe ✗, remilia_scarlet ✗ (not rem_)
        if v == n or v.startswith(n + "_"):
            chars.append(it)
    if not chars:
        return None

    # Step 4: series verification via posts API
    if series_part:
        verified: List[Dict[str, Any]] = []
        for cand in chars:
            cand_name = str(cand.get("value") or "")
            try:
                resp = await client.get(
                    _POSTS_URL,
                    params={"tags": f"{cand_name} {series_part}", "limit": 1},
                    headers={"User-Agent": _UA},
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and data:
                    verified.append(cand)
            except Exception:
                pass
        if verified:
            chars = verified
        else:
            # series combo didn't verify — don't guess across series
            return None

    chars.sort(key=lambda x: int(x.get("post_count") or 0), reverse=True)
    return chars[0]


async def _fetch_anchor_tags(character: str, client: httpx.AsyncClient) -> List[str]:
    """Fetch one real post's tag_string and extract hair/eye anchors."""
    try:
        resp = await client.get(
            _POSTS_URL,
            params={"tags": character, "limit": 1},
            headers={"User-Agent": _UA},
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            return []
        tag_string = str(data[0].get("tag_string") or "")
        anchors: List[str] = []
        for m in _COLOR_HINT_RE.finditer(tag_string):
            tag = m.group(0)
            if tag not in anchors:
                anchors.append(tag)
        return anchors[:_MAX_ANCHOR_TAGS]
    except Exception as e:
        logger.warning(f"[TagValidator] anchor fetch failed for '{character}': {e}")
        return []


def _rewrite_prompt(prompt: str, replacements: Dict[str, str]) -> str:
    """Replace character tokens in the prompt.

    replacements: {normalized_token: canonical_tag}
    NAI weight wrappers ({{{ }}}, [ ], { }) around the token are preserved
    naturally — we only swap the inner token, never the wrapper chars.
    """

    def _replace_match(m: re.Match) -> str:
        raw = m.group(1)
        norm = _normalize(raw)
        if norm not in replacements:
            return m.group(0)
        return replacements[norm]

    return _CHAR_TOKEN_RE.sub(_replace_match, prompt)



def _missing_appearance_anchors(prompt: str, anchors: List[str]) -> List[str]:
    """Skip hair/eye anchors when the prompt already describes appearance.

    Danbooru low-post samples often carry the wrong hair color for OC / 换角.
    """
    raw = str(prompt or "")
    norm = raw.lower().replace("-", "_")
    compact = re.sub(r"\s+", "_", norm)
    has_hair = bool(
        _COLOR_HINT_RE.search(compact)
        or re.search(r"twin\s*tails|twintails|ponytail|\bbraid", norm)
    )
    has_eyes = bool(re.search(r"[a-z]+_eyes|\b[a-z]+\s+eyes\b", norm))
    missing: List[str] = []
    for anchor in anchors:
        item = str(anchor or "").strip()
        if not item:
            continue
        low = item.lower()
        if "_hair" in low and has_hair:
            continue
        if "_eyes" in low and has_eyes:
            continue
        if low in norm or low.replace("_", " ") in norm:
            continue
        missing.append(item)
    return missing


class CharacterTagValidator:
    """Runtime Danbooru verification for character tags in generated prompts."""

    def __init__(self, anchor_threshold: int = _ANCHOR_THRESHOLD):
        self.anchor_threshold = anchor_threshold

    async def validate_prompt(self, prompt: str) -> str:
        """Verify + fix character tags in a prompt. Never raises."""
        if not prompt or not prompt.strip():
            return prompt
        try:
            tokens = extract_character_tokens(prompt)
            if not tokens:
                return prompt

            replacements: Dict[str, str] = {}
            low_data: Dict[str, str] = {}

            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                for tok in tokens:
                    resolved = await _resolve_character(tok, client)
                    if not resolved:
                        logger.info(f"[TagValidator] no match for '{tok}', leaving as-is")
                        continue
                    canonical = str(resolved.get("value") or tok).replace(" ", "_")
                    post_count = int(resolved.get("post_count") or 0)
                    if canonical == tok and post_count >= self.anchor_threshold:
                        continue  # already correct, no anchor needed
                    replacements[tok] = canonical
                    if post_count < self.anchor_threshold:
                        low_data[tok] = canonical
                    logger.info(
                        f"[TagValidator] '{tok}' -> '{canonical}' (posts={post_count})"
                    )

            if not replacements:
                return prompt

            new_prompt = _rewrite_prompt(prompt, replacements)

            # Inject appearance anchors for low-data characters (skip if the
            # anchor tag is already present in the prompt).
            if low_data:
                async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                    for canonical in low_data.values():
                        anchors = await _fetch_anchor_tags(canonical, client)
                        if not anchors:
                            continue
                        missing = _missing_appearance_anchors(new_prompt, anchors)
                        if missing:
                            new_prompt = f"{new_prompt}, {', '.join(missing)}"
                            logger.info(
                                f"[TagValidator] anchored '{canonical}' with {missing}"
                            )
            return new_prompt
        except Exception as e:
            logger.warning(f"[TagValidator] validation skipped (never blocks): {e}")
            return prompt


# Module-level cached validator
_validator: Optional[CharacterTagValidator] = None


def get_character_tag_validator(config: Optional[dict] = None) -> CharacterTagValidator:
    """Return a shared validator instance."""
    global _validator
    if _validator is None:
        threshold = _ANCHOR_THRESHOLD
        if config:
            try:
                threshold = int(
                    (config.get("tag_validator") or {}).get("anchor_threshold", _ANCHOR_THRESHOLD)
                )
            except (TypeError, ValueError):
                threshold = _ANCHOR_THRESHOLD
        _validator = CharacterTagValidator(anchor_threshold=threshold)
    return _validator
