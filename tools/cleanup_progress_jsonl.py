from __future__ import annotations

"""
清洗 notes_mineru/_progress.jsonl：
- 保证一行一个 JSON（jsonl）
- 统一字段：ts/paper_id/status/note/raw/model/base_url/error
- error 压成单行（把换行写成 \\n），避免编辑器显示混乱

用法（项目根目录执行）：
  python tools/cleanup_progress_jsonl.py
"""

import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _one_line(s: str, limit: int = 2000) -> str:
    t = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\n", "\\n").replace("\t", "\\t")
    t = t.strip()
    if len(t) > limit:
        t = t[:limit] + "…(truncated)"
    return t


def _safe_get(d: dict[str, Any], k: str) -> str:
    v = d.get(k, "")
    return "" if v is None else str(v)


def main() -> int:
    out_dir = REPO_ROOT / "notes_mineru"
    p = out_dir / "_progress.jsonl"
    if not p.exists():
        print(f"not found: {p}")
        return 0

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    cleaned: list[dict[str, Any]] = []

    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        obj: dict[str, Any] | None = None
        try:
            x = json.loads(s)
            if isinstance(x, dict):
                obj = x
        except Exception:
            obj = {
                "ts": "",
                "paper_id": "",
                "status": "failed",
                "note": "",
                "raw": "",
                "model": "",
                "base_url": "",
                "error": _one_line(s),
            }
        assert obj is not None

        rec = {
            "ts": _safe_get(obj, "ts"),
            "paper_id": _safe_get(obj, "paper_id"),
            "status": _safe_get(obj, "status") or "failed",
            "note": _safe_get(obj, "note"),
            "raw": _safe_get(obj, "raw"),
            "model": _safe_get(obj, "model"),
            "base_url": _safe_get(obj, "base_url"),
            "error": _one_line(_safe_get(obj, "error")),
        }
        # status 归一化
        if rec["status"] not in ("ok", "failed"):
            rec["status"] = "failed"
        cleaned.append(rec)

    p.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in cleaned).rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"cleaned: {p} ({len(cleaned)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


