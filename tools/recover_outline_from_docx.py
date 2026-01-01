from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _guess_default_docx() -> Path:
    # 默认优先：项目根目录（更可复现）
    cand_repo = REPO_ROOT / "目录大纲.docx"
    if cand_repo.exists():
        return cand_repo

    # fallback：桌面论文目录
    desktop = Path.home() / "Desktop"
    return desktop / "论文" / "目录大纲.docx"


def _is_heading_style(style_name: str) -> int | None:
    """
    识别 Word 的 Heading 1/2/3…，返回级别；否则 None。
    """
    s = (style_name or "").strip().lower()
    if s.startswith("heading"):
        parts = s.split()
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
    # 中文环境常见：标题 1/2/3
    if s.startswith("标题"):
        num = s.replace("标题", "").strip()
        if num.isdigit():
            return int(num)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Recover outline: docx -> Markdown headings")
    ap.add_argument("--docx", default=str(_guess_default_docx()), help="input docx path")
    ap.add_argument("--out", default="目录大纲.md", help="output md path (relative to repo root)")
    args = ap.parse_args()

    in_path = Path(args.docx).expanduser().resolve()
    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from docx import Document  # type: ignore
    except Exception as e:
        raise SystemExit(f"缺少依赖 python-docx。请运行：pip install python-docx ({type(e).__name__}: {e})")

    if not in_path.exists():
        raise SystemExit(f"未找到 docx：{in_path}")

    doc = Document(str(in_path))

    lines: list[str] = []
    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue
        lvl = _is_heading_style(getattr(p.style, "name", "") if getattr(p, "style", None) else "")
        if lvl is None:
            # 普通段落
            lines.append(text)
            continue
        lvl = max(1, min(6, lvl))
        lines.append("#" * lvl + " " + text)

    out_path.write_text("\n\n".join(lines).strip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



